"""
TEFAS Fund Analysis - Module 2: Correlation & Diversification Map
===================================================================
CRISP-DM: Modeling + Evaluation Phase

This module reveals hidden correlations and diversification illusions:
1. Traditional correlation matrix analysis
2. Autoencoder-based embedding extraction
3. 2D latent space visualization
4. Cluster detection in risk space

Key Insight:
Funds that appear different by name/category may cluster together
in the latent space, revealing hidden correlation structures.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import EMBEDDING_AE_CONFIG, OUTPUT_FIGURES
from src.models.autoencoder import create_embedding_autoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationMapper:
    """
    Maps hidden correlations and identifies diversification illusions.
    
    Workflow:
    1. Compute traditional correlation matrix
    2. Train autoencoder on return time series
    3. Extract 2D embeddings from bottleneck layer
    4. Cluster funds in embedding space
    5. Compare apparent vs actual diversification
    """
    
    def __init__(
        self,
        ae_config: dict = None,
        n_clusters: int = 5,
    ):
        """
        Initialize CorrelationMapper.
        
        Args:
            ae_config: Autoencoder configuration
            n_clusters: Number of clusters for grouping
        """
        self.ae_config = ae_config or EMBEDDING_AE_CONFIG.copy()
        self.n_clusters = n_clusters
        
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.correlation_matrix = None
        self.embeddings = None
        self.clusters = None
        self.results = None
    
    def compute_correlation_matrix(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix.
        
        Args:
            returns: Returns DataFrame (dates x funds)
            
        Returns:
            Correlation matrix DataFrame
        """
        self.correlation_matrix = returns.corr()
        
        logger.info(f"Computed correlation matrix: {self.correlation_matrix.shape}")
        
        return self.correlation_matrix
    
    def find_highly_correlated_pairs(
        self,
        threshold: float = 0.8,
    ) -> pd.DataFrame:
        """
        Find fund pairs with correlation above threshold.
        
        Args:
            threshold: Minimum correlation to report
            
        Returns:
            DataFrame with correlated pairs
        """
        if self.correlation_matrix is None:
            raise ValueError("Compute correlation matrix first")
        
        pairs = []
        corr = self.correlation_matrix
        
        for i, fund1 in enumerate(corr.columns):
            for j, fund2 in enumerate(corr.columns):
                if i < j:  # Upper triangle only
                    correlation = corr.loc[fund1, fund2]
                    if abs(correlation) >= threshold:
                        pairs.append({
                            "fund1": fund1,
                            "fund2": fund2,
                            "correlation": correlation,
                        })
        
        pairs_df = pd.DataFrame(pairs)
        if len(pairs_df) > 0:
            pairs_df = pairs_df.sort_values("correlation", ascending=False)
        
        logger.info(f"Found {len(pairs_df)} pairs with correlation >= {threshold}")
        
        return pairs_df
    
    def compute_embeddings(
        self,
        returns: pd.DataFrame,
        epochs: int = None,
        verbose: int = 1,
    ) -> np.ndarray:
        """
        Train autoencoder and extract 2D embeddings.
        
        Each fund's entire return history is compressed to 2D.
        
        Args:
            returns: Returns DataFrame (dates x funds)
            epochs: Training epochs
            verbose: Verbosity
            
        Returns:
            2D embedding array (funds x 2)
        """
        # Transpose: we want funds as samples, time as features
        X = returns.T.values  # Shape: (n_funds, n_timesteps)
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train autoencoder
        self.ae_config["input_dim"] = X_scaled.shape[1]
        self.autoencoder = create_embedding_autoencoder(
            input_dim=X_scaled.shape[1],
            config=self.ae_config,
        )
        
        self.autoencoder.fit(
            X_scaled,
            epochs=epochs or self.ae_config.get("epochs", 150),
            batch_size=self.ae_config.get("batch_size", 16),
            validation_split=self.ae_config.get("validation_split", 0.2),
            early_stopping_patience=self.ae_config.get("early_stopping_patience", 15),
            verbose=verbose,
        )
        
        # Extract embeddings
        self.embeddings = self.autoencoder.get_embeddings(X_scaled)
        
        logger.info(f"Extracted {self.embeddings.shape[1]}D embeddings for {len(self.embeddings)} funds")
        
        return self.embeddings
    
    def cluster_funds(
        self,
        embeddings: np.ndarray = None,
    ) -> np.ndarray:
        """
        Cluster funds in embedding space.
        
        Args:
            embeddings: Embedding array (uses stored if None)
            
        Returns:
            Cluster labels
        """
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            raise ValueError("Compute embeddings first")
        
        self.clusters = self.kmeans.fit_predict(embeddings)
        
        logger.info(f"Clustered funds into {self.n_clusters} groups")
        
        return self.clusters
    
    def build_results(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build comprehensive results DataFrame.
        
        Args:
            returns: Original returns DataFrame
            
        Returns:
            Results DataFrame with embeddings and clusters
        """
        fund_codes = returns.columns.tolist()
        
        results = pd.DataFrame({
            "fund_code": fund_codes,
            "embedding_x": self.embeddings[:, 0],
            "embedding_y": self.embeddings[:, 1],
            "cluster": self.clusters,
        })
        results = results.set_index("fund_code")
        
        self.results = results
        
        return results
    
    def identify_diversification_illusions(
        self,
        fund_metadata: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Identify funds that appear different but cluster together.
        
        This reveals "diversification illusions" - funds that seem
        independent but actually move together in risk space.
        
        Args:
            fund_metadata: Optional metadata with fund categories
            
        Returns:
            DataFrame with cluster analysis
        """
        if self.results is None:
            raise ValueError("Build results first")
        
        cluster_analysis = []
        
        for cluster_id in range(self.n_clusters):
            cluster_funds = self.results[self.results["cluster"] == cluster_id]
            
            # Get pairwise correlations within cluster
            if self.correlation_matrix is not None:
                cluster_codes = cluster_funds.index.tolist()
                cluster_corr = self.correlation_matrix.loc[cluster_codes, cluster_codes]
                avg_intra_correlation = cluster_corr.values[
                    np.triu_indices(len(cluster_codes), k=1)
                ].mean()
            else:
                avg_intra_correlation = np.nan
            
            cluster_analysis.append({
                "cluster_id": cluster_id,
                "n_funds": len(cluster_funds),
                "fund_codes": list(cluster_funds.index),
                "avg_intra_correlation": avg_intra_correlation,
                "centroid_x": cluster_funds["embedding_x"].mean(),
                "centroid_y": cluster_funds["embedding_y"].mean(),
            })
        
        analysis_df = pd.DataFrame(cluster_analysis)
        
        # Flag clusters with high internal correlation as potential illusions
        if not analysis_df["avg_intra_correlation"].isna().all():
            threshold = analysis_df["avg_intra_correlation"].median()
            analysis_df["high_correlation_cluster"] = (
                analysis_df["avg_intra_correlation"] > threshold
            )
        
        logger.info("Diversification illusion analysis complete")
        
        return analysis_df
    
    def run(
        self,
        returns: pd.DataFrame,
        epochs: int = None,
        verbose: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full correlation mapping pipeline.
        
        Args:
            returns: Returns DataFrame
            epochs: Autoencoder training epochs
            verbose: Verbosity
            
        Returns:
            Tuple of (fund results, cluster analysis)
        """
        # Step 1: Traditional correlation
        self.compute_correlation_matrix(returns)
        
        # Step 2: Compute embeddings
        self.compute_embeddings(returns, epochs=epochs, verbose=verbose)
        
        # Step 3: Cluster
        self.cluster_funds()
        
        # Step 4: Build results
        results = self.build_results(returns)
        
        # Step 5: Analyze diversification illusions
        cluster_analysis = self.identify_diversification_illusions()
        
        return results, cluster_analysis
    
    def get_cluster_members(
        self,
        cluster_id: int,
    ) -> pd.Index:
        """Get funds in a specific cluster."""
        if self.results is None:
            raise ValueError("Run analysis first")
        
        return self.results[self.results["cluster"] == cluster_id].index
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.results is None:
            return "No results available. Run correlation mapping first."
        
        lines = [
            "=" * 60,
            "CORRELATION & DIVERSIFICATION MAP SUMMARY",
            "=" * 60,
            "",
            f"Total funds analyzed: {len(self.results)}",
            f"Clusters identified: {self.n_clusters}",
            "",
            "Cluster Distribution:",
        ]
        
        for cluster_id in range(self.n_clusters):
            count = (self.results["cluster"] == cluster_id).sum()
            funds = self.results[self.results["cluster"] == cluster_id].index.tolist()
            lines.append(f"  Cluster {cluster_id}: {count} funds")
            if len(funds) <= 5:
                lines.append(f"    Funds: {', '.join(funds)}")
            else:
                lines.append(f"    Funds: {', '.join(funds[:5])}...")
        
        if self.correlation_matrix is not None:
            high_corr = self.find_highly_correlated_pairs(threshold=0.8)
            lines.extend([
                "",
                f"Highly correlated pairs (>0.8): {len(high_corr)}",
            ])
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)


def compute_correlation_map(
    returns: pd.DataFrame,
    n_clusters: int = 5,
    verbose: int = 1,
) -> Tuple[CorrelationMapper, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run correlation mapping.
    
    Args:
        returns: Returns DataFrame
        n_clusters: Number of clusters
        verbose: Verbosity
        
    Returns:
        Tuple of (mapper, results, cluster_analysis)
    """
    mapper = CorrelationMapper(n_clusters=n_clusters)
    results, cluster_analysis = mapper.run(returns, verbose=verbose)
    
    print(mapper.summary())
    
    return mapper, results, cluster_analysis


if __name__ == "__main__":
    print("Correlation Mapper module loaded")
    print("\nUsage:")
    print("  mapper, results, clusters = compute_correlation_map(returns)")
