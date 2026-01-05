"""
TEFAS Fund Analysis - Module 1: Risk Profile Extractor
========================================================
CRISP-DM: Modeling + Evaluation Phase

This module orchestrates:
1. Risk feature calculation
2. Autoencoder-based anomaly detection
3. Risk segmentation (Low / Medium / High)

Output:
- Quantitative risk scores
- Risk segment labels
- Anomaly flags for funds with unusual behavior
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    ANOMALY_PERCENTILE_THRESHOLD,
    DATA_PROCESSED,
    OUTPUT_FIGURES,
    RISK_AE_CONFIG,
    RISK_SEGMENT_THRESHOLDS,
    VOLATILITY_WINDOWS,
)
from src.features.risk_metrics import calculate_all_risk_features, segment_by_risk
from src.models.autoencoder import create_risk_autoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskProfileExtractor:
    """
    Extracts risk profiles and detects anomalies in fund behavior.
    
    Workflow:
    1. Calculate risk features from returns and prices
    2. Standardize features for autoencoder
    3. Train autoencoder and compute reconstruction errors
    4. Identify anomalies (high reconstruction error)
    5. Segment funds by composite risk score
    """
    
    def __init__(
        self,
        volatility_windows: list = None,
        ae_config: dict = None,
        anomaly_threshold: float = None,
        segment_thresholds: dict = None,
    ):
        """
        Initialize RiskProfileExtractor.
        
        Args:
            volatility_windows: Windows for volatility calculation
            ae_config: Autoencoder configuration
            anomaly_threshold: Percentile threshold for anomalies
            segment_thresholds: Thresholds for risk segmentation
        """
        self.volatility_windows = volatility_windows or VOLATILITY_WINDOWS
        self.ae_config = ae_config or RISK_AE_CONFIG.copy()
        self.anomaly_threshold = anomaly_threshold or ANOMALY_PERCENTILE_THRESHOLD
        self.segment_thresholds = segment_thresholds or RISK_SEGMENT_THRESHOLDS
        
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.feature_names = None
        self.results = None
    
    def extract_features(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate all risk features.
        
        Args:
            returns: Returns DataFrame (dates x funds)
            prices: Prices DataFrame (dates x funds)
            
        Returns:
            Feature DataFrame (funds x features)
        """
        features = calculate_all_risk_features(
            returns=returns,
            prices=prices,
            volatility_windows=self.volatility_windows,
        )
        
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Extracted {len(self.feature_names)} features for {len(features)} funds")
        
        return features
    
    def fit_autoencoder(
        self,
        features: pd.DataFrame,
        epochs: int = None,
        verbose: int = 1,
    ) -> None:
        """
        Fit autoencoder for anomaly detection.
        
        Args:
            features: Feature DataFrame
            epochs: Training epochs (uses config default if None)
            verbose: Training verbosity
        """
        # Handle missing values
        features_clean = features.dropna()
        if len(features_clean) < len(features):
            dropped = len(features) - len(features_clean)
            logger.warning(f"Dropped {dropped} funds with missing features")
        
        # Standardize features
        X = self.scaler.fit_transform(features_clean)
        
        # Create and train autoencoder
        self.ae_config["input_dim"] = X.shape[1]
        self.autoencoder = create_risk_autoencoder(
            input_dim=X.shape[1],
            config=self.ae_config,
        )
        
        self.autoencoder.fit(
            X,
            epochs=epochs or self.ae_config.get("epochs", 100),
            batch_size=self.ae_config.get("batch_size", 32),
            validation_split=self.ae_config.get("validation_split", 0.2),
            early_stopping_patience=self.ae_config.get("early_stopping_patience", 10),
            verbose=verbose,
        )
        
        logger.info("Autoencoder training complete")
    
    def compute_risk_scores(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute composite risk scores and detect anomalies.
        
        Risk score combines:
        - Volatility (higher = riskier)
        - Max drawdown (larger = riskier)
        - Reconstruction error (higher = anomalous)
        
        Args:
            features: Feature DataFrame
            
        Returns:
            DataFrame with risk scores and analysis
        """
        features_clean = features.dropna()
        X = self.scaler.transform(features_clean)
        
        # Get reconstruction errors
        reconstruction_errors = self.autoencoder.reconstruction_error(X)
        
        # Detect anomalies
        anomalies, threshold = self.autoencoder.detect_anomalies(
            X, self.anomaly_threshold
        )
        
        # Create composite risk score
        # Weighted combination of key risk factors
        vol = features_clean["volatility_annual"]
        max_dd = features_clean["max_drawdown"]
        
        # Normalize to 0-1 range
        vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        dd_norm = (max_dd - max_dd.min()) / (max_dd.max() - max_dd.min() + 1e-8)
        err_norm = (reconstruction_errors - reconstruction_errors.min()) / (
            reconstruction_errors.max() - reconstruction_errors.min() + 1e-8
        )
        
        # Composite score (higher = riskier)
        composite_risk = 0.4 * vol_norm + 0.4 * dd_norm + 0.2 * err_norm
        
        # Build results DataFrame
        results = pd.DataFrame({
            "volatility_annual": vol,
            "max_drawdown": max_dd,
            "reconstruction_error": reconstruction_errors,
            "composite_risk_score": composite_risk,
            "is_anomaly": anomalies,
        }, index=features_clean.index)
        
        # Add risk segments
        results["risk_segment"] = segment_by_risk(
            results["composite_risk_score"],
            self.segment_thresholds,
        )
        
        self.results = results
        
        logger.info(f"Risk analysis complete for {len(results)} funds")
        
        return results
    
    def run(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        train_epochs: int = None,
        verbose: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full risk profiling pipeline.
        
        Args:
            returns: Returns DataFrame
            prices: Prices DataFrame
            train_epochs: Autoencoder training epochs
            verbose: Verbosity level
            
        Returns:
            Tuple of (features DataFrame, results DataFrame)
        """
        # Step 1: Extract features
        features = self.extract_features(returns, prices)
        
        # Step 2: Train autoencoder
        self.fit_autoencoder(features, epochs=train_epochs, verbose=verbose)
        
        # Step 3: Compute scores and segment
        results = self.compute_risk_scores(features)
        
        return features, results
    
    def get_segment_funds(
        self,
        segment: str = "Low",
    ) -> pd.Index:
        """
        Get fund codes for a specific risk segment.
        
        Args:
            segment: 'Low', 'Medium', or 'High'
            
        Returns:
            Index of fund codes
        """
        if self.results is None:
            raise ValueError("Run risk profiling first")
        
        return self.results[self.results["risk_segment"] == segment].index
    
    def get_anomalies(self) -> pd.Index:
        """
        Get fund codes flagged as anomalies.
        
        Returns:
            Index of anomalous fund codes
        """
        if self.results is None:
            raise ValueError("Run risk profiling first")
        
        return self.results[self.results["is_anomaly"]].index
    
    def explain_anomalies(
        self,
        features: pd.DataFrame,
        top_n_features: int = 3,
    ) -> pd.DataFrame:
        """
        Explain why anomalies were flagged.
        
        For each anomaly, identifies which features deviate most from
        the mean, providing interpretable explanations.
        
        Args:
            features: Original feature DataFrame
            top_n_features: Number of top deviating features to report
            
        Returns:
            DataFrame with anomaly explanations
        """
        if self.results is None:
            raise ValueError("Run risk profiling first")
        
        anomaly_codes = self.get_anomalies()
        
        if len(anomaly_codes) == 0:
            logger.info("No anomalies to explain")
            return pd.DataFrame()
        
        features_clean = features.dropna()
        
        # Standardize to get z-scores
        X = self.scaler.transform(features_clean)
        X_df = pd.DataFrame(X, index=features_clean.index, columns=features_clean.columns)
        
        explanations = []
        
        for fund_code in anomaly_codes:
            if fund_code not in X_df.index:
                continue
            
            z_scores = X_df.loc[fund_code]
            
            # Get features with highest absolute z-scores
            top_features = z_scores.abs().nlargest(top_n_features)
            
            # Build explanation
            reasons = []
            for feat_name, abs_z in top_features.items():
                actual_z = z_scores[feat_name]
                direction = "high" if actual_z > 0 else "low"
                reasons.append(f"{feat_name} ({direction}, z={actual_z:.2f})")
            
            explanations.append({
                "fund_code": fund_code,
                "reconstruction_error": self.results.loc[fund_code, "reconstruction_error"],
                "risk_score": self.results.loc[fund_code, "composite_risk_score"],
                "top_deviations": "; ".join(reasons),
                "n_extreme_features": (z_scores.abs() > 2).sum(),
            })
        
        explanation_df = pd.DataFrame(explanations).set_index("fund_code")
        
        logger.info(f"Generated explanations for {len(explanation_df)} anomalies")
        
        return explanation_df
    
    def save_results(
        self,
        output_dir: Path = None,
    ) -> None:
        """Save results to disk."""
        if self.results is None:
            raise ValueError("No results to save")
        
        out_dir = output_dir or DATA_PROCESSED
        out_dir.mkdir(parents=True, exist_ok=True)
        
        self.results.to_csv(out_dir / "risk_profiles.csv")
        logger.info(f"Results saved to {out_dir / 'risk_profiles.csv'}")
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.results is None:
            return "No results available. Run risk profiling first."
        
        lines = [
            "=" * 60,
            "FUND RISK PROFILE SUMMARY",
            "=" * 60,
            "",
            f"Total funds analyzed: {len(self.results)}",
            "",
            "Risk Segmentation:",
        ]
        
        for segment in ["Low", "Medium", "High"]:
            count = (self.results["risk_segment"] == segment).sum()
            lines.append(f"  {segment}: {count} funds ({count/len(self.results)*100:.1f}%)")
        
        n_anomalies = self.results["is_anomaly"].sum()
        lines.extend([
            "",
            f"Anomalies detected: {n_anomalies} ({n_anomalies/len(self.results)*100:.1f}%)",
            "",
            "Risk Score Statistics:",
            f"  Mean: {self.results['composite_risk_score'].mean():.3f}",
            f"  Std:  {self.results['composite_risk_score'].std():.3f}",
            f"  Min:  {self.results['composite_risk_score'].min():.3f}",
            f"  Max:  {self.results['composite_risk_score'].max():.3f}",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def extract_risk_profiles(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    verbose: int = 1,
) -> Tuple[RiskProfileExtractor, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run risk profiling.
    
    Args:
        returns: Returns DataFrame
        prices: Prices DataFrame
        verbose: Verbosity
        
    Returns:
        Tuple of (extractor, features, results)
    """
    extractor = RiskProfileExtractor()
    features, results = extractor.run(returns, prices, verbose=verbose)
    
    print(extractor.summary())
    
    return extractor, features, results


if __name__ == "__main__":
    print("Risk Profile Extractor module loaded")
    print("\nUsage:")
    print("  extractor, features, results = extract_risk_profiles(returns, prices)")
