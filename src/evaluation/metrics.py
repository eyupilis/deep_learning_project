"""
TEFAS Fund Analysis - Evaluation Metrics Module
=================================================
CRISP-DM: Evaluation Phase

This module provides evaluation metrics for:
- Autoencoder performance (reconstruction quality)
- Risk segmentation quality
- Portfolio backtest performance
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_autoencoder(
    X: np.ndarray,
    reconstructed: np.ndarray,
) -> Dict:
    """
    Evaluate autoencoder reconstruction quality.
    
    Args:
        X: Original input
        reconstructed: Reconstructed output
        
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((X - reconstructed) ** 2)
    mae = np.mean(np.abs(X - reconstructed))
    
    # Explained variance ratio (like R²)
    ss_res = np.sum((X - reconstructed) ** 2)
    ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mae,
        "explained_variance_ratio": r2,
    }
    
    logger.info(f"Autoencoder MSE: {mse:.6f}, R²: {r2:.4f}")
    
    return metrics


def evaluate_clustering(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict:
    """
    Evaluate clustering quality.
    
    Args:
        embeddings: Embedding vectors
        cluster_labels: Cluster assignments
        
    Returns:
        Dictionary of metrics
    """
    # Silhouette score: [-1, 1], higher is better
    n_clusters = len(np.unique(cluster_labels))
    
    if n_clusters < 2:
        logger.warning("Need at least 2 clusters for silhouette score")
        return {"silhouette_score": np.nan, "n_clusters": n_clusters}
    
    silhouette = silhouette_score(embeddings, cluster_labels)
    
    # Cluster balance (entropy-based)
    counts = np.bincount(cluster_labels)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(n_clusters)
    balance = entropy / max_entropy if max_entropy > 0 else 0
    
    metrics = {
        "silhouette_score": silhouette,
        "n_clusters": n_clusters,
        "cluster_balance": balance,  # 1 = perfectly balanced
    }
    
    logger.info(f"Clustering Silhouette: {silhouette:.4f}")
    
    return metrics


def evaluate_risk_segmentation(
    risk_profiles: pd.DataFrame,
) -> Dict:
    """
    Evaluate risk segmentation quality.
    
    Checks if segments have distinct risk characteristics.
    
    Args:
        risk_profiles: Results from RiskProfileExtractor
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    segments = ["Low", "Medium", "High"]
    
    # Check mean risk scores per segment
    segment_means = {}
    for segment in segments:
        mask = risk_profiles["risk_segment"] == segment
        if mask.sum() > 0:
            segment_means[segment] = risk_profiles.loc[mask, "composite_risk_score"].mean()
    
    metrics["segment_mean_scores"] = segment_means
    
    # Check monotonicity (Low < Medium < High)
    if len(segment_means) == 3:
        is_monotonic = (
            segment_means.get("Low", 0) < 
            segment_means.get("Medium", 0) < 
            segment_means.get("High", 0)
        )
        metrics["is_monotonic"] = is_monotonic
    
    # Segment separation (ratio of between-segment variance to within-segment variance)
    overall_mean = risk_profiles["composite_risk_score"].mean()
    between_var = sum(
        (risk_profiles["risk_segment"] == seg).sum() * 
        (segment_means.get(seg, overall_mean) - overall_mean) ** 2
        for seg in segments
    )
    
    within_var = sum(
        risk_profiles.loc[risk_profiles["risk_segment"] == seg, "composite_risk_score"].var() *
        (risk_profiles["risk_segment"] == seg).sum()
        for seg in segments
    )
    
    if within_var > 0:
        metrics["separation_ratio"] = between_var / within_var
    else:
        metrics["separation_ratio"] = np.inf
    
    logger.info(f"Segmentation monotonic: {metrics.get('is_monotonic', 'N/A')}")
    
    return metrics


def evaluate_backtest(
    backtest_results: pd.DataFrame,
    initial_capital: float = 100000,
) -> Dict:
    """
    Evaluate backtest performance with standard metrics.
    
    Args:
        backtest_results: From PortfolioSimulator
        initial_capital: Starting capital
        
    Returns:
        Dictionary of metrics
    """
    daily_returns = backtest_results["daily_return"].dropna()
    
    n_days = len(daily_returns)
    n_years = n_days / 252
    
    final_value = backtest_results["portfolio_value"].iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    metrics = {
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0,
        "annualized_volatility": daily_returns.std() * np.sqrt(252),
        "sharpe_ratio": (
            daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            if daily_returns.std() > 0 else 0
        ),
        "max_drawdown": backtest_results["drawdown"].min(),
        "calmar_ratio": None,  # Will calculate below
        "win_rate": (daily_returns > 0).mean(),
        "avg_daily_return": daily_returns.mean(),
        "n_trading_days": n_days,
    }
    
    # Calmar ratio: annualized return / max drawdown
    if metrics["max_drawdown"] < 0:
        metrics["calmar_ratio"] = metrics["annualized_return"] / abs(metrics["max_drawdown"])
    
    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(252)
        metrics["sortino_ratio"] = (
            metrics["annualized_return"] / downside_std if downside_std > 0 else 0
        )
    else:
        metrics["sortino_ratio"] = np.inf
    
    return metrics


def generate_evaluation_report(
    ae_metrics: Dict = None,
    clustering_metrics: Dict = None,
    segmentation_metrics: Dict = None,
    backtest_metrics: Dict = None,
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        ae_metrics: Autoencoder metrics
        clustering_metrics: Clustering metrics
        segmentation_metrics: Segmentation metrics
        backtest_metrics: Backtest metrics
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "TEFAS FUND ANALYSIS - EVALUATION REPORT",
        "=" * 70,
        "",
    ]
    
    if ae_metrics:
        lines.extend([
            "AUTOENCODER PERFORMANCE",
            "-" * 40,
            f"  MSE:                      {ae_metrics.get('mse', 'N/A'):.6f}",
            f"  RMSE:                     {ae_metrics.get('rmse', 'N/A'):.6f}",
            f"  MAE:                      {ae_metrics.get('mae', 'N/A'):.6f}",
            f"  Explained Variance:       {ae_metrics.get('explained_variance_ratio', 'N/A'):.4f}",
            "",
        ])
    
    if clustering_metrics:
        lines.extend([
            "CLUSTERING QUALITY",
            "-" * 40,
            f"  Silhouette Score:         {clustering_metrics.get('silhouette_score', 'N/A'):.4f}",
            f"  Number of Clusters:       {clustering_metrics.get('n_clusters', 'N/A')}",
            f"  Cluster Balance:          {clustering_metrics.get('cluster_balance', 'N/A'):.4f}",
            "",
        ])
    
    if segmentation_metrics:
        lines.extend([
            "RISK SEGMENTATION QUALITY",
            "-" * 40,
            f"  Is Monotonic (L<M<H):     {segmentation_metrics.get('is_monotonic', 'N/A')}",
            f"  Separation Ratio:         {segmentation_metrics.get('separation_ratio', 'N/A'):.4f}",
        ])
        if "segment_mean_scores" in segmentation_metrics:
            for seg, score in segmentation_metrics["segment_mean_scores"].items():
                lines.append(f"    {seg} Mean Score:      {score:.4f}")
        lines.append("")
    
    if backtest_metrics:
        lines.extend([
            "BACKTEST PERFORMANCE (NOT INVESTMENT ADVICE)",
            "-" * 40,
            f"  Total Return:             {backtest_metrics.get('total_return', 0):.2%}",
            f"  Annualized Return:        {backtest_metrics.get('annualized_return', 0):.2%}",
            f"  Annual Volatility:        {backtest_metrics.get('annualized_volatility', 0):.2%}",
            f"  Sharpe Ratio:             {backtest_metrics.get('sharpe_ratio', 0):.2f}",
            f"  Sortino Ratio:            {backtest_metrics.get('sortino_ratio', 0):.2f}",
            f"  Max Drawdown:             {backtest_metrics.get('max_drawdown', 0):.2%}",
            f"  Calmar Ratio:             {backtest_metrics.get('calmar_ratio', 0):.2f}",
            f"  Win Rate:                 {backtest_metrics.get('win_rate', 0):.2%}",
            "",
        ])
    
    lines.extend([
        "=" * 70,
        "⚠️  This analysis is for EDUCATIONAL PURPOSES ONLY.",
        "⚠️  NOT investment advice. Past performance ≠ future results.",
        "=" * 70,
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Evaluation Metrics module loaded")
    print("\nAvailable functions:")
    print("- evaluate_autoencoder()")
    print("- evaluate_clustering()")
    print("- evaluate_risk_segmentation()")
    print("- evaluate_backtest()")
    print("- generate_evaluation_report()")
