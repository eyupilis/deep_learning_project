"""
TEFAS Fund Analysis - Visualization Module
============================================
CRISP-DM: Evaluation Phase

Plotting utilities for:
- Risk profile distributions
- 2D latent space visualization
- Backtest performance charts
- Correlation heatmaps
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import FIGURE_DPI, FIGURE_FORMAT, OUTPUT_FIGURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def save_figure(fig: plt.Figure, name: str, output_dir: Path = None) -> Path:
    """Save figure to disk."""
    out_dir = output_dir or OUTPUT_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = out_dir / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved figure: {filepath}")
    return filepath


def plot_risk_distribution(
    risk_profiles: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Plot distribution of risk scores by segment.
    
    Args:
        risk_profiles: Results from RiskProfileExtractor
        save: Whether to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Risk score histogram by segment
    ax = axes[0, 0]
    for segment in ["Low", "Medium", "High"]:
        mask = risk_profiles["risk_segment"] == segment
        ax.hist(
            risk_profiles.loc[mask, "composite_risk_score"],
            bins=15,
            alpha=0.6,
            label=segment,
        )
    ax.set_xlabel("Composite Risk Score")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution by Segment")
    ax.legend()
    
    # 2. Volatility vs Drawdown scatter
    ax = axes[0, 1]
    scatter = ax.scatter(
        risk_profiles["volatility_annual"] * 100,
        risk_profiles["max_drawdown"] * 100,
        c=risk_profiles["composite_risk_score"],
        cmap="RdYlGn_r",
        alpha=0.7,
        s=50,
    )
    ax.set_xlabel("Annual Volatility (%)")
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Volatility vs Drawdown")
    plt.colorbar(scatter, ax=ax, label="Risk Score")
    
    # 3. Segment counts
    ax = axes[1, 0]
    segment_counts = risk_profiles["risk_segment"].value_counts()
    colors = ["green", "orange", "red"]
    segment_counts = segment_counts.reindex(["Low", "Medium", "High"])
    bars = ax.bar(segment_counts.index, segment_counts.values, color=colors)
    ax.set_xlabel("Risk Segment")
    ax.set_ylabel("Number of Funds")
    ax.set_title("Funds by Risk Segment")
    
    # Add count labels
    for bar, count in zip(bars, segment_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
        )
    
    # 4. Anomaly detection
    ax = axes[1, 1]
    anomaly_counts = risk_profiles["is_anomaly"].value_counts()
    labels = ["Normal", "Anomaly"]
    sizes = [anomaly_counts.get(False, 0), anomaly_counts.get(True, 0)]
    colors = ["lightblue", "salmon"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Anomaly Detection Results")
    
    fig.suptitle("Fund Risk Profile Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save:
        save_figure(fig, "risk_distribution")
    
    return fig


def plot_2d_embedding(
    results: pd.DataFrame,
    title: str = "Fund Embedding Space",
    save: bool = True,
) -> plt.Figure:
    """
    Plot 2D latent space from embedding autoencoder.
    
    Args:
        results: Results from CorrelationMapper
        title: Plot title
        save: Whether to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique clusters and assign colors
    clusters = results["cluster"].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))
    
    for cluster, color in zip(clusters, colors):
        mask = results["cluster"] == cluster
        cluster_data = results[mask]
        
        ax.scatter(
            cluster_data["embedding_x"],
            cluster_data["embedding_y"],
            c=[color],
            label=f"Cluster {cluster}",
            s=100,
            alpha=0.7,
        )
        
        # Add fund labels
        for fund_code in cluster_data.index:
            ax.annotate(
                fund_code,
                (cluster_data.loc[fund_code, "embedding_x"],
                 cluster_data.loc[fund_code, "embedding_y"]),
                fontsize=7,
                alpha=0.8,
            )
    
    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best")
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, "2d_embedding")
    
    return fig


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Plot correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix from CorrelationMapper
        save: Whether to save
        
    Returns:
        Matplotlib figure
    """
    # Limit size if too many funds
    n_funds = len(correlation_matrix)
    figsize = (min(20, n_funds * 0.5), min(16, n_funds * 0.4))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
        annot=n_funds <= 15,  # Only annotate if small
        fmt=".2f",
    )
    
    ax.set_title("Fund Return Correlation Matrix")
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, "correlation_heatmap")
    
    return fig


def plot_backtest_performance(
    backtest_results: pd.DataFrame,
    portfolio_name: str = "Portfolio",
    benchmark_returns: pd.Series = None,
    save: bool = True,
) -> plt.Figure:
    """
    Plot backtest performance charts.
    
    Args:
        backtest_results: Results from PortfolioSimulator
        portfolio_name: Name for legend
        benchmark_returns: Optional benchmark for comparison
        save: Whether to save
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # 1. Portfolio Value
    ax = axes[0]
    ax.plot(
        backtest_results.index,
        backtest_results["portfolio_value"],
        linewidth=2,
        label=portfolio_name,
    )
    
    if benchmark_returns is not None:
        benchmark_value = 100000 * (1 + benchmark_returns).cumprod()
        ax.plot(benchmark_value.index, benchmark_value.values, 
                linewidth=2, linestyle="--", label="Benchmark")
    
    ax.set_ylabel("Portfolio Value (TRY)")
    ax.set_title("Portfolio Value Over Time")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    
    # 2. Daily Returns
    ax = axes[1]
    returns = backtest_results["daily_return"].dropna()
    ax.bar(
        returns.index,
        returns.values * 100,
        color=np.where(returns >= 0, "green", "red"),
        alpha=0.6,
        width=1,
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Daily Return (%)")
    ax.set_title("Daily Returns")
    
    # 3. Drawdown
    ax = axes[2]
    ax.fill_between(
        backtest_results.index,
        backtest_results["drawdown"] * 100,
        0,
        color="red",
        alpha=0.5,
    )
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown")
    
    fig.suptitle(
        f"Backtest Results: {portfolio_name}\n(NOT INVESTMENT ADVICE)",
        fontsize=14,
        fontweight="bold",
    )
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, "backtest_performance")
    
    return fig


def plot_cluster_analysis(
    cluster_analysis: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Plot cluster analysis results.
    
    Args:
        cluster_analysis: Results from CorrelationMapper
        save: Whether to save
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Cluster sizes
    ax = axes[0]
    ax.bar(
        cluster_analysis["cluster_id"],
        cluster_analysis["n_funds"],
        color="steelblue",
    )
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Funds")
    ax.set_title("Cluster Sizes")
    
    # 2. Intra-cluster correlations
    ax = axes[1]
    if "avg_intra_correlation" in cluster_analysis.columns:
        corrs = cluster_analysis["avg_intra_correlation"].dropna()
        colors = ["red" if c > corrs.median() else "green" for c in corrs]
        ax.bar(
            cluster_analysis["cluster_id"],
            cluster_analysis["avg_intra_correlation"],
            color=colors,
        )
        ax.axhline(
            y=corrs.median(),
            color="black",
            linestyle="--",
            label=f"Median ({corrs.median():.2f})",
        )
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Avg Intra-Cluster Correlation")
        ax.set_title("Correlation Within Clusters\n(High = Potential Diversification Illusion)")
        ax.legend()
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, "cluster_analysis")
    
    return fig


def create_all_visualizations(
    risk_profiles: pd.DataFrame,
    correlation_results: pd.DataFrame,
    cluster_analysis: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    backtest_results: pd.DataFrame,
    output_dir: Path = None,
) -> List[Path]:
    """
    Create and save all visualizations.
    
    Args:
        risk_profiles: From Module 1
        correlation_results: From Module 2
        cluster_analysis: From Module 2
        correlation_matrix: From Module 2
        backtest_results: From Module 3
        output_dir: Output directory
        
    Returns:
        List of saved file paths
    """
    out_dir = output_dir or OUTPUT_FIGURES
    saved = []
    
    if risk_profiles is not None:
        plot_risk_distribution(risk_profiles)
        saved.append(out_dir / "risk_distribution.png")
    
    if correlation_results is not None:
        plot_2d_embedding(correlation_results)
        saved.append(out_dir / "2d_embedding.png")
    
    if correlation_matrix is not None:
        plot_correlation_heatmap(correlation_matrix)
        saved.append(out_dir / "correlation_heatmap.png")
    
    if cluster_analysis is not None:
        plot_cluster_analysis(cluster_analysis)
        saved.append(out_dir / "cluster_analysis.png")
    
    if backtest_results is not None:
        plot_backtest_performance(backtest_results)
        saved.append(out_dir / "backtest_performance.png")
    
    logger.info(f"Created {len(saved)} visualizations")
    
    return saved


if __name__ == "__main__":
    print("Visualization module loaded")
    print("\nAvailable functions:")
    print("- plot_risk_distribution()")
    print("- plot_2d_embedding()")
    print("- plot_correlation_heatmap()")
    print("- plot_backtest_performance()")
    print("- create_all_visualizations()")
