"""TEFAS Fund Analysis - Evaluation Package"""

from .metrics import (
    evaluate_autoencoder,
    evaluate_backtest,
    evaluate_clustering,
    evaluate_risk_segmentation,
    generate_evaluation_report,
)
from .visualizations import (
    create_all_visualizations,
    plot_2d_embedding,
    plot_backtest_performance,
    plot_correlation_heatmap,
    plot_risk_distribution,
)

__all__ = [
    "evaluate_autoencoder",
    "evaluate_backtest",
    "evaluate_clustering",
    "evaluate_risk_segmentation",
    "generate_evaluation_report",
    "create_all_visualizations",
    "plot_2d_embedding",
    "plot_backtest_performance",
    "plot_correlation_heatmap",
    "plot_risk_distribution",
]
