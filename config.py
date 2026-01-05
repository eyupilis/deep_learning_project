"""
Global Configuration for TEFAS Fund Analysis
=============================================
CRISP-DM: Business Understanding Phase

This configuration defines:
- Fund universe and selection criteria
- Time range for analysis
- Model hyperparameters
- Risk thresholds and constraints
"""

from datetime import date
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

# =============================================================================
# DATA COLLECTION SETTINGS
# =============================================================================
# Time range for historical data
START_DATE = date(2020, 1, 1)
END_DATE = date(2026, 12, 31)  # Extended to include current data

# Fund universe - will be populated programmatically
# Focus: KuveytTürk Portföy and participation-compatible funds
FUND_SEARCH_KEYWORDS = [
    "kuveyt",
    "katılım",
]

# Participation finance filter patterns
PARTICIPATION_PATTERNS = [
    "katılım",
    "sukuk",
    "faizsiz",
    "islami",
]

# Minimum data points required for analysis (trading days)
MIN_DATA_POINTS = 252  # Approximately 1 year

# =============================================================================
# FEATURE ENGINEERING SETTINGS
# =============================================================================
# Rolling window sizes (trading days)
VOLATILITY_WINDOWS = [21, 63, 126]  # 1 month, 3 months, 6 months

# Return calculation
RETURN_FREQUENCY = "D"  # Daily

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Module 1: Risk Profile Autoencoder
# Smaller network to avoid overfitting on limited fund universe (~50 funds)
RISK_AE_CONFIG = {
    "input_dim": None,  # Set dynamically based on features
    "encoding_dims": [16, 8],  # Reduced from [32, 16, 8] - simpler architecture
    "latent_dim": 4,
    "activation": "relu",
    "epochs": 50,  # Reduced from 100 - early stopping handles convergence
    "batch_size": 16,  # Smaller batch for small datasets
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "learning_rate": 0.001,
    "dropout_rate": 0.1,
}

# Module 2: Embedding Autoencoder
# 2D latent for direct visualization without UMAP/t-SNE
EMBEDDING_AE_CONFIG = {
    "input_dim": None,  # Set dynamically
    "encoding_dims": [32, 16],  # Reduced from [64, 32]
    "latent_dim": 2,  # For direct 2D visualization
    "activation": "relu",
    "epochs": 100,  # Reduced from 150
    "batch_size": 8,  # Very small batches for few funds
    "validation_split": 0.2,
    "early_stopping_patience": 15,
    "dropout_rate": 0.2,
}

# Module 3: Risk-Return Scorer ANN
# Auxiliary role only - kept minimal to avoid overfitting
SCORER_ANN_CONFIG = {
    "hidden_layers": [16, 8],  # Reduced from [32, 16]
    "activation": "relu",
    "output_activation": "linear",
    "epochs": 50,  # Reduced from 100
    "batch_size": 8,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "dropout_rate": 0.3,  # Higher dropout for strong regularization
}

# =============================================================================
# RISK SEGMENTATION THRESHOLDS
# =============================================================================
# Percentile-based segmentation
RISK_SEGMENT_THRESHOLDS = {
    "low": 33,      # Bottom 33% by risk score
    "medium": 66,   # 33-66%
    "high": 100,    # Top 34%
}

# Anomaly detection threshold (reconstruction error percentile)
ANOMALY_PERCENTILE_THRESHOLD = 95

# =============================================================================
# PORTFOLIO SIMULATION CONSTRAINTS
# =============================================================================
PORTFOLIO_CONSTRAINTS = {
    "min_funds": 3,           # Minimum funds in basket
    "max_funds": 10,          # Maximum funds in basket
    "max_weight_per_fund": 0.30,  # Max 30% in single fund
    "max_drawdown_threshold": 0.15,  # Max 15% drawdown allowed
}

# Backtesting settings
BACKTEST_CONFIG = {
    "initial_capital": 100000,  # TRY
    "rebalance_frequency": "M",  # Monthly
    "transaction_cost": 0.001,   # 0.1%
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
RANDOM_SEED = 42
FIGURE_DPI = 150
FIGURE_FORMAT = "png"

# =============================================================================
# DISCLAIMER
# =============================================================================
DISCLAIMER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                              IMPORTANT DISCLAIMER                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This analysis is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.                ║
║  It is NOT investment advice.                                                 ║
║                                                                               ║
║  • All results are historical simulations                                     ║
║  • Past performance does not guarantee future results                         ║
║  • No buy/sell recommendations are made                                       ║
║  • Consult a licensed financial advisor for investment decisions              ║
║                                                                               ║
║  KuveytTürk Portföy is the conceptual stakeholder for academic demonstration ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
