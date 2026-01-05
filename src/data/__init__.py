"""TEFAS Fund Analysis - Data Collection Package"""

from .collector import (
    collect_fund_data,
    collect_inflation_data,
    collect_multiple_funds,
    is_participation_fund,
    load_raw_data,
    save_raw_data,
    search_funds,
)
from .preprocessor import (
    align_fund_data,
    calculate_returns,
    adjust_for_inflation,
    prepare_return_matrix,
)
from .exploratory import (
    ExploratoryAnalysis,
    run_eda,
)

__all__ = [
    "collect_fund_data",
    "collect_inflation_data",
    "collect_multiple_funds",
    "is_participation_fund",
    "load_raw_data",
    "save_raw_data",
    "search_funds",
    "align_fund_data",
    "calculate_returns",
    "adjust_for_inflation",
    "prepare_return_matrix",
    "ExploratoryAnalysis",
    "run_eda",
]
