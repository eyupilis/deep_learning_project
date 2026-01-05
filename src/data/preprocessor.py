"""
TEFAS Fund Analysis - Data Preprocessing Module
=================================================
CRISP-DM: Data Preparation Phase

This module handles:
- Date alignment across funds
- Missing value handling
- Return calculations (simple, log)
- Inflation adjustment
- Data matrix preparation for modeling
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def align_fund_data(
    fund_data: Dict[str, pd.DataFrame],
    price_column: str = "close",
    method: str = "inner",
) -> pd.DataFrame:
    """
    Align multiple fund price series to common dates.
    
    Args:
        fund_data: Dict mapping fund codes to DataFrames
        price_column: Column name containing prices (close, adj_close, etc.)
        method: Join method - 'inner' (only common dates) or 'outer'
        
    Returns:
        DataFrame with dates as index, fund codes as columns
    """
    if not fund_data:
        raise ValueError("No fund data provided")
    
    price_series = {}
    
    for code, df in fund_data.items():
        # Handle different possible column names
        col = None
        for candidate in [price_column, "Close", "close", "Adj Close", "price"]:
            if candidate in df.columns:
                col = candidate
                break
        
        if col is None:
            # If no standard column, try first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                logger.warning(f"Using column '{col}' for fund {code}")
            else:
                logger.warning(f"No numeric column found for fund {code}, skipping")
                continue
        
        series = df[col].copy()
        series.name = code
        price_series[code] = series
    
    if not price_series:
        raise ValueError("No valid price data found")
    
    # Combine all series
    aligned = pd.concat(price_series.values(), axis=1, join=method)
    aligned.columns = list(price_series.keys())
    
    # Sort by date
    aligned = aligned.sort_index()
    
    # Report alignment stats
    logger.info(
        f"Aligned {len(price_series)} funds: "
        f"{len(aligned)} common dates from {aligned.index.min()} to {aligned.index.max()}"
    )
    
    return aligned


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "simple",
    periods: int = 1,
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with prices (dates x funds)
        method: 'simple' for arithmetic returns, 'log' for log returns
        periods: Number of periods for return calculation
        
    Returns:
        DataFrame with returns
    """
    if method == "simple":
        returns = prices.pct_change(periods=periods)
    elif method == "log":
        returns = np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'")
    
    # Drop NaN rows from the start
    returns = returns.dropna()
    
    return returns


def resample_to_monthly(data: pd.DataFrame, agg_method: str = "last") -> pd.DataFrame:
    """
    Resample daily/weekly data to monthly frequency.
    
    Args:
        data: DataFrame with DatetimeIndex
        agg_method: 'last' for prices, 'sum' for returns
        
    Returns:
        Monthly resampled DataFrame
    """
    if agg_method == "last":
        return data.resample("M").last()
    elif agg_method == "sum":
        return data.resample("M").sum()
    elif agg_method == "mean":
        return data.resample("M").mean()
    else:
        raise ValueError(f"Unknown agg_method: {agg_method}")


def adjust_for_inflation(
    returns: pd.DataFrame,
    tufe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adjust nominal returns for inflation using TÜFE data.
    
    Formula: real_return = (1 + nominal_return) / (1 + inflation_rate) - 1
    
    For daily returns, we need to spread monthly inflation across days.
    
    Args:
        returns: DataFrame with nominal returns (daily)
        tufe: DataFrame with TÜFE data (monthly)
        
    Returns:
        DataFrame with inflation-adjusted (real) returns
    """
    if tufe is None or len(tufe) == 0:
        logger.warning("No TÜFE data provided, returning nominal returns")
        return returns
    
    # Ensure TÜFE has proper structure
    if isinstance(tufe, pd.DataFrame):
        # Try to find the TÜFE column
        tufe_col = None
        for col in tufe.columns:
            if "tufe" in col.lower() or "tüfe" in col.lower() or "cpi" in col.lower():
                tufe_col = col
                break
        if tufe_col is None and len(tufe.columns) > 0:
            tufe_col = tufe.columns[0]
        
        if tufe_col:
            tufe_series = tufe[tufe_col]
        else:
            logger.warning("Could not identify TÜFE column")
            return returns
    else:
        tufe_series = tufe
    
    # Ensure numeric type - TUFE might be returned as string from API
    tufe_series = pd.to_numeric(tufe_series, errors='coerce')
    
    # Calculate monthly inflation rate from TÜFE index
    # TÜFE is typically reported as an index, so inflation = pct_change
    monthly_inflation = tufe_series.pct_change()
    
    # Convert to daily by spreading evenly across month
    # (approximation - assumes uniform inflation within month)
    daily_inflation = monthly_inflation.resample("D").ffill() / 21  # ~21 trading days
    
    # Align dates
    common_dates = returns.index.intersection(daily_inflation.index)
    
    if len(common_dates) == 0:
        logger.warning("No overlapping dates between returns and TÜFE")
        return returns
    
    aligned_returns = returns.loc[common_dates]
    aligned_inflation = daily_inflation.loc[common_dates]
    
    # Calculate real returns
    real_returns = (1 + aligned_returns).div(1 + aligned_inflation, axis=0) - 1
    
    logger.info(f"Adjusted {len(real_returns)} days of returns for inflation")
    
    return real_returns


def handle_missing_values(
    data: pd.DataFrame,
    method: str = "ffill",
    max_consecutive: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle missing values in fund data.
    
    Args:
        data: DataFrame with potential missing values
        method: 'ffill' (forward fill), 'interpolate', or 'drop'
        max_consecutive: Max consecutive NaNs to fill
        
    Returns:
        Tuple of (filled DataFrame, DataFrame with missing value stats)
    """
    # Calculate missing stats before filling
    missing_stats = pd.DataFrame({
        "total_missing": data.isna().sum(),
        "pct_missing": data.isna().sum() / len(data) * 100,
        "max_consecutive": data.isna().astype(int).groupby(
            (~data.isna()).cumsum()
        ).sum().max() if data.isna().any().any() else 0,
    })
    
    if method == "ffill":
        filled = data.ffill(limit=max_consecutive)
    elif method == "interpolate":
        filled = data.interpolate(method="linear", limit=max_consecutive)
    elif method == "drop":
        filled = data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Report remaining missing after fill
    remaining = filled.isna().sum().sum()
    if remaining > 0:
        logger.warning(f"{remaining} missing values remain after {method}")
    
    return filled, missing_stats


def prepare_return_matrix(
    fund_data: Dict[str, pd.DataFrame],
    tufe: Optional[pd.DataFrame] = None,
    adjust_inflation: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: align, calculate returns, adjust inflation.
    
    Args:
        fund_data: Dict of fund DataFrames
        tufe: TÜFE DataFrame
        adjust_inflation: Whether to adjust for inflation
        
    Returns:
        Tuple of (returns matrix, aligned prices)
    """
    # Step 1: Align prices
    prices = align_fund_data(fund_data)
    
    # Step 2: Handle missing values
    prices, missing_stats = handle_missing_values(prices)
    
    if missing_stats["pct_missing"].max() > 10:
        logger.warning("Some funds have >10% missing data")
    
    # Step 3: Calculate returns
    returns = calculate_returns(prices, method="simple")
    
    # Step 4: Adjust for inflation if requested
    if adjust_inflation and tufe is not None:
        returns = adjust_for_inflation(returns, tufe)
    
    logger.info(
        f"Prepared return matrix: {returns.shape[0]} days x {returns.shape[1]} funds"
    )
    
    return returns, prices


def calculate_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative returns from period returns.
    
    Args:
        returns: DataFrame with period returns
        
    Returns:
        DataFrame with cumulative returns (growth of $1)
    """
    return (1 + returns).cumprod()


def winsorize_returns(
    returns: pd.DataFrame,
    lower_percentile: float = 1,
    upper_percentile: float = 99,
) -> pd.DataFrame:
    """
    Winsorize returns to handle extreme outliers.
    
    Args:
        returns: DataFrame with returns
        lower_percentile: Lower bound percentile
        upper_percentile: Upper bound percentile
        
    Returns:
        Winsorized returns
    """
    lower = returns.quantile(lower_percentile / 100)
    upper = returns.quantile(upper_percentile / 100)
    
    return returns.clip(lower=lower, upper=upper, axis=1)


if __name__ == "__main__":
    # Example usage - requires actual data
    print("Preprocessor module loaded successfully")
    print("Available functions:")
    print("- align_fund_data()")
    print("- calculate_returns()")
    print("- adjust_for_inflation()")
    print("- prepare_return_matrix()")
