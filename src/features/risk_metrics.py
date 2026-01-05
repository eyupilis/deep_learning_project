"""
TEFAS Fund Analysis - Risk Metrics Module
==========================================
CRISP-DM: Data Preparation / Feature Engineering Phase

This module calculates risk features for each fund:
- Volatility (rolling and overall)
- Maximum Drawdown
- Sharpe Ratio (approximation)
- Return distribution statistics
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rolling_volatility(
    returns: pd.DataFrame,
    windows: List[int] = [21, 63, 126],
) -> Dict[int, pd.DataFrame]:
    """
    Calculate rolling volatility for multiple window sizes.
    
    Args:
        returns: DataFrame with returns (dates x funds)
        windows: List of window sizes in trading days
        
    Returns:
        Dict mapping window size to volatility DataFrame
    """
    result = {}
    
    for window in windows:
        # Annualized volatility = std * sqrt(252)
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        result[window] = vol
        logger.info(f"Calculated {window}-day rolling volatility")
    
    return result


def overall_volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Calculate overall annualized volatility for each fund.
    
    Args:
        returns: DataFrame with returns
        
    Returns:
        Series with volatility per fund
    """
    return returns.std() * np.sqrt(252)


def maximum_drawdown(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate Maximum Drawdown for each fund.
    
    Maximum Drawdown = (Trough - Peak) / Peak
    
    Args:
        prices: DataFrame with prices (dates x funds)
        
    Returns:
        Tuple of (max drawdown Series, drawdown time series DataFrame)
    """
    # Calculate running maximum
    running_max = prices.cummax()
    
    # Calculate drawdown at each point
    drawdown = (prices - running_max) / running_max
    
    # Maximum drawdown is the minimum (most negative) drawdown
    max_dd = drawdown.min()
    
    logger.info(f"Max drawdown range: {max_dd.min():.2%} to {max_dd.max():.2%}")
    
    return max_dd, drawdown


def drawdown_duration(drawdown: pd.DataFrame) -> pd.Series:
    """
    Calculate maximum drawdown duration in days.
    
    Args:
        drawdown: DataFrame with drawdown time series
        
    Returns:
        Series with max drawdown duration per fund
    """
    durations = {}
    
    for fund in drawdown.columns:
        # Find periods where drawdown < 0
        in_drawdown = drawdown[fund] < 0
        
        # Calculate consecutive drawdown periods
        groups = (~in_drawdown).cumsum()
        drawdown_lengths = in_drawdown.groupby(groups).sum()
        
        max_duration = drawdown_lengths.max() if len(drawdown_lengths) > 0 else 0
        durations[fund] = max_duration
    
    return pd.Series(durations)


def sharpe_ratio(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Calculate annualized Sharpe Ratio.
    
    Sharpe = (mean_return - rf) / std_return * sqrt(periods)
    
    Args:
        returns: DataFrame with returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year
        
    Returns:
        Series with Sharpe ratio per fund
    """
    daily_rf = risk_free_rate / periods_per_year
    
    excess_returns = returns - daily_rf
    
    sharpe = (
        excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    )
    
    return sharpe


def sortino_ratio(
    returns: pd.DataFrame,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Calculate Sortino Ratio (penalizes only downside volatility).
    
    Args:
        returns: DataFrame with returns
        target_return: Target/minimum acceptable return
        periods_per_year: Number of periods per year
        
    Returns:
        Series with Sortino ratio per fund
    """
    excess_returns = returns - target_return / periods_per_year
    
    # Downside deviation: std of returns below target
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    # Avoid division by zero
    downside_std = downside_std.replace(0, np.nan)
    
    sortino = excess_returns.mean() * periods_per_year / downside_std
    
    return sortino


def return_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distribution statistics for returns.
    
    Args:
        returns: DataFrame with returns
        
    Returns:
        DataFrame with statistics per fund
    """
    stats = pd.DataFrame({
        "mean_daily": returns.mean(),
        "mean_annual": returns.mean() * 252,
        "std_daily": returns.std(),
        "std_annual": returns.std() * np.sqrt(252),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "min": returns.min(),
        "max": returns.max(),
        "median": returns.median(),
        "positive_days_pct": (returns > 0).sum() / len(returns) * 100,
    })
    
    return stats


def value_at_risk(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.Series:
    """
    Calculate historical Value at Risk.
    
    Args:
        returns: DataFrame with returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        Series with VaR per fund (negative value = potential loss)
    """
    return returns.quantile(1 - confidence_level)


def conditional_var(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.Series:
    """
    Calculate Conditional VaR (Expected Shortfall).
    
    CVaR = Expected loss given that loss exceeds VaR
    
    Args:
        returns: DataFrame with returns
        confidence_level: Confidence level
        
    Returns:
        Series with CVaR per fund
    """
    var = value_at_risk(returns, confidence_level)
    
    cvar = {}
    for fund in returns.columns:
        tail_returns = returns[fund][returns[fund] <= var[fund]]
        cvar[fund] = tail_returns.mean() if len(tail_returns) > 0 else var[fund]
    
    return pd.Series(cvar)


def calculate_all_risk_features(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    volatility_windows: List[int] = [21, 63, 126],
) -> pd.DataFrame:
    """
    Calculate all risk features for each fund.
    
    This is the main function to generate the feature matrix for Module 1.
    
    Args:
        returns: DataFrame with returns
        prices: DataFrame with prices
        volatility_windows: Windows for rolling volatility
        
    Returns:
        DataFrame with funds as rows and features as columns
    """
    features = {}
    n_obs = len(returns)
    
    # Overall volatility (always works)
    features["volatility_annual"] = overall_volatility(returns)
    
    # Adaptive rolling windows based on available data
    # Use windows that are at most 80% of data length
    max_window = max(int(n_obs * 0.8), 3)
    adaptive_windows = [w for w in volatility_windows if w <= max_window]
    
    if not adaptive_windows:
        # Use smaller windows if standard ones are too large
        adaptive_windows = [min(5, n_obs - 1), min(10, n_obs - 1)]
        adaptive_windows = [w for w in adaptive_windows if w > 1]
    
    if adaptive_windows:
        rolling_vols = rolling_volatility(returns, adaptive_windows)
        for window, vol_df in rolling_vols.items():
            # Only add if we have valid values
            if not vol_df.iloc[-1].isna().all():
                features[f"volatility_{window}d_latest"] = vol_df.iloc[-1]
                features[f"volatility_{window}d_mean"] = vol_df.mean()
                features[f"volatility_{window}d_max"] = vol_df.max()
    
    # Maximum Drawdown
    max_dd, drawdown_ts = maximum_drawdown(prices)
    features["max_drawdown"] = max_dd.abs()  # Store as positive for easier interpretation
    
    # Drawdown duration
    features["max_dd_duration"] = drawdown_duration(drawdown_ts)
    
    # Sharpe Ratio
    features["sharpe_ratio"] = sharpe_ratio(returns)
    
    # Sortino Ratio
    features["sortino_ratio"] = sortino_ratio(returns)
    
    # Distribution stats
    return_stats = return_statistics(returns)
    features["return_mean_annual"] = return_stats["mean_annual"]
    features["return_skewness"] = return_stats["skewness"]
    features["return_kurtosis"] = return_stats["kurtosis"]
    features["positive_days_pct"] = return_stats["positive_days_pct"]
    
    # VaR and CVaR
    features["var_95"] = value_at_risk(returns, 0.95).abs()
    features["cvar_95"] = conditional_var(returns, 0.95).abs()
    
    # Combine into DataFrame
    feature_df = pd.DataFrame(features)
    
    # Drop columns with all NaN
    feature_df = feature_df.dropna(axis=1, how='all')
    
    logger.info(f"Calculated {len(feature_df.columns)} features for {len(feature_df)} funds")
    
    return feature_df


def segment_by_risk(
    risk_scores: pd.Series,
    thresholds: Dict[str, float] = None,
) -> pd.Series:
    """
    Segment funds into Low/Medium/High risk categories.
    
    Args:
        risk_scores: Series with composite risk scores
        thresholds: Dict with percentile thresholds
        
    Returns:
        Series with risk segment labels
    """
    if thresholds is None:
        thresholds = {"low": 33, "medium": 66, "high": 100}
    
    low_threshold = np.percentile(risk_scores, thresholds["low"])
    medium_threshold = np.percentile(risk_scores, thresholds["medium"])
    
    segments = pd.cut(
        risk_scores,
        bins=[-np.inf, low_threshold, medium_threshold, np.inf],
        labels=["Low", "Medium", "High"],
    )
    
    logger.info(f"Risk segmentation: {segments.value_counts().to_dict()}")
    
    return segments


if __name__ == "__main__":
    print("Risk Metrics module loaded successfully")
    print("Main function: calculate_all_risk_features(returns, prices)")
