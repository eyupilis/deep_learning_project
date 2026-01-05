"""
TEFAS Fund Analysis - Data Collection Module
=============================================
CRISP-DM: Data Understanding Phase

This module handles:
- TEFAS fund data collection via borsapy
- TÜFE (CPI) inflation data collection
- Fund universe discovery and filtering
"""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import borsapy as bp
except ImportError:
    raise ImportError("borsapy is required. Install via: pip install borsapy")

from config import (
    DATA_RAW,
    END_DATE,
    FUND_SEARCH_KEYWORDS,
    MIN_DATA_POINTS,
    PARTICIPATION_PATTERNS,
    START_DATE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_funds(keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Search for funds matching specified keywords.
    
    Args:
        keywords: List of search terms. If None, uses config defaults.
        
    Returns:
        DataFrame with fund codes and names.
    """
    if keywords is None:
        keywords = FUND_SEARCH_KEYWORDS
    
    all_funds = []
    for keyword in keywords:
        try:
            results = bp.search_funds(keyword)
            if results is not None and len(results) > 0:
                # Handle different return types from borsapy
                if isinstance(results, pd.DataFrame):
                    all_funds.append(results)
                elif isinstance(results, list):
                    # Convert list to DataFrame
                    if isinstance(results[0], dict):
                        df = pd.DataFrame(results)
                    else:
                        # Assume list of fund codes/names
                        df = pd.DataFrame({"code": results})
                    all_funds.append(df)
                logger.info(f"Found {len(results)} funds for keyword '{keyword}'")
        except Exception as e:
            logger.warning(f"Error searching for '{keyword}': {e}")
    
    if not all_funds:
        logger.warning("No funds found for any keyword")
        return pd.DataFrame(columns=["code", "name"])
    
    combined = pd.concat(all_funds, ignore_index=True)
    
    # Handle duplicate removal with flexible column names
    if "code" in combined.columns:
        combined = combined.drop_duplicates(subset=["code"])
    elif "FonKodu" in combined.columns:
        combined = combined.drop_duplicates(subset=["FonKodu"])
        combined = combined.rename(columns={"FonKodu": "code", "FonAdi": "name"})
    else:
        combined = combined.drop_duplicates()
    
    return combined


def is_participation_fund(fund_name: str) -> bool:
    """
    Check if a fund is participation (Islamic) finance compatible.
    
    Args:
        fund_name: Fund name string
        
    Returns:
        True if fund matches participation patterns
    """
    name_lower = fund_name.lower()
    return any(pattern in name_lower for pattern in PARTICIPATION_PATTERNS)


def collect_fund_data(
    fund_code: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """
    Collect historical price data for a single fund.
    
    Args:
        fund_code: TEFAS fund code
        start_date: Start date for data collection
        end_date: End date for data collection
        
    Returns:
        DataFrame with OHLCV data, or None if collection fails
    """
    start = start_date or START_DATE
    end = end_date or END_DATE
    
    try:
        fund = bp.Fund(fund_code)
        
        # borsapy uses Turkish period format
        # Try different periods to get maximum data
        history = None
        for period in ["5y", "3y", "1y", "6ay", "3ay", "1ay"]:
            try:
                history = fund.history(period=period)
                if history is not None and len(history) > 0:
                    break
            except:
                continue
        
        if history is None or len(history) == 0:
            logger.warning(f"No data returned for fund {fund_code}")
            return None
        
        # Ensure datetime index
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.to_datetime(history.index)
        
        # Filter to date range
        history = history.loc[
            (history.index >= pd.Timestamp(start)) &
            (history.index <= pd.Timestamp(end))
        ]
        
        # Reduced minimum requirement for demo with limited historical data
        min_required = min(MIN_DATA_POINTS, 10)  # At least 10 days
        if len(history) < min_required:
            logger.warning(
                f"Fund {fund_code} has only {len(history)} data points "
                f"(minimum: {min_required})"
            )
            return None
        
        history["fund_code"] = fund_code
        
        logger.info(f"Collected {len(history)} data points for {fund_code}")
        return history
        
    except Exception as e:
        logger.error(f"Error collecting data for {fund_code}: {e}")
        return None


def collect_fund_info(fund_code: str) -> Optional[Dict]:
    """
    Collect fund metadata/info.
    
    Args:
        fund_code: TEFAS fund code
        
    Returns:
        Dictionary with fund info, or None if collection fails
    """
    try:
        fund = bp.Fund(fund_code)
        info = fund.info
        
        if info is not None:
            # Convert to dict if needed
            if hasattr(info, "to_dict"):
                return info.to_dict()
            return dict(info)
        return None
        
    except Exception as e:
        logger.error(f"Error collecting info for {fund_code}: {e}")
        return None


def collect_multiple_funds(
    fund_codes: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Collect data for multiple funds.
    
    Args:
        fund_codes: List of fund codes
        start_date: Start date
        end_date: End date
        
    Returns:
        Tuple of (dict mapping code to DataFrame, list of failed codes)
    """
    successful = {}
    failed = []
    
    for i, code in enumerate(fund_codes):
        logger.info(f"Collecting {code} ({i+1}/{len(fund_codes)})")
        
        data = collect_fund_data(code, start_date, end_date)
        if data is not None:
            successful[code] = data
        else:
            failed.append(code)
    
    logger.info(
        f"Collection complete: {len(successful)} successful, {len(failed)} failed"
    )
    
    return successful, failed


def collect_inflation_data() -> Optional[pd.DataFrame]:
    """
    Collect TÜFE (CPI) inflation data from TCMB via borsapy.
    
    Returns:
        DataFrame with monthly TÜFE values, or None if collection fails
    """
    try:
        inflation = bp.Inflation()
        tufe = inflation.tufe()
        
        if tufe is None or len(tufe) == 0:
            logger.warning("No TÜFE data returned")
            return None
        
        # Ensure datetime index
        if not isinstance(tufe.index, pd.DatetimeIndex):
            tufe.index = pd.to_datetime(tufe.index)
        
        logger.info(f"Collected {len(tufe)} months of TÜFE data")
        return tufe
        
    except Exception as e:
        logger.error(f"Error collecting inflation data: {e}")
        return None


def save_raw_data(
    fund_data: Dict[str, pd.DataFrame],
    inflation_data: Optional[pd.DataFrame],
    output_dir: Optional[Path] = None,
) -> None:
    """
    Save collected raw data to disk.
    
    Args:
        fund_data: Dictionary mapping fund codes to DataFrames
        inflation_data: TÜFE DataFrame
        output_dir: Output directory path
    """
    out = output_dir or DATA_RAW
    out.mkdir(parents=True, exist_ok=True)
    
    # Save fund data
    for code, df in fund_data.items():
        filepath = out / f"fund_{code}.csv"
        df.to_csv(filepath)
        logger.info(f"Saved {filepath}")
    
    # Save inflation data
    if inflation_data is not None:
        filepath = out / "tufe.csv"
        inflation_data.to_csv(filepath)
        logger.info(f"Saved {filepath}")
    
    # Save fund list
    fund_list = pd.DataFrame({
        "code": list(fund_data.keys()),
        "data_points": [len(df) for df in fund_data.values()],
    })
    fund_list.to_csv(out / "fund_list.csv", index=False)


def load_raw_data(input_dir: Optional[Path] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load previously saved raw data.
    
    Args:
        input_dir: Directory containing raw data files
        
    Returns:
        Tuple of (fund_data dict, inflation DataFrame)
    """
    inp = input_dir or DATA_RAW
    
    fund_data = {}
    
    # Load fund list
    fund_list_path = inp / "fund_list.csv"
    if fund_list_path.exists():
        fund_list = pd.read_csv(fund_list_path)
        
        for code in fund_list["code"]:
            filepath = inp / f"fund_{code}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                fund_data[code] = df
    
    # Load inflation data
    tufe_path = inp / "tufe.csv"
    inflation_data = None
    if tufe_path.exists():
        inflation_data = pd.read_csv(tufe_path, index_col=0, parse_dates=True)
    
    return fund_data, inflation_data


if __name__ == "__main__":
    # Example usage / quick test
    print("Searching for funds...")
    funds = search_funds()
    print(f"Found {len(funds)} funds")
    print(funds.head(10))
    
    print("\nCollecting inflation data...")
    tufe = collect_inflation_data()
    if tufe is not None:
        print(tufe.tail())
