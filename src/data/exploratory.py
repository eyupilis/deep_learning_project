"""
TEFAS Fund Analysis - Exploratory Data Analysis Module
========================================================
CRISP-DM: Data Understanding Phase

Comprehensive EDA utilities for:
- Fund universe overview
- Return distribution analysis
- Correlation structure exploration
- Time series decomposition
- Anomaly detection (statistical)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUT_FIGURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")


class ExploratoryAnalysis:
    """
    Comprehensive EDA toolkit for TEFAS fund data.
    
    Methods follow a consistent pattern:
    1. Compute statistics
    2. Generate visualization
    3. Return structured results
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize EDA toolkit.
        
        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = output_dir or OUTPUT_FIGURES
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # FUND UNIVERSE ANALYSIS
    # =========================================================================
    
    def fund_universe_summary(
        self,
        fund_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate summary statistics for the fund universe.
        
        Args:
            fund_data: Dict mapping fund codes to DataFrames
            
        Returns:
            Summary DataFrame with one row per fund
        """
        summaries = []
        
        for code, df in fund_data.items():
            # Find price column
            price_col = None
            for col in ["close", "Close", "price", "adj_close"]:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                continue
            
            prices = df[price_col].dropna()
            
            if len(prices) < 2:
                continue
            
            returns = prices.pct_change().dropna()
            
            summary = {
                "fund_code": code,
                "start_date": prices.index.min(),
                "end_date": prices.index.max(),
                "n_observations": len(prices),
                "first_price": prices.iloc[0],
                "last_price": prices.iloc[-1],
                "total_return": (prices.iloc[-1] / prices.iloc[0]) - 1,
                "mean_daily_return": returns.mean(),
                "std_daily_return": returns.std(),
                "min_daily_return": returns.min(),
                "max_daily_return": returns.max(),
                "missing_pct": df[price_col].isna().mean() * 100,
            }
            
            summaries.append(summary)
        
        summary_df = pd.DataFrame(summaries).set_index("fund_code")
        
        logger.info(f"Generated summary for {len(summary_df)} funds")
        
        return summary_df
    
    def plot_fund_universe(
        self,
        summary_df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """
        Visualize fund universe characteristics.
        
        Args:
            summary_df: Output from fund_universe_summary
            save: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Total returns distribution
        ax = axes[0, 0]
        ax.hist(summary_df["total_return"] * 100, bins=20, edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Total Return (%)")
        ax.set_ylabel("Number of Funds")
        ax.set_title("Distribution of Total Returns")
        
        # 2. Daily volatility distribution
        ax = axes[0, 1]
        ax.hist(summary_df["std_daily_return"] * 100, bins=20, 
                color="orange", edgecolor="black")
        ax.set_xlabel("Daily Volatility (%)")
        ax.set_ylabel("Number of Funds")
        ax.set_title("Distribution of Daily Volatility")
        
        # 3. Return vs Volatility scatter
        ax = axes[1, 0]
        ax.scatter(
            summary_df["std_daily_return"] * 100,
            summary_df["mean_daily_return"] * 100,
            alpha=0.6,
            s=50,
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Daily Volatility (%)")
        ax.set_ylabel("Mean Daily Return (%)")
        ax.set_title("Risk-Return Profile")
        
        # 4. Data availability
        ax = axes[1, 1]
        ax.hist(summary_df["n_observations"], bins=20, 
                color="green", edgecolor="black")
        ax.set_xlabel("Number of Observations")
        ax.set_ylabel("Number of Funds")
        ax.set_title("Data Availability")
        
        fig.suptitle("Fund Universe Overview", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "eda_fund_universe.png", dpi=150)
            logger.info(f"Saved: {self.output_dir / 'eda_fund_universe.png'}")
        
        return fig
    
    # =========================================================================
    # RETURN DISTRIBUTION ANALYSIS
    # =========================================================================
    
    def return_distribution_analysis(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze return distributions for all funds.
        
        Tests for normality and calculates distribution moments.
        
        Args:
            returns: Returns DataFrame (dates x funds)
            
        Returns:
            DataFrame with distribution statistics
        """
        results = []
        
        for fund in returns.columns:
            r = returns[fund].dropna()
            
            if len(r) < 30:
                continue
            
            # Moments
            mean = r.mean()
            std = r.std()
            skew = r.skew()
            kurt = r.kurtosis()
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(r.sample(min(5000, len(r))))
            jb_stat, jb_p = stats.jarque_bera(r)
            
            results.append({
                "fund_code": fund,
                "mean": mean,
                "std": std,
                "skewness": skew,
                "kurtosis": kurt,
                "is_left_skewed": skew < -0.5,
                "is_right_skewed": skew > 0.5,
                "is_leptokurtic": kurt > 0,  # Fat tails
                "shapiro_p": shapiro_p,
                "jarque_bera_p": jb_p,
                "is_normal_95": jb_p > 0.05,
            })
        
        results_df = pd.DataFrame(results).set_index("fund_code")
        
        # Summary statistics
        n_normal = results_df["is_normal_95"].sum()
        n_fat_tails = results_df["is_leptokurtic"].sum()
        
        logger.info(
            f"Return analysis: {n_normal}/{len(results_df)} normal, "
            f"{n_fat_tails}/{len(results_df)} with fat tails"
        )
        
        return results_df
    
    def plot_return_distributions(
        self,
        returns: pd.DataFrame,
        n_funds: int = 6,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot return distribution histograms for sample funds.
        
        Args:
            returns: Returns DataFrame
            n_funds: Number of funds to show
            save: Whether to save
            
        Returns:
            Matplotlib figure
        """
        funds = returns.columns[:n_funds]
        n_cols = 3
        n_rows = (len(funds) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()
        
        for i, fund in enumerate(funds):
            ax = axes[i]
            r = returns[fund].dropna()
            
            # Histogram
            ax.hist(r * 100, bins=50, density=True, alpha=0.7, label="Actual")
            
            # Normal fit
            x = np.linspace(r.min() * 100, r.max() * 100, 100)
            normal_pdf = stats.norm.pdf(x, r.mean() * 100, r.std() * 100)
            ax.plot(x, normal_pdf, "r-", linewidth=2, label="Normal Fit")
            
            ax.set_xlabel("Daily Return (%)")
            ax.set_ylabel("Density")
            ax.set_title(f"{fund}")
            ax.legend(fontsize=8)
        
        # Hide unused axes
        for j in range(len(funds), len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle("Return Distributions vs Normal", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "eda_return_distributions.png", dpi=150)
        
        return fig
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    
    def correlation_analysis(
        self,
        returns: pd.DataFrame,
    ) -> Dict:
        """
        Analyze correlation structure of returns.
        
        Args:
            returns: Returns DataFrame
            
        Returns:
            Dictionary with correlation analysis results
        """
        corr = returns.corr()
        
        # Extract upper triangle (excluding diagonal)
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        correlations = upper_tri.stack().values
        
        results = {
            "correlation_matrix": corr,
            "mean_correlation": np.mean(correlations),
            "median_correlation": np.median(correlations),
            "min_correlation": np.min(correlations),
            "max_correlation": np.max(correlations),
            "std_correlation": np.std(correlations),
            "n_high_positive": (correlations > 0.7).sum(),
            "n_negative": (correlations < 0).sum(),
            "total_pairs": len(correlations),
        }
        
        logger.info(
            f"Correlation analysis: mean={results['mean_correlation']:.3f}, "
            f"high corr pairs={results['n_high_positive']}"
        )
        
        return results
    
    def plot_correlation_analysis(
        self,
        returns: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """
        Visualize correlation structure.
        
        Args:
            returns: Returns DataFrame
            save: Whether to save
            
        Returns:
            Matplotlib figure
        """
        corr = returns.corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Correlation heatmap
        ax = axes[0]
        if len(corr) <= 20:
            sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, ax=ax,
                       annot=True, fmt=".2f", annot_kws={"size": 7})
        else:
            sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, ax=ax)
        ax.set_title("Correlation Matrix")
        
        # 2. Correlation distribution
        ax = axes[1]
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        correlations = upper_tri.stack().values
        
        ax.hist(correlations, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=np.mean(correlations), color="red", linestyle="--",
                   label=f"Mean: {np.mean(correlations):.3f}")
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Pairwise Correlation")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Pairwise Correlations")
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "eda_correlation_analysis.png", dpi=150)
        
        return fig
    
    # =========================================================================
    # TIME SERIES ANALYSIS
    # =========================================================================
    
    def time_series_analysis(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze time series characteristics.
        
        Args:
            prices: Price DataFrame
            
        Returns:
            DataFrame with time series statistics
        """
        results = []
        
        for fund in prices.columns:
            p = prices[fund].dropna()
            r = p.pct_change().dropna()
            
            if len(r) < 30:
                continue
            
            # Autocorrelation at lag 1
            autocorr_1 = r.autocorr(lag=1)
            
            # Trend (simple linear regression slope)
            x = np.arange(len(p))
            slope, intercept, r_value, _, _ = stats.linregress(x, p.values)
            trend_r2 = r_value ** 2
            
            # Volatility clustering (autocorr of squared returns)
            vol_cluster = (r ** 2).autocorr(lag=1)
            
            results.append({
                "fund_code": fund,
                "autocorr_lag1": autocorr_1,
                "has_momentum": autocorr_1 > 0.1,
                "has_mean_reversion": autocorr_1 < -0.1,
                "trend_slope": slope,
                "trend_r2": trend_r2,
                "has_trend": trend_r2 > 0.5,
                "volatility_clustering": vol_cluster,
                "has_vol_clustering": vol_cluster > 0.1,
            })
        
        return pd.DataFrame(results).set_index("fund_code")
    
    def plot_time_series_sample(
        self,
        prices: pd.DataFrame,
        n_funds: int = 4,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot sample price time series with rolling statistics.
        
        Args:
            prices: Price DataFrame
            n_funds: Number of funds to show
            save: Whether to save
            
        Returns:
            Matplotlib figure
        """
        funds = prices.columns[:n_funds]
        
        fig, axes = plt.subplots(n_funds, 2, figsize=(14, 3 * n_funds))
        
        for i, fund in enumerate(funds):
            p = prices[fund].dropna()
            r = p.pct_change().dropna()
            
            # Price chart with moving averages
            ax = axes[i, 0]
            ax.plot(p.index, p.values, label="Price", alpha=0.8)
            ax.plot(p.index, p.rolling(21).mean().values, label="21-day MA", alpha=0.7)
            ax.plot(p.index, p.rolling(63).mean().values, label="63-day MA", alpha=0.7)
            ax.set_title(f"{fund} - Price")
            ax.legend(fontsize=8)
            ax.set_ylabel("Price")
            
            # Rolling volatility
            ax = axes[i, 1]
            vol = r.rolling(21).std() * np.sqrt(252) * 100
            ax.plot(vol.index, vol.values, color="orange")
            ax.fill_between(vol.index, 0, vol.values, alpha=0.3, color="orange")
            ax.set_title(f"{fund} - Rolling 21d Volatility")
            ax.set_ylabel("Annualized Vol (%)")
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "eda_time_series_sample.png", dpi=150)
        
        return fig
    
    # =========================================================================
    # OUTLIER DETECTION (STATISTICAL)
    # =========================================================================
    
    def detect_outliers(
        self,
        returns: pd.DataFrame,
        z_threshold: float = 3.0,
    ) -> Dict:
        """
        Detect statistical outliers in returns.
        
        Args:
            returns: Returns DataFrame
            z_threshold: Z-score threshold for outliers
            
        Returns:
            Dictionary with outlier analysis
        """
        outlier_counts = {}
        outlier_dates = {}
        
        for fund in returns.columns:
            r = returns[fund].dropna()
            z_scores = np.abs((r - r.mean()) / r.std())
            
            outliers = z_scores > z_threshold
            outlier_counts[fund] = outliers.sum()
            outlier_dates[fund] = r[outliers].index.tolist()
        
        # Find dates with multiple fund outliers
        all_dates = []
        for dates in outlier_dates.values():
            all_dates.extend(dates)
        
        date_counts = pd.Series(all_dates).value_counts()
        market_events = date_counts[date_counts >= 3].index.tolist()  # 3+ funds
        
        results = {
            "outlier_counts": outlier_counts,
            "outlier_dates": outlier_dates,
            "market_events": market_events,
            "total_outliers": sum(outlier_counts.values()),
            "avg_outliers_per_fund": np.mean(list(outlier_counts.values())),
        }
        
        logger.info(
            f"Outlier detection: {results['total_outliers']} total, "
            f"{len(market_events)} potential market events"
        )
        
        return results
    
    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================
    
    def generate_eda_report(
        self,
        fund_data: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> str:
        """
        Generate comprehensive EDA report.
        
        Args:
            fund_data: Raw fund data dict
            returns: Processed returns
            prices: Processed prices
            
        Returns:
            Formatted report string
        """
        # Run all analyses
        universe = self.fund_universe_summary(fund_data)
        dist_analysis = self.return_distribution_analysis(returns)
        corr_analysis = self.correlation_analysis(returns)
        ts_analysis = self.time_series_analysis(prices)
        outliers = self.detect_outliers(returns)
        
        lines = [
            "=" * 70,
            "EXPLORATORY DATA ANALYSIS REPORT",
            "=" * 70,
            "",
            "FUND UNIVERSE",
            "-" * 40,
            f"  Total funds: {len(universe)}",
            f"  Date range: {universe['start_date'].min()} to {universe['end_date'].max()}",
            f"  Avg observations per fund: {universe['n_observations'].mean():.0f}",
            f"  Total return range: {universe['total_return'].min():.2%} to {universe['total_return'].max():.2%}",
            "",
            "RETURN DISTRIBUTIONS",
            "-" * 40,
            f"  Normal (95% CI): {dist_analysis['is_normal_95'].sum()}/{len(dist_analysis)}",
            f"  Leptokurtic (fat tails): {dist_analysis['is_leptokurtic'].sum()}/{len(dist_analysis)}",
            f"  Left-skewed: {dist_analysis['is_left_skewed'].sum()}/{len(dist_analysis)}",
            f"  Right-skewed: {dist_analysis['is_right_skewed'].sum()}/{len(dist_analysis)}",
            "",
            "CORRELATION STRUCTURE",
            "-" * 40,
            f"  Mean pairwise correlation: {corr_analysis['mean_correlation']:.3f}",
            f"  High correlation pairs (>0.7): {corr_analysis['n_high_positive']}/{corr_analysis['total_pairs']}",
            f"  Negative correlation pairs: {corr_analysis['n_negative']}/{corr_analysis['total_pairs']}",
            "",
            "TIME SERIES CHARACTERISTICS",
            "-" * 40,
            f"  Funds with momentum: {ts_analysis['has_momentum'].sum()}/{len(ts_analysis)}",
            f"  Funds with vol clustering: {ts_analysis['has_vol_clustering'].sum()}/{len(ts_analysis)}",
            "",
            "OUTLIER DETECTION",
            "-" * 40,
            f"  Total outliers (|z|>3): {outliers['total_outliers']}",
            f"  Potential market events: {len(outliers['market_events'])}",
            "",
            "=" * 70,
        ]
        
        report = "\n".join(lines)
        
        # Save report
        with open(self.output_dir.parent / "reports" / "eda_report.txt", "w") as f:
            f.write(report)
        
        return report
    
    def run_full_eda(
        self,
        fund_data: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> str:
        """
        Run complete EDA with all visualizations.
        
        Args:
            fund_data: Raw fund data
            returns: Processed returns
            prices: Processed prices
            
        Returns:
            EDA report
        """
        logger.info("=" * 50)
        logger.info("STARTING EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 50)
        
        # Generate visualizations
        universe = self.fund_universe_summary(fund_data)
        self.plot_fund_universe(universe)
        self.plot_return_distributions(returns)
        self.plot_correlation_analysis(returns)
        self.plot_time_series_sample(prices)
        
        # Generate report
        report = self.generate_eda_report(fund_data, returns, prices)
        
        print(report)
        
        logger.info("EDA complete. Figures saved to: " + str(self.output_dir))
        
        return report


def run_eda(
    fund_data: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
) -> Tuple[ExploratoryAnalysis, str]:
    """
    Convenience function to run EDA.
    
    Args:
        fund_data: Raw fund data
        returns: Processed returns
        prices: Processed prices
        
    Returns:
        Tuple of (EDA object, report string)
    """
    eda = ExploratoryAnalysis()
    report = eda.run_full_eda(fund_data, returns, prices)
    return eda, report


if __name__ == "__main__":
    print("EDA module loaded")
    print("\nUsage:")
    print("  eda, report = run_eda(fund_data, returns, prices)")
