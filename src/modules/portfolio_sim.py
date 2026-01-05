"""
TEFAS Fund Analysis - Module 3: Portfolio Simulation Engine
=============================================================
CRISP-DM: Modeling + Evaluation Phase

IMPORTANT DISCLAIMER:
This module provides HISTORICAL SIMULATION ONLY.
It is NOT investment advice. All outputs are for educational purposes.

This module handles:
1. Rule-based fund filtering (participation finance, constraints)
2. ANN-assisted scoring (auxiliary signal only)
3. Portfolio construction (equal weight or inverse volatility)
4. Historical backtesting simulation

Key Design Decisions:
- ANN provides ranking signal, but selection is rule-based
- Explicit constraints (max drawdown, diversification)
- Transparent weighting schemes
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    BACKTEST_CONFIG,
    DISCLAIMER,
    PORTFOLIO_CONSTRAINTS,
    SCORER_ANN_CONFIG,
)
from src.models.risk_scorer import create_risk_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """
    Simulates rule-based portfolio construction and backtesting.
    
    NOT INVESTMENT ADVICE - Educational simulation only.
    
    Workflow:
    1. Filter funds by risk segment and constraints
    2. Use ANN for auxiliary risk-return scoring
    3. Apply RULE-BASED selection (top K by score within constraints)
    4. Weight by equal weight or inverse volatility
    5. Run historical backtest simulation
    """
    
    def __init__(
        self,
        constraints: dict = None,
        backtest_config: dict = None,
        scorer_config: dict = None,
    ):
        """
        Initialize PortfolioSimulator.
        
        Args:
            constraints: Portfolio constraints dict
            backtest_config: Backtesting configuration
            scorer_config: ANN scorer configuration
        """
        self.constraints = constraints or PORTFOLIO_CONSTRAINTS.copy()
        self.backtest_config = backtest_config or BACKTEST_CONFIG.copy()
        self.scorer_config = scorer_config or SCORER_ANN_CONFIG.copy()
        
        self.scorer = None
        self.selected_funds = None
        self.weights = None
        self.backtest_results = None
    
    def filter_by_segment(
        self,
        risk_profiles: pd.DataFrame,
        segment: str = "Low",
    ) -> pd.Index:
        """
        Filter funds by risk segment.
        
        Args:
            risk_profiles: Results from RiskProfileExtractor
            segment: 'Low', 'Medium', or 'High'
            
        Returns:
            Index of eligible fund codes
        """
        eligible = risk_profiles[risk_profiles["risk_segment"] == segment].index
        
        logger.info(f"Segment '{segment}': {len(eligible)} eligible funds")
        
        return eligible
    
    def filter_by_constraints(
        self,
        fund_codes: pd.Index,
        risk_profiles: pd.DataFrame,
    ) -> pd.Index:
        """
        Apply constraint-based filtering.
        
        Constraints applied:
        - Maximum drawdown threshold
        - (Additional constraints can be added here)
        
        Args:
            fund_codes: Input fund codes
            risk_profiles: Risk profile data
            
        Returns:
            Filtered fund codes
        """
        max_dd_threshold = self.constraints.get("max_drawdown_threshold", 0.15)
        
        # Filter by max drawdown
        profiles = risk_profiles.loc[fund_codes]
        eligible = profiles[profiles["max_drawdown"] <= max_dd_threshold].index
        
        filtered = len(fund_codes) - len(eligible)
        if filtered > 0:
            logger.info(f"Filtered {filtered} funds exceeding drawdown threshold ({max_dd_threshold:.0%})")
        
        return eligible
    
    def filter_participation_funds(
        self,
        fund_codes: pd.Index,
        fund_names: Dict[str, str] = None,
        participation_patterns: List[str] = None,
    ) -> pd.Index:
        """
        Filter for participation (Islamic finance) compatible funds.
        
        Args:
            fund_codes: Input fund codes
            fund_names: Dict mapping code to name
            participation_patterns: Patterns to match
            
        Returns:
            Participation-compatible fund codes
        """
        if fund_names is None:
            # If no names provided, return all (filter at selection time)
            logger.warning("No fund names provided for participation filter")
            return fund_codes
        
        if participation_patterns is None:
            participation_patterns = ["katılım", "sukuk", "faizsiz", "islami"]
        
        eligible = []
        for code in fund_codes:
            name = fund_names.get(code, "").lower()
            if any(pattern in name for pattern in participation_patterns):
                eligible.append(code)
        
        logger.info(f"Participation filter: {len(eligible)}/{len(fund_codes)} funds pass")
        
        return pd.Index(eligible)
    
    def train_scorer(
        self,
        features: pd.DataFrame,
        risk_profiles: pd.DataFrame,
        verbose: int = 1,
    ) -> None:
        """
        Train ANN scorer for risk-return ranking.
        
        Note: This is an auxiliary signal. Final selection is rule-based.
        
        Args:
            features: Feature DataFrame from risk profiling
            risk_profiles: Risk profiles DataFrame
            verbose: Training verbosity
        """
        # Align indices
        common = features.index.intersection(risk_profiles.index)
        X = features.loc[common]
        profiles = risk_profiles.loc[common]
        
        # Create scorer
        self.scorer = create_risk_scorer(
            input_dim=X.shape[1],
            config=self.scorer_config,
        )
        
        # Train
        self.scorer.fit(
            features=X.values,
            returns=profiles["return_mean_annual"].values if "return_mean_annual" in profiles.columns 
                    else profiles["volatility_annual"].values * -1,  # Use inverse vol if no returns
            risk_scores=profiles["composite_risk_score"].values,
            epochs=self.scorer_config.get("epochs", 100),
            verbose=verbose,
        )
        
        logger.info("ANN scorer trained (auxiliary signal)")
    
    def select_funds(
        self,
        eligible_funds: pd.Index,
        features: pd.DataFrame,
        top_n: int = None,
    ) -> pd.Index:
        """
        Select top funds using ANN scores within constraints.
        
        Selection is RULE-BASED: Take top N by ANN score.
        
        Args:
            eligible_funds: Funds passing all filters
            features: Feature DataFrame
            top_n: Number of funds to select (uses constraint default)
            
        Returns:
            Selected fund codes
        """
        if top_n is None:
            top_n = self.constraints.get("max_funds", 10)
        
        min_funds = self.constraints.get("min_funds", 3)
        
        if len(eligible_funds) < min_funds:
            logger.warning(
                f"Only {len(eligible_funds)} eligible funds "
                f"(minimum: {min_funds})"
            )
            self.selected_funds = eligible_funds
            return eligible_funds
        
        # Get ANN scores for eligible funds
        eligible_features = features.loc[eligible_funds]
        
        if self.scorer is not None:
            scores = self.scorer.predict(eligible_features.values)
            score_series = pd.Series(scores, index=eligible_funds)
            
            # Select top N by score
            selected = score_series.nlargest(min(top_n, len(eligible_funds))).index
        else:
            # Fallback: random selection if no scorer
            logger.warning("No scorer available, selecting randomly")
            n_select = min(top_n, len(eligible_funds))
            selected = eligible_funds[:n_select]
        
        self.selected_funds = selected
        
        logger.info(f"Selected {len(selected)} funds for portfolio")
        
        return selected
    
    def calculate_weights(
        self,
        method: str = "equal",
        risk_profiles: pd.DataFrame = None,
    ) -> pd.Series:
        """
        Calculate portfolio weights.
        
        Methods:
        - 'equal': Equal weight for all funds
        - 'inverse_vol': Inverse volatility weighting
        
        Args:
            method: Weighting method
            risk_profiles: Risk profiles (needed for inverse_vol)
            
        Returns:
            Series of weights per fund
        """
        if self.selected_funds is None or len(self.selected_funds) == 0:
            raise ValueError("Select funds first")
        
        max_weight = self.constraints.get("max_weight_per_fund", 0.30)
        n_funds = len(self.selected_funds)
        
        if method == "equal":
            raw_weights = pd.Series(
                1.0 / n_funds,
                index=self.selected_funds,
            )
        
        elif method == "inverse_vol":
            if risk_profiles is None:
                logger.warning("No risk profiles for inverse_vol, using equal weights")
                return self.calculate_weights(method="equal")
            
            vols = risk_profiles.loc[self.selected_funds, "volatility_annual"]
            inv_vols = 1.0 / vols
            raw_weights = inv_vols / inv_vols.sum()
        
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Apply max weight constraint
        raw_weights = raw_weights.clip(upper=max_weight)
        
        # Renormalize to sum to 1
        self.weights = raw_weights / raw_weights.sum()
        
        logger.info(f"Weights calculated ({method}): max={self.weights.max():.2%}")
        
        return self.weights
    
    def backtest(
        self,
        returns: pd.DataFrame,
        weights: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Run historical backtest simulation.
        
        NOT INVESTMENT ADVICE - Educational simulation only.
        
        Args:
            returns: Returns DataFrame (dates x funds)
            weights: Portfolio weights (uses stored if None)
            
        Returns:
            DataFrame with backtest results
        """
        if weights is None:
            weights = self.weights
        
        if weights is None:
            raise ValueError("Calculate weights first")
        
        # Get returns for selected funds only
        portfolio_returns = returns[weights.index]
        
        # Calculate weighted portfolio returns
        weighted_returns = portfolio_returns.mul(weights, axis=1).sum(axis=1)
        
        # Apply transaction costs (simplified)
        transaction_cost = self.backtest_config.get("transaction_cost", 0.001)
        # Deduct cost at start
        initial_cost = transaction_cost * len(weights)
        
        # Calculate cumulative returns
        initial_capital = self.backtest_config.get("initial_capital", 100000)
        portfolio_value = initial_capital * (1 - initial_cost)
        
        values = [portfolio_value]
        for ret in weighted_returns:
            portfolio_value *= (1 + ret)
            values.append(portfolio_value)
        
        # Create results DataFrame
        results = pd.DataFrame({
            "date": list(weighted_returns.index) + [weighted_returns.index[-1]],
            "portfolio_value": values[:len(weighted_returns) + 1],
        })
        results["date"] = pd.to_datetime(results["date"])
        results = results.set_index("date")
        
        # Remove duplicate index
        results = results[~results.index.duplicated(keep="first")]
        
        # Calculate additional metrics
        total_return = (results["portfolio_value"].iloc[-1] / initial_capital) - 1
        
        # Add returns column
        results["daily_return"] = results["portfolio_value"].pct_change()
        
        # Calculate drawdown
        running_max = results["portfolio_value"].cummax()
        results["drawdown"] = (results["portfolio_value"] - running_max) / running_max
        
        self.backtest_results = results
        
        logger.info(f"Backtest complete: Total return = {total_return:.2%}")
        
        return results
    
    def calculate_backtest_metrics(self) -> Dict:
        """
        Calculate performance metrics from backtest.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.backtest_results is None:
            raise ValueError("Run backtest first")
        
        results = self.backtest_results
        daily_returns = results["daily_return"].dropna()
        
        initial = self.backtest_config.get("initial_capital", 100000)
        final = results["portfolio_value"].iloc[-1]
        
        # Days and annualization
        n_days = len(daily_returns)
        n_years = n_days / 252
        
        metrics = {
            "initial_capital": initial,
            "final_value": final,
            "total_return": (final / initial) - 1,
            "annualized_return": (final / initial) ** (1 / n_years) - 1 if n_years > 0 else 0,
            "volatility_annual": daily_returns.std() * np.sqrt(252),
            "sharpe_ratio": (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0,
            "max_drawdown": results["drawdown"].min(),
            "n_trading_days": n_days,
            "n_funds": len(self.selected_funds),
        }
        
        return metrics
    
    def run(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        risk_profiles: pd.DataFrame,
        segment: str = "Low",
        fund_names: Dict[str, str] = None,
        weighting_method: str = "equal",
        verbose: int = 1,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run full portfolio simulation pipeline.
        
        Args:
            returns: Returns DataFrame
            features: Feature DataFrame
            risk_profiles: Risk profiles from Module 1
            segment: Risk segment to use
            fund_names: Fund name dictionary for participation filter
            weighting_method: 'equal' or 'inverse_vol'
            verbose: Verbosity
            
        Returns:
            Tuple of (backtest results DataFrame, metrics dict)
        """
        print(DISCLAIMER)
        
        # Step 1: Filter by segment
        eligible = self.filter_by_segment(risk_profiles, segment)
        
        # Step 2: Filter by constraints
        eligible = self.filter_by_constraints(eligible, risk_profiles)
        
        # Step 3: Filter for participation funds (if names provided)
        if fund_names is not None:
            eligible = self.filter_participation_funds(eligible, fund_names)
        
        if len(eligible) == 0:
            logger.error("No eligible funds after filtering")
            return None, {}
        
        # Step 4: Train scorer (on all data, not just eligible)
        self.train_scorer(features, risk_profiles, verbose=verbose)
        
        # Step 5: Select funds
        self.select_funds(eligible, features)
        
        # Step 6: Calculate weights
        self.calculate_weights(method=weighting_method, risk_profiles=risk_profiles)
        
        # Step 7: Run backtest
        backtest_results = self.backtest(returns)
        
        # Step 8: Calculate metrics
        metrics = self.calculate_backtest_metrics()
        
        print(self.summary())
        
        return backtest_results, metrics
    
    def summary(self) -> str:
        """Generate human-readable summary with disclaimer."""
        lines = [
            "=" * 60,
            "PORTFOLIO SIMULATION RESULTS",
            "=" * 60,
            "",
            "⚠️  NOT INVESTMENT ADVICE - EDUCATIONAL SIMULATION ONLY ⚠️",
            "",
        ]
        
        if self.selected_funds is not None:
            lines.extend([
                f"Selected Funds ({len(self.selected_funds)}):",
            ])
            for fund in self.selected_funds:
                weight = self.weights.get(fund, 0)
                lines.append(f"  {fund}: {weight:.2%}")
        
        if self.backtest_results is not None:
            metrics = self.calculate_backtest_metrics()
            lines.extend([
                "",
                "Backtest Performance:",
                f"  Total Return: {metrics['total_return']:.2%}",
                f"  Annualized Return: {metrics['annualized_return']:.2%}",
                f"  Annual Volatility: {metrics['volatility_annual']:.2%}",
                f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
                f"  Max Drawdown: {metrics['max_drawdown']:.2%}",
                f"  Trading Days: {metrics['n_trading_days']}",
            ])
        
        lines.extend([
            "",
            "⚠️  Past performance does not guarantee future results ⚠️",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def simulate_portfolio(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    risk_profiles: pd.DataFrame,
    segment: str = "Low",
    fund_names: Dict[str, str] = None,
    weighting_method: str = "equal",
    verbose: int = 1,
) -> Tuple[PortfolioSimulator, pd.DataFrame, Dict]:
    """
    Convenience function to run portfolio simulation.
    
    NOT INVESTMENT ADVICE.
    
    Args:
        returns: Returns DataFrame
        features: Feature DataFrame
        risk_profiles: Risk profiles
        segment: Risk segment
        fund_names: Fund names for participation filter
        weighting_method: Weighting method
        verbose: Verbosity
        
    Returns:
        Tuple of (simulator, backtest_results, metrics)
    """
    simulator = PortfolioSimulator()
    backtest_results, metrics = simulator.run(
        returns=returns,
        features=features,
        risk_profiles=risk_profiles,
        segment=segment,
        fund_names=fund_names,
        weighting_method=weighting_method,
        verbose=verbose,
    )
    
    return simulator, backtest_results, metrics


if __name__ == "__main__":
    print(DISCLAIMER)
    print("\nPortfolio Simulator module loaded")
    print("\nUsage:")
    print("  sim, results, metrics = simulate_portfolio(returns, features, risk_profiles)")
