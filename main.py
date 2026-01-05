#!/usr/bin/env python3
"""
TEFAS Fund Analysis - Main Entry Point
========================================
CRISP-DM Compliant Deep Learning Project

University Course: Deep Learning (YBSB 4007)
Conceptual Stakeholder: KuveytTürk Portföy Management Team

DISCLAIMER:
This is an educational project. All outputs are simulations.
NOT investment advice. Consult a licensed advisor for actual investments.

Usage:
    python main.py --help
    python main.py --collect          # Collect data only
    python main.py --analyze          # Run full analysis
    python main.py --segment Low      # Simulate portfolio for segment
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PROCESSED,
    DATA_RAW,
    DISCLAIMER,
    END_DATE,
    OUTPUT_FIGURES,
    OUTPUT_REPORTS,
    START_DATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_data():
    """
    CRISP-DM Phase 2: Data Understanding
    
    Collect fund data from TEFAS and inflation data from TCMB.
    """
    from src.data.collector import (
        collect_inflation_data,
        collect_multiple_funds,
        save_raw_data,
        search_funds,
    )
    
    logger.info("=" * 60)
    logger.info("DATA COLLECTION")
    logger.info("=" * 60)
    
    # Search for funds
    logger.info("Searching for funds...")
    fund_search = search_funds()
    
    if len(fund_search) == 0:
        logger.error("No funds found. Check network connection or keywords.")
        return None, None
    
    logger.info(f"Found {len(fund_search)} funds")
    
    # Get fund codes
    code_col = "code" if "code" in fund_search.columns else fund_search.columns[0]
    fund_codes = fund_search[code_col].tolist()
    
    # Collect fund data
    logger.info(f"Collecting data for {len(fund_codes)} funds...")
    fund_data, failed = collect_multiple_funds(fund_codes)
    
    if not fund_data:
        logger.error("No fund data collected successfully")
        return None, None
    
    # Collect inflation data
    logger.info("Collecting TÜFE inflation data...")
    tufe = collect_inflation_data()
    
    # Save raw data
    logger.info("Saving raw data...")
    save_raw_data(fund_data, tufe)
    
    logger.info(f"Data collection complete: {len(fund_data)} funds saved")
    
    return fund_data, tufe


def prepare_data(fund_data=None, tufe=None):
    """
    CRISP-DM Phase 3: Data Preparation
    
    Preprocess data: align dates, calculate returns, adjust for inflation.
    """
    from src.data.collector import load_raw_data
    from src.data.preprocessor import prepare_return_matrix
    
    logger.info("=" * 60)
    logger.info("DATA PREPARATION")
    logger.info("=" * 60)
    
    # Load data if not provided
    if fund_data is None or tufe is None:
        logger.info("Loading raw data from disk...")
        fund_data, tufe = load_raw_data()
    
    if not fund_data:
        logger.error("No fund data available")
        return None, None
    
    # Prepare return matrix
    logger.info("Preparing return matrix...")
    returns, prices = prepare_return_matrix(fund_data, tufe, adjust_inflation=True)
    
    # Save processed data
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    returns.to_csv(DATA_PROCESSED / "returns.csv")
    prices.to_csv(DATA_PROCESSED / "prices.csv")
    
    logger.info(f"Prepared data: {returns.shape[0]} days × {returns.shape[1]} funds")
    
    return returns, prices


def run_module1(returns, prices):
    """
    MODULE 1: Fund Risk Profile Extractor
    
    Extract risk profiles, detect anomalies, segment funds.
    """
    from src.modules.risk_profiler import extract_risk_profiles
    
    logger.info("=" * 60)
    logger.info("MODULE 1: RISK PROFILE EXTRACTION")
    logger.info("=" * 60)
    
    extractor, features, risk_profiles = extract_risk_profiles(
        returns=returns,
        prices=prices,
        verbose=1,
    )
    
    # Save results
    features.to_csv(DATA_PROCESSED / "risk_features.csv")
    risk_profiles.to_csv(DATA_PROCESSED / "risk_profiles.csv")
    
    return extractor, features, risk_profiles


def run_module2(returns):
    """
    MODULE 2: Hidden Correlation & Diversification Map
    
    Compute embeddings and identify diversification illusions.
    """
    from src.modules.correlation_map import compute_correlation_map
    
    logger.info("=" * 60)
    logger.info("MODULE 2: CORRELATION MAPPING")
    logger.info("=" * 60)
    
    mapper, corr_results, cluster_analysis = compute_correlation_map(
        returns=returns,
        n_clusters=5,
        verbose=1,
    )
    
    # Save results
    corr_results.to_csv(DATA_PROCESSED / "embeddings.csv")
    cluster_analysis.to_csv(DATA_PROCESSED / "cluster_analysis.csv")
    
    if mapper.correlation_matrix is not None:
        mapper.correlation_matrix.to_csv(DATA_PROCESSED / "correlation_matrix.csv")
    
    return mapper, corr_results, cluster_analysis


def run_module3(returns, features, risk_profiles, segment="Low"):
    """
    MODULE 3: Portfolio Basket Simulation
    
    NOT INVESTMENT ADVICE - Educational simulation only.
    """
    from src.modules.portfolio_sim import simulate_portfolio
    
    logger.info("=" * 60)
    logger.info("MODULE 3: PORTFOLIO SIMULATION")
    logger.info(f"Target Segment: {segment}")
    logger.info("=" * 60)
    
    simulator, backtest_results, metrics = simulate_portfolio(
        returns=returns,
        features=features,
        risk_profiles=risk_profiles,
        segment=segment,
        weighting_method="equal",
        verbose=1,
    )
    
    if backtest_results is not None:
        backtest_results.to_csv(DATA_PROCESSED / f"backtest_{segment.lower()}.csv")
    
    return simulator, backtest_results, metrics


def run_evaluation(
    risk_profiles,
    corr_results,
    cluster_analysis,
    correlation_matrix,
    backtest_results,
):
    """
    CRISP-DM Phase 5: Evaluation
    
    Evaluate all modules and generate visualizations.
    """
    from src.evaluation.metrics import (
        evaluate_risk_segmentation,
        evaluate_backtest,
        generate_evaluation_report,
    )
    from src.evaluation.visualizations import create_all_visualizations
    
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    # Evaluate segmentation
    seg_metrics = evaluate_risk_segmentation(risk_profiles)
    
    # Evaluate backtest
    bt_metrics = None
    if backtest_results is not None:
        bt_metrics = evaluate_backtest(backtest_results)
    
    # Generate report
    report = generate_evaluation_report(
        segmentation_metrics=seg_metrics,
        backtest_metrics=bt_metrics,
    )
    
    print(report)
    
    # Save report
    OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORTS / "evaluation_report.txt", "w") as f:
        f.write(report)
    
    # Create visualizations
    create_all_visualizations(
        risk_profiles=risk_profiles,
        correlation_results=corr_results,
        cluster_analysis=cluster_analysis,
        correlation_matrix=correlation_matrix,
        backtest_results=backtest_results,
    )
    
    logger.info(f"Evaluation complete. Reports saved to {OUTPUT_REPORTS}")
    
    return report


def run_full_pipeline(segment="Low"):
    """
    Run the complete analysis pipeline.
    """
    print(DISCLAIMER)
    
    logger.info("=" * 60)
    logger.info("TEFAS FUND ANALYSIS - FULL PIPELINE")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info("=" * 60)
    
    # Phase 1: Data Collection
    fund_data, tufe = collect_data()
    
    if fund_data is None:
        logger.error("Data collection failed. Exiting.")
        return
    
    # Phase 2: Data Preparation
    returns, prices = prepare_data(fund_data, tufe)
    
    if returns is None:
        logger.error("Data preparation failed. Exiting.")
        return
    
    # Phase 3: Module 1 - Risk Profiling
    extractor, features, risk_profiles = run_module1(returns, prices)
    
    # Phase 4: Module 2 - Correlation Mapping
    mapper, corr_results, cluster_analysis = run_module2(returns)
    
    # Phase 5: Module 3 - Portfolio Simulation
    simulator, backtest_results, metrics = run_module3(
        returns, features, risk_profiles, segment=segment
    )
    
    # Phase 6: Evaluation
    report = run_evaluation(
        risk_profiles=risk_profiles,
        corr_results=corr_results,
        cluster_analysis=cluster_analysis,
        correlation_matrix=mapper.correlation_matrix,
        backtest_results=backtest_results,
    )
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return {
        "returns": returns,
        "prices": prices,
        "features": features,
        "risk_profiles": risk_profiles,
        "embeddings": corr_results,
        "clusters": cluster_analysis,
        "backtest": backtest_results,
        "metrics": metrics,
    }


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="TEFAS Fund Analysis - Deep Learning Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --collect         Collect data only
  python main.py --analyze         Run full analysis with default settings
  python main.py --segment Medium  Run portfolio simulation for Medium risk

DISCLAIMER: This is an educational project. NOT investment advice.
        """,
    )
    
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect data from TEFAS and TCMB",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare data (load raw, preprocess)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run full analysis pipeline",
    )
    parser.add_argument(
        "--segment",
        choices=["Low", "Medium", "High"],
        default="Low",
        help="Risk segment for portfolio simulation (default: Low)",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run exploratory data analysis only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(DISCLAIMER)
    
    if args.collect:
        collect_data()
    elif args.prepare:
        prepare_data()
    elif args.eda:
        # Run EDA only
        from src.data.collector import load_raw_data
        from src.data.preprocessor import prepare_return_matrix
        from src.data.exploratory import run_eda
        
        fund_data, tufe = load_raw_data()
        if fund_data:
            returns, prices = prepare_return_matrix(fund_data, tufe)
            run_eda(fund_data, returns, prices)
        else:
            logger.error("No data available. Run --collect first.")
    elif args.analyze:
        run_full_pipeline(segment=args.segment)
    else:
        # Default: run full pipeline
        print("\nNo action specified. Running full pipeline...")
        print("Use --help for options.\n")
        run_full_pipeline(segment=args.segment)


if __name__ == "__main__":
    main()
