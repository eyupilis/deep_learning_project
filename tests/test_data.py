"""
TEFAS Fund Analysis - Unit Tests for Data Pipeline
====================================================
Tests for data collection, preprocessing, and feature engineering.
"""

import sys
from pathlib import Path
from datetime import date
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    np.random.seed(42)
    
    # Simulate 5 funds with different characteristics
    funds = {}
    for i, code in enumerate(["AAK", "BBK", "CCK", "DDK", "EEK"]):
        # Random walk with drift
        returns = np.random.normal(0.0005 + i * 0.0001, 0.01 + i * 0.002, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        funds[code] = pd.DataFrame({
            "close": prices,
            "fund_code": code,
        }, index=dates)
    
    return funds


@pytest.fixture
def sample_returns(sample_prices):
    """Generate returns from sample prices."""
    prices = pd.DataFrame({
        code: df["close"] for code, df in sample_prices.items()
    })
    return prices.pct_change().dropna()


@pytest.fixture
def sample_tufe():
    """Generate sample TÜFE data."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    np.random.seed(42)
    
    # Monthly inflation around 2%
    tufe_values = 100 * np.cumprod(1 + np.random.normal(0.02, 0.005, len(dates)))
    
    return pd.DataFrame({
        "tufe": tufe_values,
    }, index=dates)


# =============================================================================
# DATA COLLECTION TESTS
# =============================================================================

class TestDataCollection:
    """Tests for src/data/collector.py"""
    
    def test_is_participation_fund_positive(self):
        """Test participation fund detection - positive cases."""
        from src.data.collector import is_participation_fund
        
        assert is_participation_fund("Kuveyt Türk Katılım Fonu") is True
        assert is_participation_fund("Sukuk Yatırım Fonu") is True
        assert is_participation_fund("Faizsiz Emeklilik Fonu") is True
        assert is_participation_fund("İslami Hisse Fonu") is True
    
    def test_is_participation_fund_negative(self):
        """Test participation fund detection - negative cases."""
        from src.data.collector import is_participation_fund
        
        assert is_participation_fund("Standard Equity Fund") is False
        assert is_participation_fund("Bond Fund A") is False
        assert is_participation_fund("Mixed Portfolio") is False
    
    def test_is_participation_fund_case_insensitive(self):
        """Test that participation detection is case insensitive."""
        from src.data.collector import is_participation_fund
        
        assert is_participation_fund("KATILIM FONU") is True
        assert is_participation_fund("sukuk FUND") is True


# =============================================================================
# PREPROCESSING TESTS
# =============================================================================

class TestPreprocessing:
    """Tests for src/data/preprocessor.py"""
    
    def test_align_fund_data(self, sample_prices):
        """Test fund data alignment."""
        from src.data.preprocessor import align_fund_data
        
        aligned = align_fund_data(sample_prices)
        
        assert isinstance(aligned, pd.DataFrame)
        assert len(aligned.columns) == len(sample_prices)
        assert aligned.index.is_monotonic_increasing
    
    def test_calculate_returns_simple(self, sample_prices):
        """Test simple return calculation."""
        from src.data.preprocessor import align_fund_data, calculate_returns
        
        prices = align_fund_data(sample_prices)
        returns = calculate_returns(prices, method="simple")
        
        assert len(returns) == len(prices) - 1
        assert not returns.isna().all().any()
    
    def test_calculate_returns_log(self, sample_prices):
        """Test log return calculation."""
        from src.data.preprocessor import align_fund_data, calculate_returns
        
        prices = align_fund_data(sample_prices)
        returns = calculate_returns(prices, method="log")
        
        # Log returns should be close to simple returns for small values
        simple_returns = calculate_returns(prices, method="simple")
        
        # Check correlation is very high
        for col in returns.columns:
            corr = returns[col].corr(simple_returns[col])
            assert corr > 0.99
    
    def test_handle_missing_values_ffill(self):
        """Test forward fill for missing values."""
        from src.data.preprocessor import handle_missing_values
        
        # Create data with missing values
        data = pd.DataFrame({
            "A": [1.0, np.nan, 3.0, 4.0],
            "B": [1.0, 2.0, np.nan, 4.0],
        })
        
        filled, stats = handle_missing_values(data, method="ffill")
        
        assert filled["A"].iloc[1] == 1.0  # Forward filled
        assert filled["B"].iloc[2] == 2.0  # Forward filled
    
    def test_prepare_return_matrix(self, sample_prices, sample_tufe):
        """Test full preprocessing pipeline."""
        from src.data.preprocessor import prepare_return_matrix
        
        returns, prices = prepare_return_matrix(
            sample_prices,
            sample_tufe,
            adjust_inflation=True,
        )
        
        assert isinstance(returns, pd.DataFrame)
        assert isinstance(prices, pd.DataFrame)
        assert len(returns) < len(prices)  # Returns drop first row


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Tests for src/features/risk_metrics.py"""
    
    def test_overall_volatility(self, sample_returns):
        """Test overall volatility calculation."""
        from src.features.risk_metrics import overall_volatility
        
        vol = overall_volatility(sample_returns)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_returns.columns)
        assert (vol > 0).all()  # Volatility should be positive
        assert (vol < 2).all()  # Sanity check: annual vol < 200%
    
    def test_rolling_volatility(self, sample_returns):
        """Test rolling volatility calculation."""
        from src.features.risk_metrics import rolling_volatility
        
        windows = [21, 63]
        vols = rolling_volatility(sample_returns, windows=windows)
        
        assert len(vols) == len(windows)
        for window in windows:
            assert window in vols
            assert isinstance(vols[window], pd.DataFrame)
    
    def test_maximum_drawdown(self, sample_prices):
        """Test maximum drawdown calculation."""
        from src.data.preprocessor import align_fund_data
        from src.features.risk_metrics import maximum_drawdown
        
        prices = align_fund_data(sample_prices)
        max_dd, drawdown_ts = maximum_drawdown(prices)
        
        assert isinstance(max_dd, pd.Series)
        assert (max_dd <= 0).all()  # Drawdown should be negative or zero
        assert (max_dd >= -1).all()  # Max drawdown can't exceed 100%
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        from src.features.risk_metrics import sharpe_ratio
        
        sharpe = sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, pd.Series)
        assert len(sharpe) == len(sample_returns.columns)
        # Sharpe ratio typically between -3 and 3 for most funds
        assert (sharpe.abs() < 5).all()
    
    def test_value_at_risk(self, sample_returns):
        """Test VaR calculation."""
        from src.features.risk_metrics import value_at_risk
        
        var_95 = value_at_risk(sample_returns, confidence_level=0.95)
        var_99 = value_at_risk(sample_returns, confidence_level=0.99)
        
        # VaR at 99% should be more extreme than 95%
        assert (var_99 <= var_95).all()
    
    def test_calculate_all_risk_features(self, sample_returns, sample_prices):
        """Test complete feature extraction."""
        from src.data.preprocessor import align_fund_data
        from src.features.risk_metrics import calculate_all_risk_features
        
        prices = align_fund_data(sample_prices)
        features = calculate_all_risk_features(sample_returns, prices)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_returns.columns)
        assert features.shape[1] > 10  # Should have many features
    
    def test_segment_by_risk(self, sample_returns, sample_prices):
        """Test risk segmentation."""
        from src.data.preprocessor import align_fund_data
        from src.features.risk_metrics import (
            calculate_all_risk_features,
            segment_by_risk,
        )
        
        prices = align_fund_data(sample_prices)
        features = calculate_all_risk_features(sample_returns, prices)
        
        # Create composite score
        risk_scores = features["volatility_annual"]
        segments = segment_by_risk(risk_scores)
        
        assert set(segments.unique()).issubset({"Low", "Medium", "High"})


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestModels:
    """Tests for model classes."""
    
    def test_risk_autoencoder_creation(self):
        """Test RiskAutoencoder initialization."""
        from src.models.autoencoder import RiskAutoencoder
        
        ae = RiskAutoencoder(
            input_dim=10,
            encoding_dims=[8, 4],
            latent_dim=2,
        )
        
        assert ae.model is not None
        assert ae.encoder is not None
    
    def test_risk_autoencoder_shapes(self):
        """Test RiskAutoencoder input/output shapes."""
        from src.models.autoencoder import RiskAutoencoder
        
        input_dim = 15
        ae = RiskAutoencoder(input_dim=input_dim, latent_dim=3)
        
        # Test forward pass
        X = np.random.randn(10, input_dim).astype(np.float32)
        reconstructed = ae.predict(X)
        
        assert reconstructed.shape == X.shape
        
        # Test encoding
        encoded = ae.encode(X)
        assert encoded.shape == (10, 3)
    
    def test_embedding_autoencoder_creation(self):
        """Test EmbeddingAutoencoder initialization."""
        from src.models.autoencoder import EmbeddingAutoencoder
        
        ae = EmbeddingAutoencoder(
            input_dim=100,
            encoding_dims=[32, 16],
            latent_dim=2,
        )
        
        assert ae.model is not None
        assert ae.encoder is not None
    
    def test_risk_scorer_creation(self):
        """Test RiskReturnScorer initialization."""
        from src.models.risk_scorer import RiskReturnScorer
        
        scorer = RiskReturnScorer(
            input_dim=10,
            hidden_layers=[16, 8],
        )
        
        assert scorer.model is not None


# =============================================================================
# MODULE TESTS
# =============================================================================

class TestModules:
    """Integration tests for analytical modules."""
    
    def test_risk_profile_extractor_init(self):
        """Test RiskProfileExtractor initialization."""
        from src.modules.risk_profiler import RiskProfileExtractor
        
        extractor = RiskProfileExtractor()
        
        assert extractor.volatility_windows is not None
        assert extractor.ae_config is not None
    
    def test_correlation_mapper_init(self):
        """Test CorrelationMapper initialization."""
        from src.modules.correlation_map import CorrelationMapper
        
        mapper = CorrelationMapper(n_clusters=3)
        
        assert mapper.n_clusters == 3
    
    def test_portfolio_simulator_init(self):
        """Test PortfolioSimulator initialization."""
        from src.modules.portfolio_sim import PortfolioSimulator
        
        sim = PortfolioSimulator()
        
        assert sim.constraints is not None
        assert "min_funds" in sim.constraints


# =============================================================================
# EVALUATION TESTS
# =============================================================================

class TestEvaluation:
    """Tests for evaluation functions."""
    
    def test_evaluate_autoencoder(self):
        """Test autoencoder evaluation metrics."""
        from src.evaluation.metrics import evaluate_autoencoder
        
        X = np.random.randn(100, 10)
        reconstructed = X + np.random.randn(100, 10) * 0.1  # Small noise
        
        metrics = evaluate_autoencoder(X, reconstructed)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "explained_variance_ratio" in metrics
        
        # With small noise, R² should be high
        assert metrics["explained_variance_ratio"] > 0.5
    
    def test_evaluate_clustering(self):
        """Test clustering evaluation."""
        from src.evaluation.metrics import evaluate_clustering
        
        # Create well-separated clusters
        cluster1 = np.random.randn(50, 2) + np.array([0, 0])
        cluster2 = np.random.randn(50, 2) + np.array([10, 10])
        embeddings = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 50 + [1] * 50)
        
        metrics = evaluate_clustering(embeddings, labels)
        
        assert "silhouette_score" in metrics
        assert metrics["silhouette_score"] > 0.5  # Well-separated clusters


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration."""
    
    def test_config_imports(self):
        """Test that config can be imported."""
        from config import (
            START_DATE,
            END_DATE,
            FUND_SEARCH_KEYWORDS,
            RISK_AE_CONFIG,
            PORTFOLIO_CONSTRAINTS,
        )
        
        assert START_DATE < END_DATE
        assert len(FUND_SEARCH_KEYWORDS) > 0
        assert "latent_dim" in RISK_AE_CONFIG
        assert "min_funds" in PORTFOLIO_CONSTRAINTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
