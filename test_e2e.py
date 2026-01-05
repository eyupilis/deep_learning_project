#!/usr/bin/env python3
"""
End-to-end test with synthetic data for all 3 modules.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("TEFAS FUND ANALYSIS - END-TO-END TEST")
print("=" * 60)

# Generate synthetic data
print("\n[1/6] Generating synthetic data...")
np.random.seed(42)
n_funds = 15
n_days = 252

dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
fund_codes = [f"FUND{i:02d}" for i in range(n_funds)]

# Generate prices with different characteristics
prices_dict = {}
for i, code in enumerate(fund_codes):
    drift = 0.0003 + i * 0.00005
    volatility = 0.01 + i * 0.002
    returns = np.random.normal(drift, volatility, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    prices_dict[code] = prices

prices = pd.DataFrame(prices_dict, index=dates)
returns = prices.pct_change().dropna()

print(f"   ✓ Generated {n_funds} funds x {n_days} days")

# Test Module 1: Risk Profile Extractor
print("\n[2/6] Testing Module 1: Risk Profile Extractor...")
try:
    from src.features.risk_metrics import calculate_all_risk_features, segment_by_risk
    from src.models.autoencoder import RiskAutoencoder
    
    # Calculate features
    features = calculate_all_risk_features(returns, prices)
    print(f"   ✓ Calculated {len(features.columns)} features")
    
    # Train autoencoder
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features.dropna())
    
    ae = RiskAutoencoder(input_dim=X.shape[1], encoding_dims=[8, 4], latent_dim=2)
    ae.fit(X, epochs=10, batch_size=4, verbose=0)
    
    # Detect anomalies
    errors = ae.reconstruction_error(X)
    anomalies, threshold = ae.detect_anomalies(X, threshold_percentile=90)
    print(f"   ✓ Autoencoder trained, {anomalies.sum()} anomalies detected")
    
    # Segment
    risk_scores = features["volatility_annual"]
    segments = segment_by_risk(risk_scores)
    print(f"   ✓ Risk segmentation: {dict(segments.value_counts())}")
    
    MODULE1_OK = True
except Exception as e:
    print(f"   ✗ Module 1 FAILED: {e}")
    MODULE1_OK = False

# Test Module 2: Correlation Map
print("\n[3/6] Testing Module 2: Correlation Mapper...")
try:
    from src.models.autoencoder import EmbeddingAutoencoder
    
    # Correlation matrix
    corr_matrix = returns.corr()
    print(f"   ✓ Correlation matrix: {corr_matrix.shape}")
    
    # Embedding autoencoder
    X_returns = returns.T.values  # funds x time
    X_scaled = (X_returns - X_returns.mean(axis=1, keepdims=True)) / (X_returns.std(axis=1, keepdims=True) + 1e-8)
    
    emb_ae = EmbeddingAutoencoder(input_dim=X_scaled.shape[1], encoding_dims=[16, 8], latent_dim=2)
    emb_ae.fit(X_scaled, epochs=10, batch_size=4, verbose=0)
    
    embeddings = emb_ae.get_embeddings(X_scaled)
    print(f"   ✓ Extracted 2D embeddings: {embeddings.shape}")
    
    # Cluster
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    print(f"   ✓ Clustered into {len(np.unique(clusters))} groups")
    
    MODULE2_OK = True
except Exception as e:
    print(f"   ✗ Module 2 FAILED: {e}")
    MODULE2_OK = False

# Test Module 3: Portfolio Simulator
print("\n[4/6] Testing Module 3: Portfolio Simulator...")
try:
    from src.models.risk_scorer import RiskReturnScorer
    
    # Create scorer
    scorer = RiskReturnScorer(input_dim=features.shape[1], hidden_layers=[8, 4])
    
    # Train (with simplified target) - use scorer's internal scaler
    features_clean = features.dropna()
    X_feat = features_clean.values
    y = features_clean["sharpe_ratio"].values
    returns_annual = features_clean["return_mean_annual"].values if "return_mean_annual" in features_clean.columns else y
    risk_scores = features_clean["volatility_annual"].values
    
    # Fit scorer with proper method
    scorer.fit(X_feat, returns_annual, risk_scores, epochs=10, batch_size=4, verbose=0)
    scores = scorer.predict(X_feat)
    print(f"   ✓ ANN scorer trained, scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Backtest simulation
    selected_funds = returns.columns[:5]  # Top 5
    weights = pd.Series(0.2, index=selected_funds)
    
    portfolio_returns = returns[selected_funds].mul(weights).sum(axis=1)
    cumulative = (1 + portfolio_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    
    print(f"   ✓ Backtest: Total Return = {total_return:.2%}, Max DD = {max_dd:.2%}")
    
    MODULE3_OK = True
except Exception as e:
    print(f"   ✗ Module 3 FAILED: {e}")
    MODULE3_OK = False

# Test Visualization (without display)
print("\n[5/6] Testing Visualizations...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters)
    plt.close(fig)
    print("   ✓ Visualization generation OK")
    VIZ_OK = True
except Exception as e:
    print(f"   ✗ Visualization FAILED: {e}")
    VIZ_OK = False

# Summary
print("\n[6/6] Test Summary")
print("=" * 60)
all_ok = MODULE1_OK and MODULE2_OK and MODULE3_OK and VIZ_OK
print(f"Module 1 (Risk Profile Extractor):     {'✓ PASS' if MODULE1_OK else '✗ FAIL'}")
print(f"Module 2 (Correlation Mapper):         {'✓ PASS' if MODULE2_OK else '✗ FAIL'}")
print(f"Module 3 (Portfolio Simulator):        {'✓ PASS' if MODULE3_OK else '✗ FAIL'}")
print(f"Visualization:                         {'✓ PASS' if VIZ_OK else '✗ FAIL'}")
print("=" * 60)
print(f"OVERALL: {'✓ ALL MODULES WORKING' if all_ok else '✗ SOME MODULES FAILED'}")
print("=" * 60)
