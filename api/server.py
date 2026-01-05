"""
TEFAS Fund Analysis - FastAPI Backend
=======================================
Provides REST API endpoints for the frontend dashboard.

Endpoints:
- GET /api/funds           - Get all funds with risk profiles
- GET /api/analysis/run    - Run full analysis pipeline
- GET /api/analysis/status - Get analysis status
- GET /api/portfolio       - Get portfolio simulation results
- GET /api/correlations    - Get correlation matrix
- GET /api/embeddings      - Get 2D embeddings for visualization
"""

import os
import sys
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MODELS
# =============================================================================

class Fund(BaseModel):
    id: str
    code: str
    name: str
    riskScore: float
    riskSegment: str
    volatility: float
    maxDrawdown: float
    sharpeRatio: float
    hasAnomaly: bool
    cluster: int
    embeddingX: float
    embeddingY: float


class PortfolioMetrics(BaseModel):
    totalReturn: float
    annualizedReturn: float
    sharpeRatio: float
    maxDrawdown: float
    winRate: float


class CorrelationEntry(BaseModel):
    fundA: str
    fundB: str
    correlation: float


class PortfolioHistoryEntry(BaseModel):
    date: str
    value: float


class AnalysisResponse(BaseModel):
    funds: List[Fund]
    portfolioMetrics: PortfolioMetrics
    portfolioHistory: List[PortfolioHistoryEntry]
    correlations: List[CorrelationEntry]
    riskDistribution: List[dict]
    dateRange: str
    disclaimer: str


# =============================================================================
# STATE
# =============================================================================

class AnalysisState:
    def __init__(self):
        self.is_running = False
        self.last_run = None
        self.error = None
        self.results = None

state = AnalysisState()


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load existing results if available
    try:
        load_existing_results()
        logger.info("Loaded existing analysis results")
    except Exception as e:
        logger.warning(f"No existing results: {e}")
    
    yield
    
    # Shutdown
    logger.info("API shutting down")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="TEFAS Fund Analysis API",
    description="Backend API for TEFAS Fund Risk Analysis Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080", "http://localhost:8085", "http://localhost:8081", "http://localhost:8082", "http://localhost:8083", "http://localhost:8084"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPERS
# =============================================================================

def load_existing_results():
    """Load results from disk if available."""
    import pandas as pd
    
    risk_profiles_path = DATA_PROCESSED / "risk_profiles.csv"
    embeddings_path = DATA_PROCESSED / "embeddings.csv"
    features_path = DATA_PROCESSED / "risk_features.csv"
    
    if not risk_profiles_path.exists():
        raise FileNotFoundError("No analysis results found")
    
    # Load risk profiles
    risk_profiles = pd.read_csv(risk_profiles_path, index_col=0)
    
    # Load embeddings
    embeddings = None
    if embeddings_path.exists():
        embeddings = pd.read_csv(embeddings_path, index_col=0)
    
    # Load features
    features = None
    if features_path.exists():
        features = pd.read_csv(features_path, index_col=0)
    
    # Convert to API format
    funds = []
    for idx, (code, row) in enumerate(risk_profiles.iterrows()):
        emb_x = embeddings.loc[code, "embedding_x"] if embeddings is not None and code in embeddings.index else 0
        emb_y = embeddings.loc[code, "embedding_y"] if embeddings is not None and code in embeddings.index else 0
        cluster = int(embeddings.loc[code, "cluster"]) if embeddings is not None and code in embeddings.index else 0
        
        fund = Fund(
            id=str(idx + 1),
            code=str(code),
            name=f"Fund {code}",  # TODO: Get actual names
            riskScore=float(row.get("composite_risk_score", 0)),
            riskSegment=str(row.get("risk_segment", "Medium")),
            volatility=float(row.get("volatility_annual", 0) * 100),
            maxDrawdown=float(row.get("max_drawdown", 0) * -100),
            sharpeRatio=float(features.loc[code, "sharpe_ratio"]) if features is not None and code in features.index else 0,
            hasAnomaly=bool(row.get("is_anomaly", False)),
            cluster=cluster,
            embeddingX=float(emb_x),
            embeddingY=float(emb_y),
        )
        funds.append(fund)
    
    # Load backtest results if available
    backtest_path = DATA_PROCESSED / "backtest_low.csv"
    portfolio_history = []
    portfolio_metrics = PortfolioMetrics(
        totalReturn=0,
        annualizedReturn=0,
        sharpeRatio=0,
        maxDrawdown=0,
        winRate=0,
    )
    
    if backtest_path.exists():
        backtest = pd.read_csv(backtest_path, index_col=0, parse_dates=True)
        for date_idx, row in backtest.iterrows():
            portfolio_history.append(PortfolioHistoryEntry(
                date=str(date_idx.date()) if hasattr(date_idx, 'date') else str(date_idx),
                value=float(row.get("portfolio_value", 100000)),
            ))
        
        # Calculate metrics
        if len(backtest) > 1:
            returns = backtest["daily_return"].dropna()
            final_value = backtest["portfolio_value"].iloc[-1]
            initial_value = 100000
            
            portfolio_metrics = PortfolioMetrics(
                totalReturn=round((final_value / initial_value - 1) * 100, 2),
                annualizedReturn=round(returns.mean() * 252 * 100, 2),
                sharpeRatio=round(returns.mean() / returns.std() * (252 ** 0.5), 2) if returns.std() > 0 else 0,
                maxDrawdown=round(backtest["drawdown"].min() * 100, 2),
                winRate=round((returns > 0).mean() * 100, 2),
            )
    
    # Build correlations
    correlations = []
    corr_path = DATA_PROCESSED / "correlation_matrix.csv"
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path, index_col=0)
        codes = corr_df.columns[:8]  # Limit to first 8 for visualization
        for i, code_a in enumerate(codes):
            for j, code_b in enumerate(codes):
                correlations.append(CorrelationEntry(
                    fundA=code_a,
                    fundB=code_b,
                    correlation=round(float(corr_df.loc[code_a, code_b]), 2),
                ))
    
    # Risk distribution
    risk_distribution = [
        {"segment": "Low", "count": sum(1 for f in funds if f.riskSegment == "Low"), "color": "hsl(160, 84%, 39%)"},
        {"segment": "Medium", "count": sum(1 for f in funds if f.riskSegment == "Medium"), "color": "hsl(38, 92%, 50%)"},
        {"segment": "High", "count": sum(1 for f in funds if f.riskSegment == "High"), "color": "hsl(0, 84%, 60%)"},
    ]
    
    state.results = AnalysisResponse(
        funds=funds,
        portfolioMetrics=portfolio_metrics,
        portfolioHistory=portfolio_history,
        correlations=correlations,
        riskDistribution=risk_distribution,
        dateRange=f"{START_DATE} to {END_DATE}",
        disclaimer=DISCLAIMER,
    )
    
    state.last_run = datetime.now()


def run_analysis_task(segment: str = "Low"):
    """Run the full analysis pipeline."""
    state.is_running = True
    state.error = None
    
    try:
        # Import here to avoid circular imports
        from main import run_full_pipeline
        
        logger.info("Starting analysis pipeline...")
        run_full_pipeline(segment=segment)
        
        # Reload results
        load_existing_results()
        
        state.is_running = False
        state.last_run = datetime.now()
        logger.info("Analysis complete")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        state.error = str(e)
        state.is_running = False


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "TEFAS Fund Analysis API",
        "docs": "/docs",
        "disclaimer": "This is for educational purposes only. NOT investment advice.",
    }


@app.get("/api/status")
async def get_status():
    return {
        "isRunning": state.is_running,
        "lastRun": state.last_run.isoformat() if state.last_run else None,
        "error": state.error,
        "hasResults": state.results is not None,
    }


@app.post("/api/analysis/run")
async def run_analysis(background_tasks: BackgroundTasks, segment: str = "Low"):
    """Start analysis pipeline in background."""
    if state.is_running:
        raise HTTPException(status_code=409, detail="Analysis already running")
    
    background_tasks.add_task(run_analysis_task, segment)
    
    return {"message": "Analysis started", "segment": segment}


@app.get("/api/funds", response_model=List[Fund])
async def get_funds():
    """Get all funds with risk profiles."""
    if state.results is None:
        raise HTTPException(status_code=404, detail="No analysis results. Run analysis first.")
    return state.results.funds


@app.get("/api/analysis/results", response_model=AnalysisResponse)
async def get_analysis_results():
    """Get complete analysis results."""
    if state.results is None:
        raise HTTPException(status_code=404, detail="No analysis results. Run analysis first.")
    return state.results


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio simulation results."""
    if state.results is None:
        raise HTTPException(status_code=404, detail="No analysis results")
    
    return {
        "metrics": state.results.portfolioMetrics,
        "history": state.results.portfolioHistory,
    }


@app.get("/api/correlations", response_model=List[CorrelationEntry])
async def get_correlations():
    """Get correlation matrix."""
    if state.results is None:
        raise HTTPException(status_code=404, detail="No analysis results")
    return state.results.correlations


@app.get("/api/embeddings")
async def get_embeddings():
    """Get 2D embeddings for scatter plot."""
    if state.results is None:
        raise HTTPException(status_code=404, detail="No analysis results")
    
    return [
        {
            "code": f.code,
            "x": f.embeddingX,
            "y": f.embeddingY,
            "cluster": f.cluster,
            "riskSegment": f.riskSegment,
        }
        for f in state.results.funds
    ]


@app.get("/api/figures/{filename}")
async def get_figure(filename: str):
    """Serve generated figures."""
    filepath = OUTPUT_FIGURES / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(filepath)


# =============================================================================
# AI ANALYZER ENDPOINTS
# =============================================================================

@app.get("/api/data-range")
async def get_data_range():
    """Get available data date range."""
    import pandas as pd
    
    # Try to get actual date range from data
    try:
        returns_path = DATA_PROCESSED / "returns.csv"
        if returns_path.exists():
            returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            return {
                "minDate": str(returns.index.min().date()),
                "maxDate": str(returns.index.max().date()),
                "totalDays": len(returns),
            }
    except:
        pass
    
    # Fallback to config dates
    return {
        "minDate": str(START_DATE),
        "maxDate": str(END_DATE),
        "totalDays": None,
    }


@app.get("/api/module-info/{module}")
async def get_module_info(module: str):
    """Get informative description for a module."""
    from api.ai_analyzer import MODULE_DESCRIPTIONS
    
    if module not in MODULE_DESCRIPTIONS:
        raise HTTPException(status_code=404, detail=f"Module not found: {module}")
    
    return MODULE_DESCRIPTIONS[module]


@app.get("/api/technical-details")
async def get_technical_details():
    """Get technical documentation content."""
    from api.ai_analyzer import TECHNICAL_DETAILS
    return TECHNICAL_DETAILS


@app.post("/api/ai-analyze/{module}")
async def ai_analyze_module(module: str):
    """Get AI-powered analysis for a module using Gemini."""
    from api.ai_analyzer import (
        analyze_risk_profile,
        analyze_correlation,
        analyze_portfolio,
    )
    
    if state.results is None:
        raise HTTPException(status_code=404, detail="No analysis results available")
    
    try:
        if module == "risk_profile":
            # Prepare data for AI
            funds_data = [f.model_dump() for f in state.results.funds]
            risk_dist = {r["segment"]: r["count"] for r in state.results.riskDistribution}
            analysis = analyze_risk_profile(funds_data, risk_dist)
            
        elif module == "correlation":
            # Load FULL correlation matrix for AI analysis (don't use the limited subset in state)
            import pandas as pd
            from config import DATA_PROCESSED
            
            corr_path = DATA_PROCESSED / "correlation_matrix.csv"
            full_correlations = []
            
            if corr_path.exists():
                logger.info("Loading full correlation matrix for AI analysis")
                corr_df = pd.read_csv(corr_path, index_col=0)
                
                # Convert to list format but keep ALL significant pairs
                # Using upper triangle to avoid duplicates
                import numpy as np
                mask = np.triu(np.ones(corr_df.shape), k=1).astype(bool)
                
                for i, col in enumerate(corr_df.columns):
                    for j, idx in enumerate(corr_df.index):
                        if i > j: # Upper triangle
                            val = corr_df.iloc[j, i]
                            full_correlations.append({
                                "fundA": idx,
                                "fundB": col,
                                "correlation": round(float(val), 2)
                            })
            else:
                # Fallback to state if file reading fails
                logger.warning("Could not load full correlation matrix, using state data")
                full_correlations = [c.model_dump() for c in state.results.correlations]

            embeddings = [
                {"code": f.code, "cluster": f.cluster, "x": f.embeddingX, "y": f.embeddingY}
                for f in state.results.funds
            ]
            
            # Pass full correlations to analyzer (it will filter top/bottom internally or we can do it here)
            analysis = analyze_correlation(full_correlations, embeddings)
            
        elif module == "portfolio":
            metrics = state.results.portfolioMetrics.model_dump()
            history = [h.model_dump() for h in state.results.portfolioHistory]
            selected_funds = [f.code for f in state.results.funds if f.riskSegment == "Low"][:6]
            analysis = analyze_portfolio(metrics, history, selected_funds)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown module: {module}")
        
        return {
            "module": module,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
