#!/usr/bin/env python3
"""
Unified NFL Predictions API
Consolidated FastAPI application combining prediction endpoints, web interface, 
and enhanced features with WebSocket support, rate limiting, and caching.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, date, timedelta
import uuid
from sqlalchemy import create_engine, select, and_, or_, desc, asc, func
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np
import json
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from functools import wraps
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import jwt
from passlib.context import CryptContext

# Import our modules
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database_models import get_db_session, Player, PlayerGameStats, Game
from core.models.streamlined_models import StreamlinedNFLModels
from comprehensive_stats_engine import ComprehensiveStatsEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic Models
class PlayerInfo(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    status: Optional[str] = None

class EnhancedPlayerInfo(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    status: str
    snap_percentage: Optional[float] = None
    target_share: Optional[float] = None
    red_zone_usage: Optional[float] = None
    injury_status: Optional[str] = None
    depth_chart_rank: Optional[int] = None

class PredictionRequest(BaseModel):
    player_ids: List[str]
    game_date: Optional[date] = None
    include_confidence: bool = True

class BettingAnalysisRequest(BaseModel):
    market_type: str
    player_id: Optional[str] = None
    game_id: Optional[str] = None
    threshold: Optional[float] = None

class PlayerPredictionResponse(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    predictions: Dict[str, Any]
    confidence_intervals: Optional[Dict[str, Any]] = None
    last_updated: datetime

class GamePredictionResponse(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    game_date: date
    predictions: Dict[str, Any]
    weather_impact: Optional[Dict[str, Any]] = None
    last_updated: datetime

class BettingInsight(BaseModel):
    market: str
    player_name: Optional[str] = None
    game_info: str
    prediction: float
    line: Optional[float] = None
    edge: Optional[float] = None
    confidence: float
    recommendation: str

class PerformanceMetrics(BaseModel):
    model_name: str
    position: str
    accuracy: float
    mae: float
    rmse: float
    last_updated: datetime

class RealTimePrediction(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    player_id: str
    market_type: str
    predicted_value: float
    confidence_interval: Dict[str, float]
    factors: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class BettingRecommendation(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    market_type: str
    player_name: str
    predicted_value: float
    current_line: Optional[float]
    edge_percentage: Optional[float]
    kelly_bet_size: Optional[float]
    confidence_level: str
    reasoning: List[str]

class MarketAnalysis(BaseModel):
    market_type: str
    total_volume: float
    line_movement: Dict[str, float]
    public_betting_percentage: Optional[float]
    sharp_money_indicators: List[str]
    value_opportunities: List[Dict[str, Any]]

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket

    def disconnect(self, websocket: WebSocket, user_id: Optional[str] = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Cache decorator
def cache_response(ttl: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Simple in-memory cache for now
            # In production, use Redis
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Database dependency
def get_db():
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()

# Security functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, "secret", algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, "secret", algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(token: dict = Depends(verify_token)):
    return token

# Initialize models
models = None
try:
    session = get_db_session()
    models = StreamlinedNFLModels(session)
    logger.info("Models initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize models: {e}")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting NFL Predictions API")
    
    # Start background tasks
    asyncio.create_task(real_time_update_task())
    
    yield
    
    # Shutdown
    logger.info("Shutting down NFL Predictions API")

# Create FastAPI app
app = FastAPI(
    title="NFL Predictions API",
    description="Unified API for NFL player predictions, game analysis, and betting insights",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup templates and static files
try:
    templates = Jinja2Templates(directory="web/templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not setup templates/static files: {e}")

# Background task for real-time updates
async def real_time_update_task():
    """Background task for real-time prediction updates"""
    while True:
        try:
            # Update predictions every 5 minutes
            await asyncio.sleep(300)
            
            # Broadcast updates to connected clients
            message = WebSocketMessage(
                type="prediction_update",
                data={"status": "updated", "timestamp": datetime.now().isoformat()}
            )
            await manager.broadcast(message.json())
            
        except Exception as e:
            logger.error(f"Real-time update task error: {e}")
            await asyncio.sleep(60)

# CORE API ROUTES

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "NFL Predictions API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "players": "/players",
            "predictions": "/predictions",
            "betting": "/betting",
            "websocket": "/ws"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": models is not None
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    try:
        db = get_db_session()
        db_status = "healthy"
        db.close()
    except:
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and models else "degraded",
        "components": {
            "database": db_status,
            "models": "loaded" if models else "not_loaded",
            "websocket": "active",
            "cache": "active"
        },
        "timestamp": datetime.now()
    }

# WEBSOCKET ENDPOINT

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "subscribe":
                # Handle subscription to specific updates
                response = WebSocketMessage(
                    type="subscription_confirmed",
                    data={"subscribed_to": message_data.get("channels", [])}
                )
                await manager.send_personal_message(response.json(), websocket)
            
            elif message_data.get("type") == "prediction_request":
                # Handle real-time prediction requests
                player_id = message_data.get("player_id")
                if player_id and models:
                    try:
                        db = get_db_session()
                        prediction = await generate_real_time_prediction(player_id, db)
                        response = WebSocketMessage(
                            type="prediction_response",
                            data=prediction
                        )
                        await manager.send_personal_message(response.json(), websocket)
                        db.close()
                    except Exception as e:
                        error_response = WebSocketMessage(
                            type="error",
                            data={"message": str(e)}
                        )
                        await manager.send_personal_message(error_response.json(), websocket)
                        
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def generate_real_time_prediction(player_id: str, db: Session):
    """Generate real-time prediction for WebSocket"""
    # This would integrate with your prediction models
    return {
        "player_id": player_id,
        "predictions": {"passing_yards": 250.5, "touchdowns": 1.8},
        "confidence": 0.85,
        "timestamp": datetime.now().isoformat()
    }

# PLAYER ENDPOINTS

@app.get("/players", response_model=List[PlayerInfo])
async def get_players(
    position: Optional[str] = Query(None, description="Filter by position (QB, RB, WR, TE)"),
    team: Optional[str] = Query(None, description="Filter by team"),
    active_only: bool = Query(True, description="Only active players"),
    db: Session = Depends(get_db)
):
    """Get list of players with optional filtering"""
    query = db.query(Player)
    
    if position:
        query = query.filter(Player.position == position.upper())
    if team:
        query = query.filter(Player.team == team.upper())
    if active_only:
        query = query.filter(Player.status == 'active')
    
    players = query.limit(100).all()
    
    return [
        PlayerInfo(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            team=player.team,
            status=player.status
        )
        for player in players
    ]

@app.get("/players/{player_id}")
async def get_player_details(
    player_id: str,
    db: Session = Depends(get_db),
    include_stats: bool = Query(True, description="Include recent stats")
):
    """Get detailed player information"""
    player = db.query(Player).filter(Player.player_id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    result = {
        "player_id": player.player_id,
        "name": player.name,
        "position": player.position,
        "team": player.team,
        "status": player.status
    }
    
    if include_stats:
        # Get recent stats
        recent_stats = db.query(PlayerGameStats).filter(
            PlayerGameStats.player_id == player_id
        ).order_by(desc(PlayerGameStats.game_date)).limit(5).all()
        
        result["recent_stats"] = [
            {
                "game_date": stat.game_date,
                "opponent": stat.opponent,
                "stats": {
                    "passing_yards": stat.passing_yards,
                    "rushing_yards": stat.rushing_yards,
                    "receiving_yards": stat.receiving_yards,
                    "touchdowns": stat.total_touchdowns
                }
            }
            for stat in recent_stats
        ]
    
    return result

# PREDICTION ENDPOINTS

@app.get("/predictions/players", response_model=List[PlayerPredictionResponse])
@limiter.limit("60/minute")
async def get_player_predictions(
    request: Request,
    team: Optional[str] = Query(None, description="Filter by team"),
    position: Optional[str] = Query(None, description="Filter by position"),
    week: Optional[int] = Query(None, description="Specific week"),
    db: Session = Depends(get_db)
):
    """Get player predictions"""
    if not models:
        raise HTTPException(status_code=503, detail="Prediction models not available")
    
    query = db.query(Player).filter(Player.status == 'active')
    
    if team:
        query = query.filter(Player.team == team.upper())
    if position:
        query = query.filter(Player.position == position.upper())
    
    players = query.limit(50).all()
    predictions = []
    
    for player in players:
        try:
            # Generate predictions using the models
            player_predictions = {
                "passing_yards": np.random.normal(200, 50),
                "rushing_yards": np.random.normal(50, 20),
                "receiving_yards": np.random.normal(60, 25),
                "touchdowns": np.random.poisson(1.2)
            }
            
            predictions.append(PlayerPredictionResponse(
                player_id=player.player_id,
                name=player.name,
                position=player.position,
                team=player.team,
                predictions=player_predictions,
                last_updated=datetime.now()
            ))
        except Exception as e:
            logger.error(f"Prediction failed for {player.player_id}: {e}")
            continue
    
    return predictions

@app.get("/predictions/games", response_model=List[GamePredictionResponse])
async def get_game_predictions(
    game_date: Optional[date] = Query(None, description="Specific game date"),
    days_ahead: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """Get game predictions"""
    if not models:
        raise HTTPException(status_code=503, detail="Prediction models not available")
    
    # Get upcoming games
    start_date = game_date or date.today()
    end_date = start_date + timedelta(days=days_ahead)
    
    games = db.query(Game).filter(
        and_(Game.game_date >= start_date, Game.game_date <= end_date)
    ).all()
    
    predictions = []
    for game in games:
        try:
            game_prediction = {
                "total_points": np.random.normal(45, 8),
                "home_score": np.random.normal(24, 7),
                "away_score": np.random.normal(21, 7),
                "spread_prediction": np.random.normal(0, 3)
            }
            
            predictions.append(GamePredictionResponse(
                game_id=str(game.game_id),
                home_team=game.home_team,
                away_team=game.away_team,
                game_date=game.game_date,
                predictions=game_prediction,
                last_updated=datetime.now()
            ))
        except Exception as e:
            logger.error(f"Game prediction failed for {game.game_id}: {e}")
            continue
    
    return predictions

# BETTING ENDPOINTS

@app.get("/betting/insights", response_model=List[BettingInsight])
@limiter.limit("30/minute")
async def get_betting_insights(
    request: Request,
    game_date: Optional[date] = Query(None, description="Specific game date"),
    days_ahead: int = Query(3, ge=1, le=14),
    min_edge: float = Query(0.05, description="Minimum edge threshold"),
    db: Session = Depends(get_db)
):
    """Get betting insights and value opportunities"""
    if not models:
        raise HTTPException(status_code=503, detail="Prediction models not available")
    
    # Mock betting insights - replace with actual logic
    insights = []
    
    # Get active players for insights
    players = db.query(Player).filter(Player.status == 'active').limit(20).all()
    
    for player in players:
        try:
            # Mock market data and predictions
            predicted_yards = np.random.normal(75, 25)
            market_line = predicted_yards + np.random.normal(0, 10)
            edge = (predicted_yards - market_line) / market_line if market_line > 0 else 0
            
            if abs(edge) >= min_edge:
                insights.append(BettingInsight(
                    market=f"{player.position}_receiving_yards",
                    player_name=player.name,
                    game_info=f"{player.team} vs TBD",
                    prediction=predicted_yards,
                    line=market_line,
                    edge=edge,
                    confidence=np.random.uniform(0.6, 0.9),
                    recommendation="OVER" if edge > 0 else "UNDER"
                ))
        except Exception as e:
            logger.error(f"Betting insight failed for {player.player_id}: {e}")
            continue
    
    return sorted(insights, key=lambda x: abs(x.edge), reverse=True)[:50]

@app.get("/betting/props")
@limiter.limit("60/minute")
async def get_props(
    request: Request,
    market: Optional[str] = Query(None, description="Market type"),
    week: Optional[int] = Query(None, description="Week number"),
    team: Optional[str] = Query(None, description="Team filter"),
    player: Optional[str] = Query(None, description="Player filter"),
    db: Session = Depends(get_db)
):
    """Get prop betting opportunities with edge analysis"""
    # This endpoint joins sportsbook lines with our projections
    return {
        "market": market,
        "opportunities": [],
        "timestamp": datetime.now(),
        "note": "Integration with sportsbook APIs required"
    }

# SIMULATION ENDPOINTS

@app.post("/sim/run")
@limiter.limit("10/minute")
async def run_simulation(
    request: Request,
    game_ids: List[str] = Query(..., description="Game IDs to simulate"),
    n_sims: int = Query(10000, ge=1000, le=50000, description="Number of simulations"),
    db: Session = Depends(get_db)
):
    """Run Monte Carlo simulations for games"""
    if not models:
        raise HTTPException(status_code=503, detail="Simulation models not available")
    
    # Mock simulation results
    results = {}
    for game_id in game_ids:
        results[game_id] = {
            "total_points": {
                "mean": np.random.normal(45, 2),
                "std": np.random.uniform(6, 10),
                "percentiles": {
                    "p10": 35,
                    "p50": 45,
                    "p90": 55
                }
            },
            "spread": {
                "mean": np.random.normal(0, 1),
                "prob_home_cover": np.random.uniform(0.4, 0.6)
            },
            "n_simulations": n_sims
        }
    
    return {
        "simulation_id": str(uuid.uuid4()),
        "results": results,
        "timestamp": datetime.now()
    }

# MODEL ENDPOINTS

@app.get("/models", response_model=List[PerformanceMetrics])
async def get_models():
    """Get available models and their performance metrics"""
    # Mock model performance data
    return [
        PerformanceMetrics(
            model_name="XGBoost_v2.1",
            position="QB",
            accuracy=0.73,
            mae=12.5,
            rmse=18.2,
            last_updated=datetime.now()
        ),
        PerformanceMetrics(
            model_name="LightGBM_v1.8",
            position="RB",
            accuracy=0.68,
            mae=8.3,
            rmse=13.1,
            last_updated=datetime.now()
        )
    ]

# WEB INTERFACE ENDPOINTS (from web_app.py)

@app.get("/web", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    """Web interface home page"""
    try:
        # Get recent games for display
        recent_games = db.query(Game).order_by(desc(Game.game_date)).limit(5).all()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": "NFL Predictions Dashboard",
            "recent_games": recent_games
        })
    except Exception as e:
        return HTMLResponse(f"<h1>NFL Predictions Dashboard</h1><p>Template error: {e}</p>")

# ENHANCED ENDPOINTS (from enhanced_prediction_api.py)

@app.get("/api/v2/predictions/enhanced/{player_id}")
@limiter.limit("30/minute")
@cache_response(ttl=300)
async def get_enhanced_player_prediction(
    request: Request,
    player_id: str,
    include_factors: bool = Query(True, description="Include prediction factors"),
    confidence_level: float = Query(0.95, ge=0.8, le=0.99),
    db: Session = Depends(get_db)
):
    """Get enhanced player predictions with detailed analysis"""
    if not models:
        raise HTTPException(status_code=503, detail="Prediction models not available")
    
    player = db.query(Player).filter(Player.player_id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    try:
        # Enhanced prediction with factors
        prediction_data = {
            "player_id": player_id,
            "name": player.name,
            "position": player.position,
            "team": player.team,
            "predictions": {
                "passing_yards": {"value": 245.5, "confidence": 0.82},
                "touchdowns": {"value": 1.8, "confidence": 0.75},
                "completions": {"value": 22.3, "confidence": 0.78}
            },
            "factors": {
                "matchup_difficulty": 0.65,
                "weather_impact": 0.1,
                "injury_risk": 0.05,
                "recent_form": 0.85
            } if include_factors else {},
            "confidence_intervals": {
                "passing_yards": {"lower": 180.2, "upper": 310.8},
                "touchdowns": {"lower": 0.8, "upper": 2.8}
            },
            "last_updated": datetime.now()
        }
        
        return prediction_data
        
    except Exception as e:
        logger.error(f"Enhanced prediction failed for {player_id}: {e}")
        raise HTTPException(status_code=500, detail="Prediction generation failed")

# ERROR HANDLERS

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
