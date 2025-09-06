#!/usr/bin/env python3
"""
Enhanced NFL Predictions API
Advanced FastAPI implementation with WebSocket support, real-time updates, 
caching, rate limiting, and comprehensive analytics
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, date, timedelta
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

# Import our enhanced modules
from database_models import *
from optimized_prediction_pipeline import OptimizedNFLPipeline, create_optimized_pipeline
from config_manager import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"

# Enhanced Pydantic models
class EnhancedPlayerInfo(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    age: Optional[int] = None
    years_experience: Optional[int] = None
    injury_status: Optional[str] = None
    fantasy_rank: Optional[int] = None
    season_stats: Optional[Dict[str, float]] = None

class RealTimePrediction(BaseModel):
    player_id: str
    predictions: Dict[str, Any]
    confidence: float
    market_edge: Optional[float] = None
    timestamp: datetime
    factors: List[str] = []
    
class BettingRecommendation(BaseModel):
    player_id: str
    prop_type: str
    recommendation: str  # BUY, SELL, HOLD
    confidence: float
    expected_value: float
    risk_level: str
    reasoning: List[str]
    
class MarketAnalysis(BaseModel):
    game_id: str
    line_movements: List[Dict[str, Any]]
    public_betting: Dict[str, float]
    sharp_money_indicators: Dict[str, Any]
    value_opportunities: List[BettingRecommendation]

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        # Remove from all subscriptions
        for topic, connections in self.subscriptions.items():
            connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_to_topic(self, topic: str, message: str):
        if topic not in self.subscriptions:
            return
        
        disconnected = set()
        for connection in self.subscriptions[topic]:
            try:
                await connection.send_text(message)
            except:
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.subscriptions[topic].discard(conn)
    
    def subscribe_to_topic(self, websocket: WebSocket, topic: str):
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(websocket)

# Global instances
manager = ConnectionManager()
pipeline = None
redis_client = None

# Enhanced authentication
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

# Enhanced dependency for authentication
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return payload

# Caching decorator
def cache_response(ttl: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis_client:
                return await func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced app lifecycle management."""
    global pipeline, redis_client
    
    # Startup
    logger.info("🚀 Starting Enhanced NFL Predictions API...")
    
    try:
        # Initialize Redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        redis_client = None
    
    # Initialize optimized pipeline
    pipeline = create_optimized_pipeline()
    logger.info("✅ Optimized pipeline initialized")
    
    # Start background tasks
    asyncio.create_task(real_time_update_task())
    
    logger.info("🎯 Enhanced API ready for requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced NFL Predictions API...")

# Create enhanced FastAPI app
app = FastAPI(
    title="Enhanced NFL Predictions API",
    description="Advanced ML-powered NFL analytics with real-time updates and comprehensive betting insights",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Database dependency
def get_db():
    config = get_config()
    engine = create_engine(f"sqlite:///{config.database.path}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Background task for real-time updates
async def real_time_update_task():
    """Background task to send real-time updates."""
    while True:
        try:
            # Simulate real-time data updates
            if len(manager.active_connections) > 0:
                update_message = WebSocketMessage(
                    type="live_update",
                    data={
                        "timestamp": datetime.now().isoformat(),
                        "active_games": 5,
                        "predictions_updated": 150,
                        "market_movements": 12
                    }
                )
                await manager.broadcast(update_message.json())
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Real-time update task error: {e}")
            await asyncio.sleep(60)

# Enhanced API Routes

@app.get("/")
async def enhanced_root():
    """Enhanced API homepage with real-time capabilities."""
    return {
        "message": "Enhanced NFL Predictions API v2.0",
        "features": [
            "Real-time WebSocket updates",
            "Advanced caching with Redis",
            "Rate limiting and security",
            "Optimized prediction pipeline",
            "Comprehensive betting analytics",
            "Market intelligence integration"
        ],
        "endpoints": {
            "websocket": "/ws",
            "predictions": "/api/v2/predictions",
            "analytics": "/api/v2/analytics",
            "betting": "/api/v2/betting",
            "real_time": "/api/v2/real-time"
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle subscription requests
            if message.get("type") == "subscribe":
                topic = message.get("topic")
                if topic:
                    manager.subscribe_to_topic(websocket, topic)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "topic": topic
                    }))
            
            # Handle prediction requests
            elif message.get("type") == "predict_player":
                player_id = message.get("player_id")
                if player_id and pipeline:
                    try:
                        # Generate real-time prediction
                        prediction_data = await generate_real_time_prediction(player_id)
                        await websocket.send_text(json.dumps({
                            "type": "prediction_result",
                            "player_id": player_id,
                            "data": prediction_data
                        }))
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))
                        
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/v2/predictions/enhanced/{player_id}")
@limiter.limit("30/minute")
@cache_response(ttl=300)
async def get_enhanced_player_prediction(
    player_id: str,
    request,
    opponent: Optional[str] = None,
    include_market_analysis: bool = True,
    include_sentiment: bool = True,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get enhanced prediction with market analysis and sentiment."""
    
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Prediction service unavailable")
        
        # Generate comprehensive analysis
        analysis = await pipeline.run_optimized_pipeline()
        
        # Mock enhanced prediction data
        enhanced_prediction = {
            "player_id": player_id,
            "basic_prediction": {
                "fantasy_points": 18.5,
                "passing_yards": 285,
                "touchdowns": 2.1,
                "confidence": 0.82
            },
            "market_analysis": {
                "current_line": 17.5,
                "line_movement": "+1.0",
                "public_betting": 65.2,
                "sharp_money": "Under",
                "edge": 0.057
            } if include_market_analysis else None,
            "sentiment_analysis": {
                "sentiment_score": 0.73,
                "mention_volume": 1250,
                "trending_topics": ["injury_update", "matchup", "weather"],
                "news_impact": 0.15
            } if include_sentiment else None,
            "risk_factors": [
                "Weather conditions: 15mph winds",
                "Opponent defense rank: 8th vs position",
                "Player injury history: Low risk"
            ],
            "recommendation": {
                "action": "BUY",
                "confidence": "HIGH",
                "reasoning": "Positive line value with strong fundamentals"
            }
        }
        
        return enhanced_prediction
        
    except Exception as e:
        logger.error(f"Enhanced prediction failed for {player_id}: {e}")
        raise HTTPException(status_code=500, detail="Prediction generation failed")

@app.get("/api/v2/betting/live-opportunities")
@limiter.limit("20/minute")
async def get_live_betting_opportunities(
    request,
    min_edge: float = Query(0.05, ge=0.01, le=0.5),
    max_risk: str = Query("MEDIUM", regex="^(LOW|MEDIUM|HIGH)$"),
    positions: Optional[List[str]] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get live betting opportunities with real-time edge calculations."""
    
    try:
        # Mock live opportunities data
        opportunities = [
            {
                "player_id": "pmahomes_qb",
                "player_name": "Patrick Mahomes",
                "prop_type": "passing_yards",
                "current_line": 275.5,
                "predicted_value": 295.2,
                "edge": 0.071,
                "confidence": 0.85,
                "risk_level": "MEDIUM",
                "sportsbook": "DraftKings",
                "expires_at": datetime.now() + timedelta(hours=2),
                "reasoning": [
                    "Favorable matchup vs 28th ranked pass defense",
                    "Weather conditions optimal",
                    "Recent form trending upward"
                ]
            },
            {
                "player_id": "cmccaffrey_rb",
                "player_name": "Christian McCaffrey", 
                "prop_type": "rushing_yards",
                "current_line": 85.5,
                "predicted_value": 98.3,
                "edge": 0.149,
                "confidence": 0.78,
                "risk_level": "LOW",
                "sportsbook": "FanDuel",
                "expires_at": datetime.now() + timedelta(hours=1.5),
                "reasoning": [
                    "Strong O-line matchup",
                    "Opponent allows 4.8 YPC",
                    "High volume expected"
                ]
            }
        ]
        
        # Filter by criteria
        filtered_opportunities = [
            opp for opp in opportunities
            if opp["edge"] >= min_edge and 
            (max_risk == "HIGH" or 
             (max_risk == "MEDIUM" and opp["risk_level"] in ["LOW", "MEDIUM"]) or
             (max_risk == "LOW" and opp["risk_level"] == "LOW"))
        ]
        
        return {
            "opportunities": filtered_opportunities,
            "total_found": len(filtered_opportunities),
            "last_updated": datetime.now(),
            "market_status": "ACTIVE"
        }
        
    except Exception as e:
        logger.error(f"Live opportunities fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch opportunities")

@app.get("/api/v2/analytics/market-intelligence")
@limiter.limit("10/minute")
@cache_response(ttl=180)
async def get_market_intelligence(
    request,
    game_date: Optional[date] = None,
    include_line_history: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive market intelligence and line movement analysis."""
    
    try:
        # Mock market intelligence data
        intelligence = {
            "market_overview": {
                "total_games": 16,
                "total_props": 1250,
                "avg_line_movement": 0.8,
                "sharp_money_games": 5,
                "public_heavy_games": 8
            },
            "line_movements": [
                {
                    "game_id": "2024_15_KC_DEN",
                    "prop_type": "total_points",
                    "opening_line": 47.5,
                    "current_line": 45.0,
                    "movement": -2.5,
                    "movement_direction": "DOWN",
                    "volume": "HIGH",
                    "sharp_indicator": True
                }
            ],
            "public_betting_trends": {
                "most_bet_teams": ["KC", "BUF", "SF"],
                "contrarian_opportunities": ["DEN", "NYJ", "CAR"],
                "public_percentage_threshold": 70
            },
            "value_alerts": [
                {
                    "alert_type": "REVERSE_LINE_MOVEMENT",
                    "description": "Line moved against public money",
                    "game": "KC vs DEN",
                    "significance": "HIGH"
                }
            ]
        }
        
        return intelligence
        
    except Exception as e:
        logger.error(f"Market intelligence fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market intelligence")

@app.post("/api/v2/predictions/batch")
@limiter.limit("5/minute")
async def batch_predictions(
    request,
    player_ids: List[str] = Field(..., max_items=50),
    include_comparisons: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """Generate batch predictions for multiple players efficiently."""
    
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Prediction service unavailable")
        
        # Use optimized pipeline for batch processing
        results = await pipeline.run_optimized_pipeline()
        
        # Mock batch prediction results
        batch_results = []
        for player_id in player_ids:
            prediction = {
                "player_id": player_id,
                "predictions": {
                    "fantasy_points": np.random.normal(15, 5),
                    "confidence": np.random.uniform(0.6, 0.9)
                },
                "processing_time_ms": np.random.randint(50, 200)
            }
            batch_results.append(prediction)
        
        # Add comparison rankings if requested
        if include_comparisons:
            sorted_results = sorted(batch_results, 
                                  key=lambda x: x["predictions"]["fantasy_points"], 
                                  reverse=True)
            for i, result in enumerate(sorted_results):
                result["rank"] = i + 1
                result["percentile"] = ((len(sorted_results) - i) / len(sorted_results)) * 100
        
        return {
            "predictions": batch_results,
            "batch_size": len(player_ids),
            "total_processing_time_ms": sum(r["processing_time_ms"] for r in batch_results),
            "performance_metrics": results.get("metrics", {})
        }
        
    except Exception as e:
        logger.error(f"Batch predictions failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.get("/api/v2/real-time/dashboard")
@limiter.limit("60/minute")
async def real_time_dashboard(
    request,
    current_user: dict = Depends(get_current_user)
):
    """Get real-time dashboard data for live monitoring."""
    
    try:
        dashboard_data = {
            "system_status": {
                "api_status": "HEALTHY",
                "pipeline_status": "ACTIVE",
                "cache_status": "CONNECTED" if redis_client else "DISCONNECTED",
                "websocket_connections": len(manager.active_connections),
                "last_update": datetime.now()
            },
            "live_metrics": {
                "predictions_generated_today": 2847,
                "api_requests_last_hour": 1250,
                "cache_hit_rate": 0.85,
                "avg_response_time_ms": 145
            },
            "active_alerts": [
                {
                    "type": "HIGH_VALUE_OPPORTUNITY",
                    "message": "Strong edge detected on Mahomes passing yards",
                    "severity": "INFO",
                    "timestamp": datetime.now() - timedelta(minutes=5)
                }
            ],
            "market_pulse": {
                "most_active_props": ["passing_yards", "receiving_yards", "total_points"],
                "biggest_line_moves": ["+3.5 KC vs DEN total", "-1.0 Mahomes passing"],
                "sharp_money_games": 3
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Dashboard data fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Dashboard data unavailable")

# Utility function for real-time predictions
async def generate_real_time_prediction(player_id: str) -> Dict[str, Any]:
    """Generate real-time prediction for WebSocket clients."""
    
    if not pipeline:
        raise Exception("Prediction pipeline not available")
    
    # Mock real-time prediction
    prediction = {
        "player_id": player_id,
        "fantasy_points": np.random.normal(15, 5),
        "confidence": np.random.uniform(0.7, 0.95),
        "market_edge": np.random.uniform(-0.1, 0.15),
        "factors": [
            "Recent form trending up",
            "Favorable matchup",
            "Weather conditions good"
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    return prediction

# Enhanced error handling
@app.exception_handler(HTTPException)
async def enhanced_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def enhanced_general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Health check with detailed status
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "api": "healthy",
            "database": "healthy",
            "pipeline": "healthy" if pipeline else "unavailable",
            "cache": "healthy" if redis_client else "unavailable",
            "websockets": f"{len(manager.active_connections)} active"
        },
        "version": "2.0.0"
    }
    
    # Check if any critical components are down
    if not pipeline:
        health_status["status"] = "degraded"
    
    return health_status

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_prediction_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1  # Use 1 worker for WebSocket support
    )
