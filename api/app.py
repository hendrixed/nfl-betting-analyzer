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
import json
import logging
import asyncio
import time
import random
from contextlib import asynccontextmanager
from functools import wraps
from dotenv import load_dotenv
try:
    # Optional rate limiting; tests should not require slowapi
    from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore
    from slowapi.errors import RateLimitExceeded  # type: ignore
except Exception:  # pragma: no cover - fallback when slowapi isn't installed
    Limiter = None  # type: ignore
    def get_remote_address(request):  # type: ignore
        return "anonymous"
    class RateLimitExceeded(Exception):  # type: ignore
        pass
    def _rate_limit_exceeded_handler(request, exc):  # type: ignore
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
try:
    from jose import jwt, JWTError  # type: ignore
    USING_DUMMY_JWT = False
except Exception:  # pragma: no cover - optional dependency for tests
    class JWTError(Exception):
        pass
    class _DummyJWT:
        @staticmethod
        def encode(data, key, algorithm="HS256"):
            return "dummy-token"
        @staticmethod
        def decode(token, key, algorithms=None):
            return {"sub": "anonymous"}
    jwt = _DummyJWT()  # type: ignore
    USING_DUMMY_JWT = True
try:
    from passlib.context import CryptContext
except Exception:  # pragma: no cover - optional dependency for tests
    class CryptContext:  # type: ignore
        def __init__(self, schemes=None, deprecated="auto"):
            pass
        def hash(self, password: str) -> str:
            return password
        def verify(self, password: str, hashed: str) -> bool:
            return True

# Import our modules
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database_models import get_db_session, Player, PlayerGameStats, Game

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env if present
try:
    load_dotenv()
except Exception:
    pass

# Rate limiting setup (no-op if slowapi unavailable)
if Limiter is not None:
    limiter = Limiter(key_func=get_remote_address)
else:
    class _NoopLimiter:
        def limit(self, *_args, **_kwargs):
            def decorator(func):
                return func
            return decorator
    limiter = _NoopLimiter()

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Environment-driven security configuration
IS_PROD = os.getenv("APP_ENV", "").lower() == "production" or os.getenv("ENV", "").lower() == "production"
JWT_SECRET = os.getenv("JWT_SECRET") or ("dev-secret" if not IS_PROD else None)
JWT_ALGORITHM = os.getenv("ALGORITHM", "HS256")

if IS_PROD:
    # Enforce presence of real JWT and secret in production
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET must be set in production environment")
    if 'USING_DUMMY_JWT' in globals() and USING_DUMMY_JWT:
        raise RuntimeError("python-jose must be installed; dummy JWT is not allowed in production")

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

class FantasyPredictionResponse(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    fantasy_points_ppr: float
    confidence: float
    model: str
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

# Random helpers (avoid numpy dependency)
def poisson(lam: float) -> int:
    """Sample from Poisson(lam) using Knuth's method."""
    if lam <= 0:
        return 0
    L = pow(2.718281828459045, -lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def latest_snapshot_dir(base: str = "data/snapshots") -> Optional[Path]:
    """Return latest snapshot directory by name (YYYY-MM-DD)."""
    try:
        base_path = Path(base)
        if not base_path.exists():
            return None
        dirs = [p for p in base_path.iterdir() if p.is_dir()]
        if not dirs:
            return None
        return sorted(dirs)[-1]
    except Exception:
        return None

def models_available() -> bool:
    """Return True if any streamlined model files exist on disk."""
    try:
        models_path = Path("models/streamlined")
        if not models_path.exists():
            return False
        return any(models_path.glob("*.pkl"))
    except Exception:
        return False

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
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)  # type: ignore[arg-type]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])  # type: ignore[arg-type]
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(token: dict = Depends(verify_token)):
    return token

# Initialize models
models = None
try:
    session = get_db_session()
    # Lazy import to avoid heavy dependency at import time
    from core.models.streamlined_models import StreamlinedNFLModels
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

# Add rate limiting (works with no-op limiter too)
app.state.limiter = limiter
try:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except Exception:
    # Fallback already handled by optional handler definition
    pass

# Setup templates and static files
try:
    templates = Jinja2Templates(directory="web/templates")
    app.mount("/static", StaticFiles(directory="web/static"), name="static")
    # Expose reports so calibration plots and artifacts can be viewed in the browser
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")
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

# ADMIN ENDPOINTS

@app.post("/admin/odds/snapshot")
@limiter.limit("5/minute")
async def snapshot_mock_odds(
    request: Request,
    max_offers: int = Query(100, ge=1, le=2000),
    db: Session = Depends(get_db)
):
    """Generate a mock odds snapshot file under data/snapshots/YYYY-MM-DD/odds.csv.
    This is a placeholder until real odds integrations are enabled.
    """
    try:
        from core.data.odds_snapshot import write_mock_odds_snapshot
        path = write_mock_odds_snapshot(db, max_offers=max_offers)
        # Count rows written (excluding header)
        rows = 0
        try:
            import csv
            with open(path, newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                next(r, None)
                for _ in r:
                    rows += 1
        except Exception:
            pass
        return {
            "status": "ok",
            "snapshot_path": str(path),
            "rows": rows,
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Failed to snapshot mock odds: {e}")
        raise HTTPException(status_code=500, detail="Failed to snapshot odds")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": models_available()
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
        query = query.filter(Player.current_team == team.upper())
    if active_only:
        query = query.filter(Player.is_active == True)

    players = query.limit(100).all()

    return [
        PlayerInfo(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            team=player.current_team or "",
            status=("active" if getattr(player, "is_active", False) else "inactive")
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
        "team": player.current_team or "",
        "status": ("active" if getattr(player, "is_active", False) else "inactive")
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
                    "touchdowns": (
                        (getattr(stat, "passing_touchdowns", 0) or 0)
                        + (getattr(stat, "rushing_touchdowns", 0) or 0)
                        + (getattr(stat, "receiving_touchdowns", 0) or 0)
                    )
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
    if not models_available():
        raise HTTPException(status_code=503, detail="Prediction models not available")

    query = db.query(Player).filter(Player.is_active == True)

    if team:
        query = query.filter(Player.current_team == team.upper())
    if position:
        query = query.filter(Player.position == position.upper())
    
    players = query.limit(50).all()
    predictions = []
    
    for player in players:
        try:
            # Generate predictions using the models
            player_predictions = {
                "passing_yards": random.gauss(200, 50),
                "rushing_yards": random.gauss(50, 20),
                "receiving_yards": random.gauss(60, 25),
                "touchdowns": poisson(1.2)
            }
            
            predictions.append(PlayerPredictionResponse(
                player_id=player.player_id,
                name=player.name,
                position=player.position,
                team=player.current_team or "",
                predictions=player_predictions,
                last_updated=datetime.now()
            ))
        except Exception as e:
            logger.error(f"Prediction failed for {player.player_id}: {e}")
            continue
    
    return predictions

@app.get("/predictions/players/fantasy", response_model=List[FantasyPredictionResponse])
@limiter.limit("60/minute")
async def get_player_fantasy_predictions(
    request: Request,
    team: Optional[str] = Query(None, description="Filter by team"),
    position: Optional[str] = Query(None, description="Filter by position"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Get fantasy (PPR) predictions using streamlined trained models"""
    if not models_available():
        raise HTTPException(status_code=403, detail="Prediction models not available")

    # Ensure we have a model instance
    local_models = None
    try:
        from core.models.streamlined_models import StreamlinedNFLModels
        local_models = models or StreamlinedNFLModels(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")

    query = db.query(Player).filter(Player.is_active == True)
    if team:
        query = query.filter(Player.current_team == team.upper())
    if position:
        query = query.filter(Player.position == position.upper())

    players = query.limit(limit).all()
    out: List[FantasyPredictionResponse] = []

    for p in players:
        try:
            pred = local_models.predict_player(p.player_id, target_stat='fantasy_points_ppr')
            if not pred:
                continue
            out.append(FantasyPredictionResponse(
                player_id=p.player_id,
                name=p.name,
                position=p.position,
                team=p.current_team or "",
                fantasy_points_ppr=float(pred.predicted_value),
                confidence=float(pred.confidence),
                model=pred.model_used,
                last_updated=datetime.now()
            ))
        except Exception as e:
            logger.debug(f"Fantasy prediction failed for {p.player_id}: {e}")
            continue

    return out

@app.get("/predictions/games", response_model=List[GamePredictionResponse])
async def get_game_predictions(
    game_date: Optional[date] = Query(None, description="Specific game date"),
    days_ahead: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """Get game predictions"""
    if not models_available():
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
                "total_points": random.gauss(45, 8),
                "home_score": random.gauss(24, 7),
                "away_score": random.gauss(21, 7),
                "spread_prediction": random.gauss(0, 3)
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
    players = db.query(Player).filter(Player.is_active == True).limit(20).all()
    
    for player in players:
        try:
            # Mock market data and predictions
            predicted_yards = random.gauss(75, 25)
            market_line = predicted_yards + random.gauss(0, 10)
            edge = (predicted_yards - market_line) / market_line if market_line > 0 else 0
            
            if abs(edge) >= min_edge:
                insights.append(BettingInsight(
                    market=f"{player.position}_receiving_yards",
                    player_name=player.name,
                    game_info=f"{player.current_team or ''} vs TBD",
                    prediction=predicted_yards,
                    line=market_line,
                    edge=edge,
                    confidence=random.uniform(0.6, 0.9),
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
    market: Optional[str] = Query(None, description="Sportsbook market label (e.g., 'Passing Yards')"),
    book: Optional[str] = Query(None, description="Sportsbook name (e.g., 'DraftKings')"),
    week: Optional[int] = Query(None, description="Week number"),
    team: Optional[str] = Query(None, description="Team filter"),
    player: Optional[str] = Query(None, description="Player filter"),
    db: Session = Depends(get_db)
):
    """Get prop betting opportunities with edge analysis.
    Canonicalizes sportsbook identifiers using `core.data.market_mapping`.
    """
    canonical_book = None
    canonical_market = None
    try:
        if market:
            from core.data.market_mapping import to_internal
            canonical_book, canonical_market = to_internal(book or "", market)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid market/book: {e}")

    # Attempt to read odds snapshot
    offers = []
    snap_dir = latest_snapshot_dir()
    if snap_dir is not None:
        odds_path = snap_dir / "odds.csv"
        if odds_path.exists():
            try:
                import csv
                with open(odds_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_book = (row.get("book") or "").lower()
                        row_market = (row.get("market") or "").lower()
                        if canonical_book and row_book != canonical_book:
                            continue
                        if canonical_market and row_market != canonical_market:
                            continue
                        if team:
                            from core.data.market_mapping import normalize_team_name
                            # Prefer canonical abbreviation equality; then fallback to substring
                            team_abbr = normalize_team_name(team)
                            row_team_abbr = normalize_team_name(row.get("team_id") or "")
                            if team_abbr and row_team_abbr:
                                if team_abbr != row_team_abbr:
                                    continue
                            else:
                                team_q = (team or "").strip().lower()
                                row_team = (row.get("team_id") or "").strip().lower()
                                if team_q and team_q not in row_team:
                                    continue
                        if player:
                            player_q = (player or "").strip().lower()
                            row_player = (row.get("player_id") or "").strip().lower()
                            if player_q and player_q not in row_player:
                                continue
                        # Basic typing
                        try:
                            offer = {
                                "timestamp": row.get("timestamp"),
                                "book": row_book,
                                "market": row_market,
                                "player_id": row.get("player_id"),
                                "team_id": row.get("team_id"),
                                "line": float(row.get("line")) if row.get("line") else None,
                                "over_odds": int(row.get("over_odds")) if row.get("over_odds") else None,
                                "under_odds": int(row.get("under_odds")) if row.get("under_odds") else None,
                            }
                            offers.append(offer)
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Failed to read odds snapshot: {e}")

    return {
        "book": canonical_book or (book or "").lower(),
        "market": canonical_market or (market or "").lower(),
        "offers": offers,
        "opportunities": offers,
        "timestamp": datetime.now(),
        "note": "Integration with sportsbook APIs required"
    }

# WEB PAGES
@app.get("/web/odds", response_class=HTMLResponse)
async def web_odds(request: Request):
    """Render a simple odds explorer page (team + player markets)."""
    try:
        return templates.TemplateResponse("odds.html", {"request": request})
    except Exception as e:
        logger.warning(f"Failed to render odds page: {e}")
        return HTMLResponse(content="<h3>Odds page unavailable</h3>", status_code=500)

@app.get("/web/backtests", response_class=HTMLResponse)
async def web_backtests(request: Request):
    """Render a page that lists backtest results from reports/backtests."""
    try:
        return templates.TemplateResponse("backtests.html", {"request": request})
    except Exception as e:
        logger.warning(f"Failed to render backtests page: {e}")
        return HTMLResponse(content="<h3>Backtests page unavailable</h3>", status_code=500)

@app.get("/reports/backtests")
async def list_backtests():
    """Return a list of backtest metrics discovered under reports/backtests/**/metrics.json"""
    items: List[Dict[str, Any]] = []
    try:
        base = Path("reports/backtests")
        if not base.exists():
            return {"backtests": []}
        for metrics_file in base.rglob("metrics.json"):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                rel_plot = m.get("calibration_plot")
                if rel_plot and not str(rel_plot).startswith("/"):
                    rel_plot = f"/{rel_plot}"
                items.append({
                    "market": m.get("market") or metrics_file.parent.name,
                    "target": m.get("target"),
                    "metrics": {
                        "count": m.get("count"),
                        "rmse": m.get("rmse"),
                        "mae": m.get("mae"),
                        "hit_rate": m.get("hit_rate"),
                        "roi": m.get("roi"),
                        "brier": m.get("brier"),
                        "crps": m.get("crps"),
                    },
                    "metrics_path": f"/reports/{metrics_file.relative_to('reports')}",
                    "calibration_plot": rel_plot,
                })
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"list_backtests error: {e}")
        items = []
    return {"backtests": items}

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
    if not models_available():
        raise HTTPException(status_code=503, detail="Simulation models not available")
    
    # Mock simulation results
    results = {}
    for game_id in game_ids:
        results[game_id] = {
            "total_points": {
                "mean": random.gauss(45, 2),
                "std": random.uniform(6, 10),
                "percentiles": {
                    "p10": 35,
                    "p50": 45,
                    "p90": 55
                }
            },
            "spread": {
                "mean": random.gauss(0, 1),
                "prob_home_cover": random.uniform(0.4, 0.6)
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
    results: List[PerformanceMetrics] = []
    try:
        if models_available() and models:
            summary = models.get_model_summary() or {}
            for position, info in summary.items():
                model_name = f"{info.get('model_type','unknown')}_{info.get('target_stat','fantasy_points_ppr')}"
                results.append(PerformanceMetrics(
                    model_name=model_name,
                    position=position,
                    accuracy=float(info.get('r2_score', 0.0)),
                    mae=float(0.0),
                    rmse=float(0.0),
                    last_updated=datetime.now()
                ))
    except Exception as e:
        logger.warning(f"Failed to build models summary: {e}")
    if not results:
        raise HTTPException(status_code=403, detail="Prediction models not available")
    return results

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

# API ENDPOINTS FOR WEB INTERFACE

@app.get("/api/teams")
async def get_teams(db: Session = Depends(get_db)):
    """Get all NFL teams"""
    try:
        # Get unique teams from players table
        teams = db.query(Player.current_team).distinct().filter(Player.current_team.isnot(None)).all()
        team_list = [{"id": team[0], "name": team[0], "abbreviation": team[0]} for team in teams if team[0]]
        return {"teams": sorted(team_list, key=lambda x: x["name"])}
    except Exception as e:
        logger.error(f"Failed to get teams: {e}")
        return {"teams": []}

@app.get("/api/players")
async def get_players(
    team: Optional[str] = Query(None, description="Filter by team"),
    position: Optional[str] = Query(None, description="Filter by position"),
    limit: int = Query(50, ge=1, le=200, description="Number of players to return"),
    db: Session = Depends(get_db)
):
    """Get NFL players with optional filters"""
    try:
        query = db.query(Player).filter(Player.is_active == True)
        
        if team:
            query = query.filter(Player.current_team == team.upper())
        if position:
            query = query.filter(Player.position == position.upper())
            
        players = query.limit(limit).all()
        
        player_list = []
        for player in players:
            player_list.append({
                "id": player.player_id,
                "name": player.name,
                "position": player.position,
                "team": player.current_team,
                "status": "ACT" if player.is_active else "INA"
            })
            
        return {"players": player_list}
    except Exception as e:
        logger.error(f"Failed to get players: {e}")
        return {"players": []}

@app.get("/api/games")
async def get_games(
    week: Optional[int] = Query(None, description="Filter by week"),
    season: Optional[int] = Query(None, description="Filter by season"),
    limit: int = Query(20, ge=1, le=100, description="Number of games to return"),
    db: Session = Depends(get_db)
):
    """Get NFL games with optional filters"""
    try:
        query = db.query(Game).order_by(desc(Game.game_date))
        
        if week:
            query = query.filter(Game.week == week)
        if season:
            query = query.filter(Game.season == season)
            
        games = query.limit(limit).all()
        
        game_list = []
        for game in games:
            game_list.append({
                "id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "week": game.week,
                "season": game.season,
                "game_date": game.game_date.isoformat() if game.game_date else None,
                "home_score": game.home_score,
                "away_score": game.away_score
            })
            
        return {"games": game_list}
    except Exception as e:
        logger.error(f"Failed to get games: {e}")
        return {"games": []}

@app.get("/api/stats")
async def get_stats(
    player_id: Optional[str] = Query(None, description="Filter by player ID"),
    game_id: Optional[str] = Query(None, description="Filter by game ID"),
    limit: int = Query(50, ge=1, le=200, description="Number of stats to return"),
    db: Session = Depends(get_db)
):
    """Get player game statistics"""
    try:
        # Join with Game table to get game_date
        query = db.query(PlayerGameStats, Game.game_date).join(
            Game, PlayerGameStats.game_id == Game.game_id
        ).order_by(desc(Game.game_date))
        
        if player_id:
            query = query.filter(PlayerGameStats.player_id == player_id)
        if game_id:
            query = query.filter(PlayerGameStats.game_id == game_id)
            
        results = query.limit(limit).all()
        
        stats_list = []
        for stat, game_date in results:
            stats_list.append({
                "player_id": stat.player_id,
                "game_id": stat.game_id,
                "game_date": game_date.isoformat() if game_date else None,
                "passing_yards": stat.passing_yards,
                "passing_tds": stat.passing_touchdowns,
                "rushing_yards": stat.rushing_yards,
                "rushing_tds": stat.rushing_touchdowns,
                "receiving_yards": stat.receiving_yards,
                "receiving_tds": stat.receiving_touchdowns,
                "receptions": stat.receptions,
                "fantasy_points_ppr": stat.fantasy_points_ppr
            })
            
        return {"stats": stats_list}
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"stats": []}

@app.get("/api/teams/{team_id}/roster")
async def get_team_roster(team_id: str, db: Session = Depends(get_db)):
    """Get roster for a specific team"""
    try:
        players = db.query(Player).filter(
            Player.current_team == team_id.upper(),
            Player.is_active == True
        ).all()
        
        roster = []
        for player in players:
            roster.append({
                "id": player.player_id,
                "name": player.name,
                "position": player.position,
                "team": player.current_team,
                "status": "ACT" if player.is_active else "INA"
            })
            
        return {"team": team_id.upper(), "roster": roster}
    except Exception as e:
        logger.error(f"Failed to get roster for {team_id}: {e}")
        return {"team": team_id.upper(), "roster": []}

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
