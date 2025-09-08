"""
NFL Predictions API & Dashboard
FastAPI-based REST API and web dashboard for accessing NFL player predictions,
game outcomes, and betting insights.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, select, and_, or_, desc, asc
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# Import our models and components
from database_models import (
    Player, Team, Game, PlayerGameStats, BettingLine,
    PlayerPrediction, GamePrediction, ModelPerformance
)
from prediction_pipeline import NFLPredictionPipeline, PipelineConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "postgresql://user:password@localhost/nfl_predictions"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Security
security = HTTPBearer()

# Pydantic models for API responses
class PlayerInfo(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    age: Optional[int] = None
    years_experience: Optional[int] = None

class PlayerPredictionResponse(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str
    game_date: date
    opponent: str
    is_home: bool
    predictions: Dict[str, Any]
    confidence_overall: float
    model_version: str
    prediction_timestamp: datetime

class GamePredictionResponse(BaseModel):
    game_id: str
    game_date: date
    home_team: str
    away_team: str
    predicted_home_score: float
    predicted_away_score: float
    predicted_total_points: float
    predicted_spread: float
    home_win_probability: float
    away_win_probability: float
    confidence_score: float

class BettingInsight(BaseModel):
    player_id: str
    prop_type: str
    predicted_value: float
    betting_line: Optional[float] = None
    edge: Optional[float] = None
    recommendation: str
    confidence: float

class PerformanceMetrics(BaseModel):
    model_version: str
    position: str
    accuracy_metrics: Dict[str, float]
    total_predictions: int
    evaluation_period: str

# API Request models
class PredictionRequest(BaseModel):
    player_ids: Optional[List[str]] = None
    team: Optional[str] = None
    position: Optional[str] = None
    game_date: Optional[date] = None
    include_confidence: bool = True

class BettingAnalysisRequest(BaseModel):
    game_date: Optional[date] = None
    sportsbook: Optional[str] = None
    min_edge: float = 0.05
    min_confidence: float = 0.7

# Global pipeline instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup for FastAPI app lifecycle."""
    global pipeline
    
    # Startup
    logger.info("Initializing NFL Prediction API...")
    
    config = PipelineConfig(
        database_url=DATABASE_URL,
        data_collection_enabled=False,  # API mode - no data collection
        feature_engineering_enabled=True,
        model_retraining_enabled=False,
        enable_scheduler=False
    )
    
    pipeline = NFLPredictionPipeline(config)
    await pipeline.initialize()
    
    logger.info("API initialization completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NFL Prediction API...")

# Create FastAPI app
app = FastAPI(
    title="NFL Player Performance Predictions API",
    description="Advanced ML-powered NFL player statistics and game outcome predictions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication dependency (simplified for demo)
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In production, implement proper JWT validation
    if credentials.credentials != "demo_token":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return {"user_id": "demo_user"}

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard homepage."""
    return """
    <html>
        <head>
            <title>NFL Predictions Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #1f4e79; color: white; padding: 20px; border-radius: 8px; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 4px; }
                pre { background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏈 NFL Predictions API Dashboard</h1>
                <p>Advanced machine learning predictions for NFL player performance and game outcomes</p>
            </div>
            
            <div class="section">
                <h2>📊 Available Endpoints</h2>
                
                <div class="endpoint">
                    <strong>GET /players</strong> - Get all players with optional filtering
                </div>
                
                <div class="endpoint">
                    <strong>GET /predictions/players</strong> - Get player performance predictions
                </div>
                
                <div class="endpoint">
                    <strong>GET /predictions/games</strong> - Get game outcome predictions
                </div>
                
                <div class="endpoint">
                    <strong>GET /betting/insights</strong> - Get betting recommendations
                </div>
                
                <div class="endpoint">
                    <strong>GET /performance/models</strong> - Get model performance metrics
                </div>
            </div>
            
            <div class="section">
                <h2>🔧 Quick Start</h2>
                <p>Access the interactive API documentation at:</p>
                <ul>
                    <li><a href="/docs">/docs</a> - Swagger UI</li>
                    <li><a href="/redoc">/redoc</a> - ReDoc</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📈 Example Usage</h2>
                <pre>
# Get predictions for Kansas City Chiefs players
curl -H "Authorization: Bearer demo_token" \\
     "http://localhost:8000/predictions/players?team=KC"

# Get upcoming game predictions
curl -H "Authorization: Bearer demo_token" \\
     "http://localhost:8000/predictions/games?days_ahead=7"

# Get betting insights
curl -H "Authorization: Bearer demo_token" \\
     "http://localhost:8000/betting/insights?min_edge=0.1"
                </pre>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

# Player endpoints
@app.get("/players", response_model=List[PlayerInfo])
async def get_players(
    position: Optional[str] = Query(None, description="Filter by position (QB, RB, WR, TE)"),
    team: Optional[str] = Query(None, description="Filter by team"),
    active_only: bool = Query(True, description="Only return active players"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get players with optional filtering."""
    
    query = db.query(Player)
    
    if active_only:
        query = query.filter(Player.is_active == True)
    if position:
        query = query.filter(Player.position == position.upper())
    if team:
        query = query.filter(Player.current_team == team.upper())
        
    players = query.limit(limit).all()
    
    return [
        PlayerInfo(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            team=player.current_team or "FA",
            age=player.age,
            years_experience=player.years_experience
        )
        for player in players
    ]

@app.get("/players/{player_id}")
async def get_player_details(
    player_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information for a specific player."""
    
    player = db.query(Player).filter(Player.player_id == player_id).first()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
        
    # Get recent game stats
    recent_stats = db.query(PlayerGameStats).join(
        Game, PlayerGameStats.game_id == Game.game_id
    ).filter(
        PlayerGameStats.player_id == player_id
    ).order_by(desc(Game.game_date)).limit(10).all()
    
    return {
        "player_info": PlayerInfo(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            team=player.current_team or "FA",
            age=player.age,
            years_experience=player.years_experience
        ),
        "recent_stats": [
            {
                "game_date": stat.game.game_date if hasattr(stat, 'game') else None,
                "opponent": stat.opponent,
                "passing_yards": stat.passing_yards,
                "rushing_yards": stat.rushing_yards,
                "receiving_yards": stat.receiving_yards,
                "fantasy_points": stat.fantasy_points_ppr
            }
            for stat in recent_stats
        ]
    }

# Prediction endpoints
@app.get("/predictions/players", response_model=List[PlayerPredictionResponse])
async def get_player_predictions(
    team: Optional[str] = Query(None, description="Filter by team"),
    position: Optional[str] = Query(None, description="Filter by position"),
    player_id: Optional[str] = Query(None, description="Specific player ID"),
    game_date: Optional[date] = Query(None, description="Specific game date"),
    days_ahead: int = Query(7, ge=1, le=30, description="Days ahead to predict"),
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get player performance predictions."""
    
    # Build query for predictions
    query = db.query(
        PlayerPrediction,
        Player.name,
        Player.position,
        Player.current_team,
        Game.game_date,
        Game.home_team,
        Game.away_team
    ).join(
        Player, PlayerPrediction.player_id == Player.player_id
    ).join(
        Game, PlayerPrediction.game_id == Game.game_id
    )
    
    # Apply filters
    if game_date:
        query = query.filter(Game.game_date == game_date)
    else:
        end_date = date.today() + timedelta(days=days_ahead)
        query = query.filter(
            and_(
                Game.game_date >= date.today(),
                Game.game_date <= end_date
            )
        )
        
    if team:
        query = query.filter(Player.current_team == team.upper())
    if position:
        query = query.filter(Player.position == position.upper())
    if player_id:
        query = query.filter(PlayerPrediction.player_id == player_id)
        
    query = query.filter(PlayerPrediction.confidence_overall >= min_confidence)
    
    results = query.order_by(desc(Game.game_date)).limit(100).all()
    
    predictions = []
    for pred, name, pos, team, game_date, home_team, away_team in results:
        # Determine opponent and home/away status
        player_team = team
        is_home = player_team == home_team
        opponent = away_team if is_home else home_team
        
        # Build predictions dictionary
        prediction_dict = {}
        if pred.predicted_passing_yards:
            prediction_dict['passing_yards'] = pred.predicted_passing_yards
        if pred.predicted_passing_tds:
            prediction_dict['passing_touchdowns'] = pred.predicted_passing_tds
        if pred.predicted_rushing_yards:
            prediction_dict['rushing_yards'] = pred.predicted_rushing_yards
        if pred.predicted_rushing_tds:
            prediction_dict['rushing_touchdowns'] = pred.predicted_rushing_tds
        if pred.predicted_receiving_yards:
            prediction_dict['receiving_yards'] = pred.predicted_receiving_yards
        if pred.predicted_receiving_tds:
            prediction_dict['receiving_touchdowns'] = pred.predicted_receiving_tds
        if pred.predicted_receptions:
            prediction_dict['receptions'] = pred.predicted_receptions
        if pred.predicted_fantasy_points:
            prediction_dict['fantasy_points'] = pred.predicted_fantasy_points
            
        predictions.append(
            PlayerPredictionResponse(
                player_id=pred.player_id,
                player_name=name,
                position=pos,
                team=player_team,
                game_date=game_date,
                opponent=opponent,
                is_home=is_home,
                predictions=prediction_dict,
                confidence_overall=pred.confidence_overall,
                model_version=pred.model_version,
                prediction_timestamp=pred.prediction_timestamp
            )
        )
        
    return predictions

@app.get("/predictions/games", response_model=List[GamePredictionResponse])
async def get_game_predictions(
    game_date: Optional[date] = Query(None, description="Specific game date"),
    days_ahead: int = Query(7, ge=1, le=30),
    team: Optional[str] = Query(None, description="Filter by team (home or away)"),
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get game outcome predictions."""
    
    query = db.query(GamePrediction, Game).join(
        Game, GamePrediction.game_id == Game.game_id
    )
    
    # Apply filters
    if game_date:
        query = query.filter(Game.game_date == game_date)
    else:
        end_date = date.today() + timedelta(days=days_ahead)
        query = query.filter(
            and_(
                Game.game_date >= date.today(),
                Game.game_date <= end_date
            )
        )
        
    if team:
        query = query.filter(
            or_(
                Game.home_team == team.upper(),
                Game.away_team == team.upper()
            )
        )
        
    query = query.filter(GamePrediction.confidence_score >= min_confidence)
    
    results = query.order_by(desc(Game.game_date)).limit(50).all()
    
    return [
        GamePredictionResponse(
            game_id=pred.game_id,
            game_date=game.game_date,
            home_team=game.home_team,
            away_team=game.away_team,
            predicted_home_score=pred.predicted_home_score,
            predicted_away_score=pred.predicted_away_score,
            predicted_total_points=pred.predicted_total_points,
            predicted_spread=pred.predicted_spread,
            home_win_probability=pred.home_win_probability,
            away_win_probability=pred.away_win_probability,
            confidence_score=pred.confidence_score
        )
        for pred, game in results
    ]

# Betting endpoints
@app.get("/betting/insights", response_model=List[BettingInsight])
async def get_betting_insights(
    game_date: Optional[date] = Query(None, description="Specific game date"),
    days_ahead: int = Query(3, ge=1, le=14),
    min_edge: float = Query(0.05, ge=0.0, le=1.0),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0),
    prop_type: Optional[str] = Query(None, description="Type of prop bet"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get betting insights and recommendations."""
    
    # Get player predictions with betting lines
    query = db.query(
        PlayerPrediction,
        Player.name,
        Player.position,
        BettingLine.line_value,
        BettingLine.prop_type,
        Game.game_date
    ).join(
        Player, PlayerPrediction.player_id == Player.player_id
    ).join(
        Game, PlayerPrediction.game_id == Game.game_id
    ).outerjoin(
        BettingLine,
        and_(
            BettingLine.game_id == Game.game_id,
            BettingLine.player_id == Player.player_id
        )
    )
    
    # Apply filters
    if game_date:
        query = query.filter(Game.game_date == game_date)
    else:
        end_date = date.today() + timedelta(days=days_ahead)
        query = query.filter(
            and_(
                Game.game_date >= date.today(),
                Game.game_date <= end_date
            )
        )
        
    if prop_type:
        query = query.filter(BettingLine.prop_type == prop_type)
        
    query = query.filter(PlayerPrediction.confidence_overall >= min_confidence)
    
    results = query.all()
    
    insights = []
    for pred, name, position, betting_line, bet_prop_type, game_date in results:
        # Calculate edges for different prop types
        prop_mappings = {
            'passing_yards': pred.predicted_passing_yards,
            'rushing_yards': pred.predicted_rushing_yards,
            'receiving_yards': pred.predicted_receiving_yards,
            'receptions': pred.predicted_receptions,
            'fantasy_points': pred.predicted_fantasy_points
        }
        
        for prop, predicted_value in prop_mappings.items():
            if predicted_value is None:
                continue
                
            # Skip if we have a specific prop type filter
            if bet_prop_type and prop != bet_prop_type:
                continue
                
            # Calculate edge if we have a betting line
            edge = None
            recommendation = "Hold"
            
            if betting_line is not None:
                edge = (predicted_value - betting_line) / betting_line
                
                if edge >= min_edge:
                    recommendation = "Over"
                elif edge <= -min_edge:
                    recommendation = "Under"
                    
            else:
                # No line available, base recommendation on confidence
                if pred.confidence_overall >= 0.8:
                    recommendation = "Strong Play"
                elif pred.confidence_overall >= 0.7:
                    recommendation = "Consider"
                    
            # Only include if it meets edge requirements
            if edge is None or abs(edge) >= min_edge:
                insights.append(
                    BettingInsight(
                        player_id=pred.player_id,
                        prop_type=prop,
                        predicted_value=predicted_value,
                        betting_line=betting_line,
                        edge=edge,
                        recommendation=recommendation,
                        confidence=pred.confidence_overall
                    )
                )
                
    # Sort by edge (highest first)
    insights.sort(key=lambda x: abs(x.edge) if x.edge else 0, reverse=True)
    
    return insights[:50]  # Limit to top 50

# Performance endpoints
@app.get("/performance/models", response_model=List[PerformanceMetrics])
async def get_model_performance(
    position: Optional[str] = Query(None, description="Filter by position"),
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get model performance metrics."""
    
    query = db.query(ModelPerformance)
    
    if model_version:
        query = query.filter(ModelPerformance.model_version == model_version)
        
    if position:
        query = query.filter(ModelPerformance.evaluation_period.contains(position))
        
    results = query.order_by(desc(ModelPerformance.created_at)).limit(limit).all()
    
    return [
        PerformanceMetrics(
            model_version=perf.model_version,
            position=perf.evaluation_period.split('_')[0] if '_' in perf.evaluation_period else "All",
            accuracy_metrics=perf.accuracy_metrics,
            total_predictions=perf.total_predictions,
            evaluation_period=perf.evaluation_period
        )
        for perf in results
    ]

# Analytics endpoints
@app.get("/analytics/player-trends/{player_id}")
async def get_player_trends(
    player_id: str,
    weeks_back: int = Query(10, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get performance trends for a specific player."""
    
    # Get recent game stats
    stats = db.query(PlayerGameStats, Game.game_date).join(
        Game, PlayerGameStats.game_id == Game.game_id
    ).filter(
        PlayerGameStats.player_id == player_id
    ).order_by(desc(Game.game_date)).limit(weeks_back).all()
    
    if not stats:
        raise HTTPException(status_code=404, detail="No stats found for player")
        
    # Calculate trends
    fantasy_points = [stat[0].fantasy_points_ppr for stat in stats if stat[0].fantasy_points_ppr]
    game_dates = [stat[1] for stat in stats]
    
    if len(fantasy_points) < 2:
        trend = 0
    else:
        # Simple linear trend
        x = list(range(len(fantasy_points)))
        trend = np.polyfit(x, fantasy_points, 1)[0]
        
    return {
        "player_id": player_id,
        "weeks_analyzed": len(stats),
        "average_fantasy_points": np.mean(fantasy_points) if fantasy_points else 0,
        "trend_slope": float(trend),
        "trend_direction": "up" if trend > 0 else "down" if trend < 0 else "flat",
        "recent_games": [
            {
                "game_date": game_date,
                "fantasy_points": stat.fantasy_points_ppr,
                "passing_yards": stat.passing_yards,
                "rushing_yards": stat.rushing_yards,
                "receiving_yards": stat.receiving_yards
            }
            for stat, game_date in stats
        ]
    }

@app.get("/analytics/team-performance/{team}")
async def get_team_performance(
    team: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get aggregated team performance analytics."""
    
    # Get team players
    players = db.query(Player).filter(
        and_(
            Player.current_team == team.upper(),
            Player.is_active == True
        )
    ).all()
    
    if not players:
        raise HTTPException(status_code=404, detail="Team not found or no active players")
        
    # Get recent team stats
    team_stats = []
    for player in players:
        recent_stats = db.query(PlayerGameStats).join(
            Game, PlayerGameStats.game_id == Game.game_id
        ).filter(
            PlayerGameStats.player_id == player.player_id
        ).order_by(desc(Game.game_date)).limit(5).all()
        
        team_stats.extend(recent_stats)
        
    # Calculate team metrics
    total_fantasy = sum(stat.fantasy_points_ppr for stat in team_stats if stat.fantasy_points_ppr)
    avg_fantasy = total_fantasy / len(team_stats) if team_stats else 0
    
    position_breakdown = {}
    for player in players:
        pos = player.position
        if pos not in position_breakdown:
            position_breakdown[pos] = 0
        position_breakdown[pos] += 1
        
    return {
        "team": team.upper(),
        "total_active_players": len(players),
        "position_breakdown": position_breakdown,
        "average_team_fantasy_points": avg_fantasy,
        "total_games_analyzed": len(team_stats)
    }

# Background task endpoint
@app.post("/admin/trigger-prediction-update")
async def trigger_prediction_update(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger a manual prediction update (admin only)."""
    
    async def run_prediction_update():
        try:
            if pipeline:
                await pipeline.run_daily_pipeline()
                logger.info("Manual prediction update completed")
        except Exception as e:
            logger.error(f"Manual prediction update failed: {e}")
            
    background_tasks.add_task(run_prediction_update)
    
    return {"message": "Prediction update triggered", "status": "running"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "prediction_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )