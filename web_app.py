"""
NFL Prediction Web Application

Interactive web interface for browsing teams, players, matchups, and predictions
"""

from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Optional, Dict, Any
import asyncio
from datetime import date, datetime

from simplified_database_models import Player, Game
from comprehensive_stats_engine import ComprehensiveStatsEngine
from game_prediction_engine import GamePredictionEngine
from real_time_nfl_system import RealTimeNFLSystem

app = FastAPI(title="NFL Prediction System", description="Interactive NFL betting analysis")

# Database setup
engine = create_engine("sqlite:///nfl_predictions.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Initialize engines
nfl_system = RealTimeNFLSystem()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    """Home page with overview"""
    
    # Get quick stats
    total_players = db.query(Player).count()
    total_games = db.query(Game).count()
    
    # Get recent games
    recent_games = db.query(Game).order_by(Game.game_date.desc()).limit(5).all()
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "total_players": total_players,
        "total_games": total_games,
        "recent_games": recent_games
    })

@app.get("/teams", response_class=HTMLResponse)
async def teams_page(request: Request, db: Session = Depends(get_db)):
    """Teams overview page"""
    
    # Get all teams
    teams_query = db.query(Player.current_team).filter(
        Player.current_team.isnot(None)
    ).distinct().all()
    
    teams = [team[0] for team in teams_query if team[0]]
    teams.sort()
    
    return templates.TemplateResponse("teams.html", {
        "request": request,
        "teams": teams
    })

@app.get("/team/{team_code}", response_class=HTMLResponse)
async def team_detail(request: Request, team_code: str, db: Session = Depends(get_db)):
    """Individual team detail page"""
    
    # Get team players by position
    team_players = db.query(Player).filter(
        Player.current_team == team_code,
        Player.is_active == True
    ).order_by(Player.position, Player.name).all()
    
    # Group by position
    players_by_position = {}
    for player in team_players:
        pos = player.position
        if pos not in players_by_position:
            players_by_position[pos] = []
        players_by_position[pos].append(player)
    
    # Get recent games
    recent_games = db.query(Game).filter(
        (Game.home_team == team_code) | (Game.away_team == team_code)
    ).order_by(Game.game_date.desc()).limit(10).all()
    
    return templates.TemplateResponse("team_detail.html", {
        "request": request,
        "team_code": team_code,
        "players_by_position": players_by_position,
        "recent_games": recent_games
    })

@app.get("/player/{player_id}", response_class=HTMLResponse)
async def player_detail(request: Request, player_id: str, db: Session = Depends(get_db)):
    """Individual player detail page"""
    
    player = db.query(Player).filter(Player.player_id == player_id).first()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get comprehensive stats
    stats_engine = ComprehensiveStatsEngine(db)
    comprehensive_stats = stats_engine.get_comprehensive_player_stats(player_id)
    
    # Get ML predictions
    try:
        predictions = await nfl_system.predict_player_performance(player_id, player.position)
    except:
        predictions = {}
    
    return templates.TemplateResponse("player_detail.html", {
        "request": request,
        "player": player,
        "stats": comprehensive_stats,
        "predictions": predictions
    })

@app.get("/games", response_class=HTMLResponse)
async def games_page(request: Request, db: Session = Depends(get_db)):
    """Games overview page"""
    
    # Get upcoming games
    upcoming_games = db.query(Game).filter(
        Game.game_date >= date.today()
    ).order_by(Game.game_date).limit(20).all()
    
    # Get recent games
    recent_games = db.query(Game).filter(
        Game.game_date < date.today()
    ).order_by(Game.game_date.desc()).limit(10).all()
    
    return templates.TemplateResponse("games.html", {
        "request": request,
        "upcoming_games": upcoming_games,
        "recent_games": recent_games
    })

@app.get("/game/{game_id}", response_class=HTMLResponse)
async def game_detail(request: Request, game_id: str, db: Session = Depends(get_db)):
    """Individual game detail page with complete predictions"""
    
    game = db.query(Game).filter(Game.game_id == game_id).first()
    
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Generate complete game prediction
    game_engine = GamePredictionEngine(db)
    
    try:
        game_prediction = await game_engine.predict_complete_game(game_id)
    except Exception as e:
        # Fallback if prediction fails
        game_prediction = None
    
    return templates.TemplateResponse("game_detail.html", {
        "request": request,
        "game": game,
        "prediction": game_prediction
    })

@app.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request, 
                          position: Optional[str] = Query(None),
                          team: Optional[str] = Query(None),
                          db: Session = Depends(get_db)):
    """Predictions overview page"""
    
    # Get players for predictions
    query = db.query(Player).filter(Player.is_active == True)
    
    if position:
        query = query.filter(Player.position == position)
    if team:
        query = query.filter(Player.current_team == team)
    
    players = query.limit(50).all()
    
    # Generate predictions for each player
    predictions = []
    stats_engine = ComprehensiveStatsEngine(db)
    
    for player in players[:20]:  # Limit to avoid timeout
        try:
            comprehensive_stats = stats_engine.get_comprehensive_player_stats(player.player_id)
            if comprehensive_stats.fantasy_points_ppr > 0:
                predictions.append({
                    'player': player,
                    'stats': comprehensive_stats
                })
        except:
            continue
    
    return templates.TemplateResponse("predictions.html", {
        "request": request,
        "predictions": predictions,
        "selected_position": position,
        "selected_team": team
    })

# API Endpoints

@app.get("/api/player/{player_id}/stats")
async def api_player_stats(player_id: str, db: Session = Depends(get_db)):
    """API endpoint for player comprehensive stats"""
    
    stats_engine = ComprehensiveStatsEngine(db)
    
    try:
        stats = stats_engine.get_comprehensive_player_stats(player_id)
        return {
            "player_id": stats.player_id,
            "name": stats.name,
            "position": stats.position,
            "team": stats.team,
            "stats": {
                "passing": {
                    "attempts": stats.passing_attempts,
                    "completions": stats.passing_completions,
                    "yards": stats.passing_yards,
                    "touchdowns": stats.passing_touchdowns,
                    "interceptions": stats.passing_interceptions,
                    "rating": stats.passing_rating
                },
                "rushing": {
                    "attempts": stats.rushing_attempts,
                    "yards": stats.rushing_yards,
                    "touchdowns": stats.rushing_touchdowns,
                    "yards_per_carry": stats.rushing_yards_per_carry
                },
                "receiving": {
                    "targets": stats.targets,
                    "receptions": stats.receptions,
                    "yards": stats.receiving_yards,
                    "touchdowns": stats.receiving_touchdowns,
                    "catch_percentage": stats.receiving_catch_percentage
                },
                "fantasy": {
                    "standard": stats.fantasy_points_standard,
                    "ppr": stats.fantasy_points_ppr,
                    "half_ppr": stats.fantasy_points_half_ppr
                },
                "betting": {
                    "anytime_td_prob": stats.anytime_touchdown_probability,
                    "total_yards": stats.over_under_yards,
                    "confidence": stats.prediction_confidence
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/game/{game_id}/prediction")
async def api_game_prediction(game_id: str, db: Session = Depends(get_db)):
    """API endpoint for complete game prediction"""
    
    game_engine = GamePredictionEngine(db)
    
    try:
        prediction = await game_engine.predict_complete_game(game_id)
        
        return {
            "game_id": game_id,
            "score_prediction": {
                "home_team": prediction.game_info.home_team,
                "away_team": prediction.game_info.away_team,
                "home_score": prediction.game_info.home_score,
                "away_score": prediction.game_info.away_score,
                "total_points": prediction.game_info.total_points,
                "spread": prediction.game_info.spread,
                "game_script": prediction.game_info.game_script,
                "confidence": prediction.game_info.score_confidence
            },
            "top_performers": prediction.top_fantasy_performers,
            "game_totals": {
                "total_passing_yards": prediction.total_passing_yards,
                "total_rushing_yards": prediction.total_rushing_yards,
                "total_touchdowns": prediction.total_touchdowns
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/position/{position}/rankings")
async def api_position_rankings(position: str, db: Session = Depends(get_db)):
    """API endpoint for position rankings"""
    
    stats_engine = ComprehensiveStatsEngine(db)
    
    try:
        comprehensive_stats = stats_engine.get_all_position_comprehensive_stats(position, limit=20)
        
        rankings = []
        for i, stats in enumerate(comprehensive_stats, 1):
            rankings.append({
                "rank": i,
                "player_id": stats.player_id,
                "name": stats.name,
                "team": stats.team,
                "fantasy_ppr": stats.fantasy_points_ppr,
                "anytime_td_prob": stats.anytime_touchdown_probability,
                "total_yards": stats.over_under_yards,
                "confidence": stats.prediction_confidence
            })
        
        return {
            "position": position,
            "rankings": rankings
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
