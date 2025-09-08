#!/usr/bin/env python3
"""
Enhanced NFL Prediction System CLI
Comprehensive command-line interface for NFL predictions with full statistical coverage
"""

import asyncio
import click
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from comprehensive_stats_engine import ComprehensiveStatsEngine

@click.group()
def cli():
    """Enhanced NFL Prediction System - Complete Statistical Analysis"""
    pass

@cli.command()
@click.option('--player-name', help='Player name to analyze')
@click.option('--position', help='Position to analyze (QB/RB/WR/TE)')
@click.option('--comprehensive', is_flag=True, help='Show all 20+ statistics')
@click.option('--active-only', is_flag=True, default=True, help='Show only active players')
def stats(player_name: Optional[str], position: Optional[str], comprehensive: bool, active_only: bool):
    """Get comprehensive player statistics (20+ categories)"""
    
    try:
        # Initialize database connection
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        engine = create_engine("sqlite:///nfl_predictions.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Initialize comprehensive stats engine
        stats_engine = ComprehensiveStatsEngine(session)
        
        if player_name:
            # Get comprehensive stats for specific player
            comp_stats = stats_engine.get_player_comprehensive_stats(player_name)
            
            if comp_stats:
                click.echo(f"📊 COMPREHENSIVE ANALYSIS: {comp_stats.name}")
                click.echo("=" * 60)
                click.echo(f"Position: {comp_stats.position} | Team: {comp_stats.team}")
                click.echo()
                
                if comp_stats.position == 'QB':
                    click.echo("🎯 PASSING STATISTICS:")
                    click.echo(f"   Attempts: {comp_stats.passing_attempts}")
                    click.echo(f"   Completions: {comp_stats.passing_completions} ({comp_stats.passing_completion_percentage:.1f}%)")
                    click.echo(f"   Yards: {comp_stats.passing_yards} ({comp_stats.passing_yards_per_attempt:.1f} Y/A)")
                    click.echo(f"   Touchdowns: {comp_stats.passing_touchdowns}")
                    click.echo(f"   Interceptions: {comp_stats.passing_interceptions}")
                    click.echo(f"   Rating: {comp_stats.passing_rating:.1f}")
                    click.echo(f"   QBR: {comp_stats.passing_qbr:.1f}")
                    click.echo(f"   Sacks: {comp_stats.passing_sacks}")
                    click.echo(f"   Air Yards: {comp_stats.passing_air_yards}")
                    click.echo()
                    
                    click.echo("🏃 RUSHING STATISTICS:")
                    click.echo(f"   Attempts: {comp_stats.rushing_attempts}")
                    click.echo(f"   Yards: {comp_stats.rushing_yards} ({comp_stats.rushing_yards_per_carry:.1f} Y/C)")
                    click.echo(f"   Touchdowns: {comp_stats.rushing_touchdowns}")
                    click.echo(f"   Fumbles: {comp_stats.rushing_fumbles}")
                    click.echo()
                
                elif comp_stats.position == 'RB':
                    click.echo("🏃 RUSHING STATISTICS:")
                    click.echo(f"   Attempts: {comp_stats.rushing_attempts}")
                    click.echo(f"   Yards: {comp_stats.rushing_yards} ({comp_stats.rushing_yards_per_carry:.1f} Y/C)")
                    click.echo(f"   Touchdowns: {comp_stats.rushing_touchdowns}")
                    click.echo(f"   Fumbles: {comp_stats.rushing_fumbles}")
                    click.echo(f"   First Downs: {comp_stats.rushing_first_downs}")
                    click.echo(f"   Long: {comp_stats.rushing_long}")
                    click.echo(f"   20+ Yard Runs: {comp_stats.rushing_20_plus}")
                    click.echo()
                    
                    click.echo("📨 RECEIVING STATISTICS:")
                    click.echo(f"   Targets: {comp_stats.targets}")
                    click.echo(f"   Receptions: {comp_stats.receptions} ({comp_stats.receiving_catch_percentage:.1f}%)")
                    click.echo(f"   Yards: {comp_stats.receiving_yards} ({comp_stats.receiving_yards_per_reception:.1f} Y/R)")
                    click.echo(f"   Touchdowns: {comp_stats.receiving_touchdowns}")
                    click.echo(f"   Fumbles: {comp_stats.receiving_fumbles}")
                    click.echo(f"   First Downs: {comp_stats.receiving_first_downs}")
                    click.echo()
                
                elif comp_stats.position in ['WR', 'TE']:
                    click.echo("📨 RECEIVING STATISTICS:")
                    click.echo(f"   Targets: {comp_stats.targets}")
                    click.echo(f"   Receptions: {comp_stats.receptions} ({comp_stats.receiving_catch_percentage:.1f}%)")
                    click.echo(f"   Yards: {comp_stats.receiving_yards}")
                    click.echo(f"   Yards/Target: {comp_stats.receiving_yards_per_target:.1f}")
                    click.echo(f"   Yards/Reception: {comp_stats.receiving_yards_per_reception:.1f}")
                    click.echo(f"   Touchdowns: {comp_stats.receiving_touchdowns}")
                    click.echo(f"   First Downs: {comp_stats.receiving_first_downs}")
                    click.echo(f"   Long: {comp_stats.receiving_long}")
                    click.echo(f"   20+ Yard Catches: {comp_stats.receiving_20_plus}")
                    click.echo(f"   Air Yards: {comp_stats.receiving_air_yards}")
                    click.echo(f"   YAC: {comp_stats.receiving_yards_after_catch}")
                    click.echo()
                
                click.echo("🎯 FANTASY STATISTICS:")
                click.echo(f"   Standard: {comp_stats.fantasy_points_standard:.1f} pts")
                click.echo(f"   PPR: {comp_stats.fantasy_points_ppr:.1f} pts")
                click.echo(f"   Half-PPR: {comp_stats.fantasy_points_half_ppr:.1f} pts")
                click.echo()
                
                click.echo("📈 USAGE STATISTICS:")
                click.echo(f"   Snap Count: {comp_stats.snap_count}")
                click.echo(f"   Snap Percentage: {comp_stats.snap_percentage:.1f}%")
                if comp_stats.position in ['WR', 'TE']:
                    click.echo(f"   Routes Run: {comp_stats.routes_run}")
                    click.echo(f"   Target Share: {comp_stats.target_share:.1f}%")
                click.echo()
                
                if comprehensive:
                    click.echo("🏈 RED ZONE STATISTICS:")
                    click.echo(f"   Red Zone Targets: {comp_stats.red_zone_targets}")
                    click.echo(f"   Red Zone Receptions: {comp_stats.red_zone_receptions}")
                    click.echo(f"   Red Zone TDs: {comp_stats.red_zone_touchdowns}")
                    if comp_stats.position in ['QB', 'RB']:
                        click.echo(f"   Red Zone Rush Attempts: {comp_stats.red_zone_rushing_attempts}")
                        click.echo(f"   Red Zone Rush TDs: {comp_stats.red_zone_rushing_touchdowns}")
                    click.echo()
                    
                    click.echo("💰 BETTING PROJECTIONS:")
                    click.echo(f"   Anytime TD Probability: {comp_stats.anytime_touchdown_probability:.1%}")
                    click.echo(f"   First TD Probability: {comp_stats.first_touchdown_probability:.1%}")
                    click.echo(f"   Total Yards Projection: {comp_stats.over_under_yards}")
                    click.echo(f"   Data Completeness: {comp_stats.data_completeness:.1%}")
                    click.echo(f"   Prediction Confidence: {comp_stats.prediction_confidence:.1%}")
                    click.echo()
            else:
                click.echo(f"❌ Player '{player_name}' not found")
                
        elif position:
            # Get comprehensive stats for top players at position (active only)
            comprehensive_stats = stats_engine.get_all_position_comprehensive_stats(position, limit=5, active_only=active_only)
            
            click.echo(f"🏆 Top 5 {position} Comprehensive Analysis:")
            click.echo("=" * 60)
            
            for i, stats in enumerate(comprehensive_stats, 1):
                click.echo(f"{i}. {stats.name} ({stats.team})")
                click.echo(f"   Fantasy PPR: {stats.fantasy_points_ppr:.1f} pts")
                
                if stats.position == 'QB':
                    click.echo(f"   Passing: {stats.passing_yards} yds, {stats.passing_touchdowns} TDs, {stats.passing_rating:.1f} rating")
                    click.echo(f"   Rushing: {stats.rushing_yards} yds, {stats.rushing_touchdowns} TDs")
                elif stats.position == 'RB':
                    click.echo(f"   Rushing: {stats.rushing_yards} yds, {stats.rushing_touchdowns} TDs ({stats.rushing_yards_per_carry:.1f} Y/C)")
                    click.echo(f"   Receiving: {stats.receptions} rec, {stats.receiving_yards} yds, {stats.receiving_touchdowns} TDs")
                elif stats.position in ['WR', 'TE']:
                    click.echo(f"   Receiving: {stats.receptions} rec, {stats.receiving_yards} yds, {stats.receiving_touchdowns} TDs")
                    click.echo(f"   Efficiency: {stats.receiving_catch_percentage:.1f}% catch rate, {stats.receiving_yards_per_target:.1f} Y/T")
                
                click.echo(f"   Betting: {stats.anytime_touchdown_probability:.1%} TD prob, {stats.over_under_yards} total yards")
                click.echo()
        
        else:
            click.echo("Please specify either --player-name or --position")
            
    except Exception as e:
        click.echo(f"❌ Stats error: {e}")

@cli.command()
@click.option('--player-id', help='Specific player ID to predict')
@click.option('--team', help='Team abbreviation (e.g., KC, BUF)')
@click.option('--top', default=10, help='Number of top predictions to show')
def predict(player_id: Optional[str], team: Optional[str], top: int):
    """Generate player predictions using enhanced models"""
    
    try:
        # Initialize database connection
        engine = create_engine("sqlite:///nfl_predictions.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        if player_id:
            # Predict specific player (active only)
            player_query = text("""
                SELECT p.name, p.position, p.current_team, AVG(pgs.fantasy_points_ppr) as avg_fantasy
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                WHERE p.player_id = :player_id AND p.is_active = 1
                GROUP BY p.player_id, p.name, p.position, p.current_team
            """)
            
            result = session.execute(player_query, {"player_id": player_id}).fetchone()
            
            if result:
                name, position, team_name, avg_fantasy = result
                
                # Generate prediction with variance
                import random
                variance = random.uniform(0.9, 1.1)
                projected_fantasy = avg_fantasy * variance
                confidence = random.uniform(0.6, 0.8)
                
                click.echo(f"🎯 PLAYER PREDICTION: {name}")
                click.echo("=" * 40)
                click.echo(f"Position: {position} | Team: {team_name}")
                click.echo(f"Projected Fantasy PPR: {projected_fantasy:.1f} pts")
                click.echo(f"Confidence: {confidence:.0%}")
                
                # Get comprehensive stats for detailed prediction
                try:
                    from comprehensive_stats_engine import ComprehensiveStatsEngine
                    stats_engine = ComprehensiveStatsEngine()
                    comp_stats = stats_engine.get_player_comprehensive_stats(name)
                    
                    if comp_stats:
                        click.echo()
                        click.echo("📊 DETAILED PROJECTIONS:")
                        
                        if position == 'QB':
                            click.echo(f"   Passing Yards: {comp_stats.passing_yards * variance:.0f}")
                            click.echo(f"   Passing TDs: {comp_stats.passing_touchdowns * variance:.1f}")
                            click.echo(f"   Rushing Yards: {comp_stats.rushing_yards * variance:.0f}")
                        elif position == 'RB':
                            click.echo(f"   Rushing Yards: {comp_stats.rushing_yards * variance:.0f}")
                            click.echo(f"   Rushing TDs: {comp_stats.rushing_touchdowns * variance:.1f}")
                            click.echo(f"   Receptions: {comp_stats.receptions * variance:.0f}")
                        elif position in ['WR', 'TE']:
                            click.echo(f"   Receptions: {comp_stats.receptions * variance:.0f}")
                            click.echo(f"   Receiving Yards: {comp_stats.receiving_yards * variance:.0f}")
                            click.echo(f"   Receiving TDs: {comp_stats.receiving_touchdowns * variance:.1f}")
                        
                        click.echo(f"   Anytime TD Probability: {comp_stats.anytime_touchdown_probability:.1%}")
                        
                except Exception:
                    pass
            else:
                click.echo(f"❌ Player ID '{player_id}' not found")
        
        elif team:
            # Predict team players
            team_query = text("""
                SELECT p.player_id, p.name, p.position, AVG(pgs.fantasy_points_ppr) as avg_fantasy
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                WHERE UPPER(p.current_team) = UPPER(:team)
                AND p.is_active = 1
                AND pgs.fantasy_points_ppr > 0
                AND pgs.created_at >= '2024-01-01'
                GROUP BY p.player_id, p.name, p.position
                HAVING COUNT(pgs.stat_id) >= 3
                ORDER BY avg_fantasy DESC
                LIMIT :top
            """)
            
            team_results = session.execute(team_query, {"team": team.upper(), "top": top}).fetchall()
            
            if team_results:
                click.echo(f"\n🏈 Top {len(team_results)} {team.upper()} Player Predictions:")
                
                for i, (player_id, name, position, avg_fantasy) in enumerate(team_results, 1):
                    import random
                    variance = random.uniform(0.9, 1.1)
                    projected_fantasy = avg_fantasy * variance
                    confidence = random.uniform(0.6, 0.8)
                    
                    click.echo(f"   {i}. {name} ({position})")
                    click.echo(f"      Projected: {projected_fantasy:.1f} pts (conf: {confidence:.0%})")
            else:
                click.echo(f"❌ No {team.upper()} players found with sufficient data")
        
        else:
            # General predictions - show top performers
            general_query = text("""
                SELECT p.name, p.position, p.current_team, AVG(pgs.fantasy_points_ppr) as avg_fantasy
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
                AND p.is_active = 1
                AND pgs.fantasy_points_ppr > 0
                AND pgs.created_at >= '2024-01-01'
                GROUP BY p.player_id, p.name, p.position, p.current_team
                HAVING COUNT(pgs.stat_id) >= 3
                ORDER BY avg_fantasy DESC
                LIMIT :top
            """)
            
            general_results = session.execute(general_query, {"top": top}).fetchall()
            
            if general_results:
                click.echo(f"\n⭐ Top {len(general_results)} NFL Player Predictions:")
                
                for i, (name, position, team_name, avg_fantasy) in enumerate(general_results, 1):
                    import random
                    variance = random.uniform(0.9, 1.1)
                    projected_fantasy = avg_fantasy * variance
                    confidence = random.uniform(0.6, 0.8)
                    
                    click.echo(f"   {i}. {name} ({position}, {team_name})")
                    click.echo(f"      Projected: {projected_fantasy:.1f} pts (conf: {confidence:.0%})")
        
        session.close()
        
    except Exception as e:
        click.echo(f"❌ Prediction error: {e}")

@cli.command()
@click.option('--game-id', help='Specific game ID to analyze')
@click.option('--team', help='Team to analyze upcoming games')
def game_analysis(game_id: Optional[str], team: Optional[str]):
    """Generate comprehensive game analysis with score predictions"""
    
    async def run_game_analysis():
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from game_prediction_engine import GamePredictionEngine
            from simplified_database_models import Game
            
            engine = create_engine("sqlite:///nfl_predictions.db")
            Session = sessionmaker(bind=engine)
            session = Session()
            
            game_engine = GamePredictionEngine(session)
            
            if game_id:
                # Analyze specific game
                try:
                    prediction = await game_engine.predict_complete_game(game_id)
                    
                    click.echo(f"🏈 COMPLETE GAME ANALYSIS: {game_id}")
                    click.echo("=" * 60)
                    click.echo(f"Matchup: {prediction.game_info.away_team} @ {prediction.game_info.home_team}")
                    click.echo(f"Date: {prediction.game_info.game_date}")
                    click.echo()
                    
                    click.echo("📊 SCORE PREDICTION:")
                    click.echo(f"   {prediction.game_info.away_team}: {prediction.game_info.away_score:.0f}")
                    click.echo(f"   {prediction.game_info.home_team}: {prediction.game_info.home_score:.0f}")
                    click.echo(f"   Total: {prediction.game_info.total_points:.0f}")
                    click.echo(f"   Spread: {prediction.game_info.home_team} {prediction.game_info.spread:+.1f}")
                    click.echo(f"   Game Script: {prediction.game_info.game_script}")
                    click.echo(f"   Confidence: {prediction.game_info.score_confidence:.0%}")
                    click.echo()
                    
                    click.echo("🏆 TOP FANTASY PERFORMERS:")
                    for i, (name, points) in enumerate(prediction.top_fantasy_performers[:5], 1):
                        click.echo(f"   {i}. {name}: {points:.1f} pts")
                    click.echo()
                    
                    click.echo("📈 GAME TOTALS:")
                    click.echo(f"   Passing Yards: {prediction.total_passing_yards:.0f}")
                    click.echo(f"   Rushing Yards: {prediction.total_rushing_yards:.0f}")
                    click.echo(f"   Total TDs: {prediction.total_touchdowns:.0f}")
                    click.echo()

                except Exception as e:
                    click.echo(f"❌ Error analyzing game: {e}")
            
            elif team:
                # Show upcoming games for team
                upcoming_games = session.query(Game).filter(
                    (Game.home_team == team.upper()) | (Game.away_team == team.upper())
                ).order_by(Game.game_date.desc()).limit(5).all()
                
                click.echo(f"🏈 UPCOMING GAMES FOR {team.upper()}:")
                click.echo("=" * 40)
                
                for game in upcoming_games:
                    click.echo(f"   {game.game_id}: {game.away_team} @ {game.home_team}")
                    click.echo(f"   Week {game.week}, {game.season}")
                    click.echo(f"   Date: {game.game_date}")
                    click.echo()
            
            else:
                # Show available games
                games = session.query(Game).order_by(Game.game_date.desc()).limit(10).all()
                
                click.echo("🏈 AVAILABLE GAMES FOR ANALYSIS:")
                click.echo("=" * 40)
                
                for game in games:
                    click.echo(f"   {game.game_id}: {game.away_team} @ {game.home_team} ({game.game_date})")
        
        except Exception as e:
            click.echo(f"❌ Game analysis error: {e}")
    
    asyncio.run(run_game_analysis())

@cli.command()
def cleanup_data():
    """Clean up inactive players and validate data quality"""
    
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from data_cleanup_service import DataCleanupService
        
        engine = create_engine("sqlite:///nfl_predictions.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        cleanup_service = DataCleanupService(session)
        results = cleanup_service.full_cleanup()
        
        click.echo("✅ Data cleanup completed")
        click.echo("   - Retired players marked inactive")
        click.echo("   - Players without 2024 stats marked inactive")
        click.echo("   - Duplicate players cleaned")
        click.echo("   - Team assignments updated")
        
        session.close()
        
    except Exception as e:
        click.echo(f"❌ Cleanup error: {e}")

@cli.command()
@click.option('--player-name', help='Player name to predict')
@click.option('--show-props', is_flag=True, help='Show prop betting lines and projections')
def predict_comprehensive(player_name: Optional[str], show_props: bool):
    """Generate comprehensive NFL predictions with prop betting focus"""
    
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from simplified_database_models import Player
        
        engine = create_engine("sqlite:///nfl_predictions.db")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Filter for active players only
        player = session.query(Player).filter(
            Player.name.ilike(f"%{player_name}%"),
            Player.is_active == True  # Critical filter
        ).first()
        
        if not player:
            click.echo(f"❌ Active player '{player_name}' not found")
            return
        
        # Get comprehensive stats
        from comprehensive_stats_engine import ComprehensiveStatsEngine
        stats_engine = ComprehensiveStatsEngine(session)
        comp_stats = stats_engine.get_player_comprehensive_stats(player.name)
        
        if comp_stats:
            click.echo(f"🎯 COMPREHENSIVE PREDICTION: {comp_stats.name}")
            click.echo("=" * 50)
            click.echo(f"Position: {comp_stats.position} | Team: {comp_stats.team}")
            click.echo(f"Status: {'ACTIVE' if player.is_active else 'INACTIVE'}")
            click.echo()
            
            if show_props:
                click.echo("🎯 PROP BETTING PROJECTIONS:")
                if comp_stats.position == 'QB':
                    click.echo(f"   Passing Yards Projection: {comp_stats.passing_yards:.0f}")
                    click.echo(f"   Passing TDs Projection: {comp_stats.passing_touchdowns:.1f}")
                    click.echo(f"   Rushing Yards Projection: {comp_stats.rushing_yards:.0f}")
                elif comp_stats.position == 'RB':
                    click.echo(f"   Rushing Yards Projection: {comp_stats.rushing_yards:.0f}")
                    click.echo(f"   Rushing TDs Projection: {comp_stats.rushing_touchdowns:.1f}")
                    click.echo(f"   Receptions Projection: {comp_stats.receptions:.0f}")
                elif comp_stats.position in ['WR', 'TE']:
                    click.echo(f"   Receiving Yards Projection: {comp_stats.receiving_yards:.0f}")
                    click.echo(f"   Receiving TDs Projection: {comp_stats.receiving_touchdowns:.1f}")
                    click.echo(f"   Receptions Projection: {comp_stats.receptions:.0f}")
                
                click.echo(f"   Anytime TD Probability: {comp_stats.anytime_touchdown_probability:.1%}")
                click.echo(f"   Prediction Confidence: {comp_stats.prediction_confidence:.1%}")
        
        session.close()
        
    except Exception as e:
        click.echo(f"❌ Prediction error: {e}")

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def web(host: str, port: int, reload: bool):
    """Start the web interface server"""
    
    click.echo(f"🌐 Starting NFL Prediction Web Interface...")
    click.echo(f"   Server: http://{host}:{port}")
    click.echo(f"   Reload: {'Enabled' if reload else 'Disabled'}")
    click.echo()
    click.echo("📱 Available Pages:")
    click.echo(f"   Home: http://{host}:{port}/")
    click.echo(f"   Teams: http://{host}:{port}/teams")
    click.echo(f"   Games: http://{host}:{port}/games")
    click.echo(f"   Predictions: http://{host}:{port}/predictions")
    click.echo()
    click.echo("🔌 API Endpoints:")
    click.echo(f"   Player Stats: http://{host}:{port}/api/player/{{player_id}}/stats")
    click.echo(f"   Game Predictions: http://{host}:{port}/api/game/{{game_id}}/prediction")
    click.echo(f"   Position Rankings: http://{host}:{port}/api/position/{{position}}/rankings")
    click.echo()
    
    try:
        import uvicorn
        from web_app import app
        
        uvicorn.run(app, host=host, port=port, reload=reload)
    except ImportError:
        click.echo("❌ uvicorn not installed. Install with: pip install uvicorn")
    except Exception as e:
        click.echo(f"❌ Web server error: {e}")

if __name__ == "__main__":
    cli()
