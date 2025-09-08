#!/usr/bin/env python3
"""
NFL Betting Analyzer - Unified Command Line Interface
Consolidated CLI combining all system functionality for 2025 season.
"""

import asyncio
import click
import sys
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime, date
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "core"))

from core.database_models import (
    create_all_tables, get_db_session, validate_player_status, migrate_database,
    Player, PlayerGameStats, Game, Team
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--database', default='nfl_predictions.db', help='Database file path')
@click.pass_context
def cli(ctx, debug: bool, database: str):
    """NFL Betting Analyzer - 2025 Season Ready System"""
    ctx.ensure_object(dict)
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo("Debug mode enabled")
    
    ctx.obj['database'] = database
    ctx.obj['debug'] = debug


@cli.group()
@click.pass_context
def setup(ctx):
    """Database and system setup commands"""
    pass


@setup.command()
@click.pass_context
def database(ctx):
    """Initialize database with 2025 season schema"""
    db_path = ctx.obj['database']
    
    try:
        click.echo("Setting up NFL database...")
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Migrate existing database if needed
        click.echo("Migrating database schema...")
        migrate_database(engine)
        
        # Create all tables
        create_all_tables(engine)
        
        # Validate player status (mark retired players)
        click.echo("Validating player status...")
        validate_player_status()
        
        click.echo("Database setup complete!")
        click.echo(f"Database location: {db_path}")
        
    except Exception as e:
        click.echo(f"Database setup failed: {e}")
        sys.exit(1)


@setup.command()
@click.pass_context
def validate(ctx):
    """Validate system readiness for 2025 season"""
    db_path = ctx.obj['database']
    
    try:
        session = get_db_session(f"sqlite:///{db_path}")
        
        # Check for retired players marked as active
        click.echo("Checking for retired players...")
        retired_names = ['Tom Brady', 'Matt Ryan', 'Ben Roethlisberger', 'Rob Gronkowski']
        
        for name in retired_names:
            player = session.query(Player).filter(Player.name.ilike(f'%{name}%')).first()
            if player:
                if player.is_active:
                    click.echo(f"WARNING: {name} is marked as active but should be retired")
                else:
                    click.echo(f"OK: {name} correctly marked as retired")
        
        # Check data freshness
        click.echo("Checking data freshness...")
        latest_game = session.query(Game).filter(Game.season == 2025).first()
        if latest_game:
            click.echo(f"OK: 2025 season data found")
        else:
            click.echo(f"WARNING: No 2025 season data found")
        
        session.close()
        
    except Exception as e:
        click.echo(f"Validation failed: {e}")


@cli.group()
@click.pass_context
def data(ctx):
    """Data collection and management commands"""
    pass


@data.command()
@click.option('--season', default=2025, help='Season year')
@click.option('--week', help='Specific week to collect')
@click.pass_context
def collect(ctx, season: int, week: Optional[int]):
    """Collect current NFL data for 2025 season"""
    click.echo(f"Collecting NFL data for {season} season...")
    
    try:
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        
        # Import and run the 2025 data collector
        from core.data.nfl_2025_data_collector import NFL2025DataCollector
        
        collector = NFL2025DataCollector(session)
        results = asyncio.run(collector.collect_comprehensive_data())
        
        click.echo("Data collection complete!")
        click.echo("Results:")
        for key, value in results.items():
            click.echo(f"  {key}: {value}")
        
        session.close()
        
    except Exception as e:
        click.echo(f"Data collection failed: {e}")
        if ctx.obj.get('debug'):
            import traceback
            traceback.print_exc()


@data.command()
@click.pass_context
def process(ctx):
    """Run data processing pipeline"""
    click.echo("Running NFL data processing pipeline...")
    
    try:
        import asyncio
        from core.data.data_processing_pipeline import NFLDataProcessingPipeline
        
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        pipeline = NFLDataProcessingPipeline(session)
        
        # Run the pipeline
        result = asyncio.run(pipeline.run_full_pipeline())
        
        click.echo("Data processing complete!")
        click.echo("Results:")
        click.echo(f"  Records Processed: {result.records_processed}")
        click.echo(f"  Records Updated: {result.records_updated}")
        click.echo(f"  Records Created: {result.records_created}")
        click.echo(f"  Errors Found: {result.errors_found}")
        click.echo(f"  Overall Quality Score: {result.quality_metrics.overall_score:.1f}%")
        
        if result.quality_metrics.issues_found:
            click.echo("Issues Found:")
            for issue in result.quality_metrics.issues_found[:10]:  # Show first 10
                click.echo(f"  - {issue}")
        
        session.close()
        
    except Exception as e:
        click.echo(f"Data processing failed: {e}")
        if ctx.obj.get('debug'):
            import traceback
            traceback.print_exc()


@data.command()
@click.pass_context
def validate_data(ctx):
    """Validate data quality and accuracy"""
    click.echo("Validating data quality...")
    
    try:
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        
        # Count total players
        total_players = session.query(Player).count()
        active_players = session.query(Player).filter(Player.is_active == True).count()
        retired_players = session.query(Player).filter(Player.is_retired == True).count()
        
        click.echo(f"Total players: {total_players}")
        click.echo(f"Active players: {active_players}")
        click.echo(f"Retired players: {retired_players}")
        
        # Check for data inconsistencies
        inconsistent = session.query(Player).filter(
            Player.is_active == True, 
            Player.is_retired == True
        ).count()
        
        if inconsistent > 0:
            click.echo(f"WARNING: Found {inconsistent} players marked as both active and retired")
        else:
            click.echo("OK: No active/retired conflicts found")
        
        session.close()
        
    except Exception as e:
        click.echo(f"Data validation failed: {e}")


@cli.group()
@click.pass_context
def predict(ctx):
    """NFL prediction commands"""
    pass


@predict.command()
@click.option('--target', default='fantasy_points_ppr', help='Target statistic to predict')
@click.pass_context
def train(ctx, target: str):
    """Train prediction models"""
    click.echo(f"Training NFL prediction models for {target}...")
    
    try:
        from core.models.streamlined_models import StreamlinedNFLModels
        
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        models = StreamlinedNFLModels(session)
        
        # Train all position models
        results = models.train_all_models(target)
        
        click.echo("Model training complete!")
        click.echo("Results:")
        
        for position, position_results in results.items():
            click.echo(f"  {position}:")
            for result in position_results:
                click.echo(f"    {result.model_type}: R² = {result.r2_score:.3f}, "
                          f"RMSE = {result.rmse:.3f}, Samples = {result.sample_count}")
        
        # Show model summary
        summary = models.get_model_summary()
        click.echo(f"Total Models Trained: {len(summary['available_models'])}")
        
        session.close()
        
    except Exception as e:
        click.echo(f"Model training failed: {e}")
        if ctx.obj.get('debug'):
            import traceback
            traceback.print_exc()


@predict.command()
@click.option('--player', help='Player name to predict')
@click.option('--position', help='Position to analyze (QB/RB/WR/TE)')
@click.option('--week', type=int, help='Week to predict for')
@click.option('--season', default=2025, help='Season year')
@click.pass_context
def player(ctx, player: Optional[str], position: Optional[str], week: Optional[int], season: int):
    """Generate player performance predictions"""
    
    if not player and not position:
        click.echo("❌ Must specify either --player or --position")
        return
    
    try:
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        
        if player:
            # Find specific player
            player_obj = session.query(Player).filter(
                Player.name.ilike(f'%{player}%'),
                Player.is_active == True
            ).first()
            
            if not player_obj:
                click.echo(f"Player '{player}' not found or not active")
                return
            
            click.echo(f"Predictions for {player_obj.name} ({player_obj.position})")
            click.echo("=" * 50)
            
            # Generate predictions (placeholder)
            click.echo("Generating predictions...")
            click.echo("Prediction engine integration pending")
            
        elif position:
            # List top players by position
            players = session.query(Player).filter(
                Player.position == position.upper(),
                Player.is_active == True
            ).limit(10).all()
            
            click.echo(f"Top {position.upper()} predictions for Week {week or 'Next'}:")
            click.echo("=" * 50)
            
            for p in players:
                click.echo(f"• {p.name} ({p.current_team})")
        
        session.close()
        
    except Exception as e:
        click.echo(f"Prediction failed: {e}")


@predict.command()
@click.option('--week', type=int, help='Week to predict')
@click.option('--season', default=2025, help='Season year')
@click.pass_context
def games(ctx, week: Optional[int], season: int):
    """Generate game outcome predictions"""
    
    try:
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        
        # Find upcoming games
        upcoming_games = session.query(Game).filter(
            Game.season == season,
            Game.game_status == 'scheduled'
        )
        
        if week:
            upcoming_games = upcoming_games.filter(Game.week == week)
        
        games_list = upcoming_games.limit(10).all()
        
        if not games_list:
            click.echo(f"❌ No upcoming games found for season {season}")
            return
        
        click.echo(f"🏈 Game predictions for {season} season:")
        click.echo("=" * 50)
        
        for game in games_list:
            click.echo(f"Week {game.week}: {game.away_team} @ {game.home_team}")
            click.echo("  📊 Prediction engine integration pending")
            click.echo()
        
        session.close()
        
    except Exception as e:
        click.echo(f"❌ Game prediction failed: {e}")


@cli.group()
@click.pass_context
def stats(ctx):
    """Statistical analysis commands"""
    pass


@stats.command()
@click.option('--player', help='Player name')
@click.option('--season', default=2025, help='Season year')
@click.pass_context
def player_stats(ctx, player: str, season: int):
    """Get comprehensive player statistics"""
    
    if not player:
        click.echo("❌ Player name required")
        return
    
    try:
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        
        # Find player
        player_obj = session.query(Player).filter(
            Player.name.ilike(f'%{player}%')
        ).first()
        
        if not player_obj:
            click.echo(f"❌ Player '{player}' not found")
            return
        
        click.echo(f"📊 Statistics for {player_obj.name}")
        click.echo("=" * 50)
        click.echo(f"Position: {player_obj.position}")
        click.echo(f"Team: {player_obj.current_team}")
        click.echo(f"Status: {'Active' if player_obj.is_active else 'Inactive'}")
        
        if player_obj.is_retired:
            click.echo(f"🏁 Retired: {player_obj.retirement_date}")
        
        # Get recent game stats
        recent_stats = session.query(PlayerGameStats).filter(
            PlayerGameStats.player_id == player_obj.player_id
        ).order_by(PlayerGameStats.created_at.desc()).limit(5).all()
        
        if recent_stats:
            click.echo("\n🏈 Recent Games:")
            for stat in recent_stats:
                click.echo(f"  vs {stat.opponent}: {stat.fantasy_points_ppr:.1f} PPR points")
        
        session.close()
        
    except Exception as e:
        click.echo(f"❌ Stats lookup failed: {e}")


@stats.command()
@click.option('--position', help='Position (QB/RB/WR/TE)')
@click.option('--limit', default=10, help='Number of players to show')
@click.pass_context
def leaders(ctx, position: Optional[str], limit: int):
    """Show statistical leaders by position"""
    
    try:
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        
        query = session.query(Player).filter(Player.is_active == True)
        
        if position:
            query = query.filter(Player.position == position.upper())
        
        players = query.limit(limit).all()
        
        click.echo(f"🏆 Top {limit} {'Active Players' if not position else position.upper() + ' Players'}:")
        click.echo("=" * 50)
        
        for i, player in enumerate(players, 1):
            click.echo(f"{i:2d}. {player.name} ({player.position}) - {player.current_team}")
        
        session.close()
        
    except Exception as e:
        click.echo(f"❌ Leaders lookup failed: {e}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and health"""
    
    try:
        session = get_db_session(f"sqlite:///{ctx.obj['database']}")
        
        click.echo("NFL Betting Analyzer System Status")
        click.echo("=" * 40)
        
        # Database stats
        total_players = session.query(Player).count()
        active_players = session.query(Player).filter(Player.is_active == True).count()
        total_games = session.query(Game).count()
        games_2025 = session.query(Game).filter(Game.season == 2025).count()
        
        click.echo(f"Database Statistics:")
        click.echo(f"  Total Players: {total_players}")
        click.echo(f"  Active Players: {active_players}")
        click.echo(f"  Total Games: {total_games}")
        click.echo(f"  2025 Games: {games_2025}")
        
        # System health
        click.echo(f"System Health:")
        click.echo(f"  Database: {'Connected' if session else 'Error'}")
        click.echo(f"  2025 Ready: {'Yes' if games_2025 > 0 else 'No 2025 data'}")
        
        session.close()
        
    except Exception as e:
        click.echo(f"Status check failed: {e}")


if __name__ == '__main__':
    cli()
