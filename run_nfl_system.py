#!/usr/bin/env python3
"""
NFL Prediction System - Main Entry Point
Run different components of the NFL prediction system.
"""

import click
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import get_config, load_config
from database_models import create_all_tables, PlayerGameStats, Game
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool):
    """NFL Prediction System - Main Entry Point"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        system_config = load_config(config)
    else:
        system_config = get_config()
    
    if debug:
        system_config.debug = True
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.obj['config'] = system_config


@cli.command()
@click.pass_context
def setup(ctx):
    """Setup the complete system including database and directories."""
    config = ctx.obj['config']
    
    logger.info("🔧 Setting up NFL Prediction System...")
    
    # Create database tables
    try:
        engine = create_engine(config.database.url)
        create_all_tables(engine)
        logger.info("✅ Database tables created")
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return
    
    # Create required directories
    directories = [
        "data", "data/raw", "data/processed", "data/features",
        "logs", "models", "models/trained", "models/performance"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directory structure created")
    logger.info("🎉 System setup completed successfully!")


@cli.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
@click.pass_context
def api(ctx, host: str, port: int):
    """Start the API server."""
    config = ctx.obj['config']
    
    try:
        import uvicorn
        from prediction_api import app
        
        logger.info(f"🚀 Starting API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        logger.error("❌ uvicorn not installed. Install with: pip install uvicorn")
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")


@cli.command()
@click.option('--force', is_flag=True, help='Force pipeline run even if recently executed')
@click.pass_context
def pipeline(ctx, force: bool):
    """Run the prediction pipeline."""
    config = ctx.obj['config']
    
    async def run_pipeline():
        try:
            from prediction_pipeline import NFLPredictionPipeline, PipelineConfig
            
            pipeline_config = PipelineConfig(
                database_url=config.database.url,
                **config.pipeline.__dict__
            )
            
            pipeline = NFLPredictionPipeline(pipeline_config)
            await pipeline.initialize()
            
            logger.info("🔄 Running prediction pipeline...")
            await pipeline.run_daily_pipeline()
            logger.info("✅ Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    asyncio.run(run_pipeline())


@cli.command()
@click.option('--position', help='Train models for specific position (QB, RB, WR, TE)')
@click.option('--force-retrain', is_flag=True, help='Force model retraining')
@click.pass_context
def train(ctx, position: Optional[str], force_retrain: bool):
    """Train the ML models."""
    config = ctx.obj['config']
    
    async def run_training():
        try:
            from ml_models import NFLPredictor, ModelConfig
            from feature_engineering import AdvancedFeatureEngineer
            
            logger.info("🤖 Starting model training...")
            
            # Initialize components
            model_config = ModelConfig(
                model_types=config.models.model_types,
                ensemble_method=config.models.ensemble_method,
                test_size=config.models.test_size,
                random_state=config.models.random_state,
                save_models=config.models.save_models,
                model_directory=config.models.model_directory,
                hyperparameter_tuning=config.models.hyperparameter_tuning
            )
            predictor = NFLPredictor(model_config)
            feature_engineer = AdvancedFeatureEngineer(
                config.database.url, 
                config.features
            )
            
            # Load training data from database
            import pandas as pd
            import joblib
            from database_models import PlayerGameStats, Player
            
            engine = create_engine(config.database.url)
            Session = sessionmaker(bind=engine)
            
            with Session() as session:
                # Get all player stats - use raw SQL for better compatibility
                
                # First, let's get QB stats directly since we know they exist
                if position == 'QB':
                    query = text("""
                        SELECT *, 'QB' as position 
                        FROM player_game_stats 
                        WHERE player_id LIKE '%_qb' 
                        AND fantasy_points_ppr > 0
                        LIMIT 1000
                    """)
                else:
                    query = text("""
                        SELECT pgs.*, 
                               CASE 
                                   WHEN pgs.player_id LIKE '%_qb' THEN 'QB'
                                   WHEN pgs.player_id LIKE '%_rb' THEN 'RB' 
                                   WHEN pgs.player_id LIKE '%_wr' THEN 'WR'
                                   WHEN pgs.player_id LIKE '%_te' THEN 'TE'
                                   ELSE 'UNKNOWN'
                               END as position
                        FROM player_game_stats pgs 
                        WHERE fantasy_points_ppr > 0
                        AND (player_id LIKE '%_qb' OR player_id LIKE '%_rb' 
                             OR player_id LIKE '%_wr' OR player_id LIKE '%_te')
                        LIMIT 1000
                    """)
                
                result = session.execute(query)
                rows = result.fetchall()
                
                if not rows:
                    logger.warning("⚠️  No training data available")
                    return
                
                logger.info(f"Found {len(rows)} training samples")
                
                # Create simple training data
                training_data = []
                for row in rows:
                    pos = row.position
                    if pos == position or position is None:
                        # Create basic features for demonstration
                        features = {
                            'position': pos,
                            'is_home': float(getattr(row, 'is_home', 0) or 0),
                            'week': float(getattr(row, 'week', 1) or 1),
                        }
                        
                        # Add position-specific stats as both features and targets
                        if pos == 'QB':
                            features.update({
                                'passing_attempts': float(getattr(row, 'passing_attempts', 0) or 0),
                                'passing_completions': float(getattr(row, 'passing_completions', 0) or 0),
                                'passing_yards': float(getattr(row, 'passing_yards', 0) or 0),
                                'passing_touchdowns': float(getattr(row, 'passing_touchdowns', 0) or 0),
                                'fantasy_points': float(getattr(row, 'fantasy_points_ppr', 0) or 0)
                            })
                        elif pos in ['RB', 'WR', 'TE']:
                            features.update({
                                'rushing_attempts': float(getattr(row, 'rushing_attempts', 0) or 0),
                                'rushing_yards': float(getattr(row, 'rushing_yards', 0) or 0),
                                'targets': float(getattr(row, 'targets', 0) or 0),
                                'receptions': float(getattr(row, 'receptions', 0) or 0),
                                'receiving_yards': float(getattr(row, 'receiving_yards', 0) or 0),
                                'fantasy_points': float(getattr(row, 'fantasy_points_ppr', 0) or 0)
                            })
                        
                        training_data.append(features)
                
                if len(training_data) < 50:
                    logger.warning(f"⚠️  Insufficient training data: {len(training_data)} samples")
                    return
                
                # Convert to DataFrames
                df = pd.DataFrame(training_data)
                
                # Split features and targets
                target_columns = ['fantasy_points']
                if position == 'QB' or position is None:
                    target_columns.extend(['passing_yards', 'passing_touchdowns'])
                
                feature_columns = [col for col in df.columns if col not in target_columns + ['position']]
                
                positions_to_train = [position] if position else df['position'].unique()
                
                for pos in positions_to_train:
                    pos_data = df[df['position'] == pos]
                    
                    if len(pos_data) < 50:
                        logger.warning(f"⚠️  Insufficient data for {pos}: {len(pos_data)} samples")
                        continue
                    
                    logger.info(f"Training models for {pos} with {len(pos_data)} samples...")
                    
                    X = pos_data[feature_columns].fillna(0)
                    
                    for target in target_columns:
                        if target in pos_data.columns:
                            y = pos_data[target].fillna(0)
                            
                            if y.std() > 0:  # Only train if there's variance
                                try:
                                    # Use sklearn models directly for quick training
                                    from sklearn.ensemble import RandomForestRegressor
                                    from sklearn.model_selection import train_test_split
                                    from sklearn.metrics import r2_score
                                    
                                    # Split data
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.2, random_state=42
                                    )
                                    
                                    # Train model
                                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                                    model.fit(X_train, y_train)
                                    
                                    # Evaluate
                                    y_pred = model.predict(X_test)
                                    r2 = r2_score(y_test, y_pred)
                                    
                                    # Save model
                                    model_path = Path(config.models.model_directory) / f"{pos}_{target}_model.pkl"
                                    model_path.parent.mkdir(parents=True, exist_ok=True)
                                    joblib.dump(model, model_path)
                                    
                                    logger.info(f"✅ {pos} {target}: R² = {r2:.3f} (saved to {model_path})")
                                    
                                except Exception as e:
                                    logger.warning(f"⚠️  Failed to train {pos} {target}: {e}")
                            else:
                                logger.warning(f"⚠️  No variance in {pos} {target} data")
            
            logger.info("🎉 Model training completed!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    asyncio.run(run_training())


@cli.command()
@click.option('--week', type=int, help='Generate predictions for specific week')
@click.option('--player', help='Generate predictions for specific player')
@click.pass_context
def predict(ctx, week: Optional[int], player: Optional[str]):
    """Generate predictions."""
    config = ctx.obj['config']
    
    async def run_predictions():
        try:
            from prediction_pipeline import NFLPredictionPipeline, PipelineConfig
            
            pipeline_config = PipelineConfig(
                database_url=config.database.url,
                **config.pipeline.__dict__
            )
            
            pipeline = NFLPredictionPipeline(pipeline_config)
            await pipeline.initialize()
            
            logger.info("🔮 Generating predictions...")
            
            if player:
                # Generate predictions for specific player
                predictions = await pipeline._generate_player_predictions(player, week)
                logger.info(f"✅ Generated predictions for {player}")
            else:
                # Generate predictions for all eligible players
                await pipeline._generate_predictions()
                logger.info("✅ Generated predictions for all players")
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise
    
    asyncio.run(run_predictions())


@cli.command()
@click.option('--seasons', help='Seasons to download (comma-separated)')
@click.pass_context
def download_data(ctx, seasons: Optional[str]):
    """Download initial NFL data."""
    config = ctx.obj['config']
    
    async def download():
        try:
            from data_collector import NFLDataCollector, DataCollectionConfig
            
            season_list = config.data.seasons
            if seasons:
                season_list = [int(s.strip()) for s in seasons.split(',')]
            
            data_config = DataCollectionConfig(
                database_url=config.database.url,
                api_keys={},  # Empty for now, will use environment variables
                data_sources={"nfl_data_py": True},
                seasons=season_list,
                current_season=config.data.current_season,
                current_week=config.data.current_week,
                enable_live_data=config.data.enable_live_data
            )
            
            collector = NFLDataCollector(data_config)
            await collector.initialize_database()
            
            logger.info(f"📊 Downloading data for seasons: {season_list}")
            await collector.run_full_collection()
            logger.info("✅ Data download completed!")
            
        except Exception as e:
            logger.error(f"❌ Data download failed: {e}")
            raise
    
    asyncio.run(download())


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose test output')
@click.pass_context
def test(ctx, verbose: bool):
    """Run system tests."""
    import subprocess
    
    logger.info("🧪 Running system tests...")
    
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        if result.returncode == 0:
            logger.info("✅ All tests passed!")
        else:
            logger.error("❌ Some tests failed!")
            
    except FileNotFoundError:
        logger.error("❌ pytest not found. Install with: pip install pytest")


@cli.command()
@click.pass_context
def collect(ctx):
    """Collect and update NFL data."""
    logger.info("📥 Starting data collection...")
    
    try:
        from data_collector import NFLDataCollector
        
        collector = NFLDataCollector()
        
        # Collect recent data
        logger.info("Collecting recent NFL data...")
        collector.collect_recent_data()
        
        logger.info("✅ Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        raise click.ClickException(f"Data collection failed: {e}")


@cli.command()
@click.option('--player', help='Analyze sentiment for specific player')
@click.pass_context
def sentiment(ctx, player: Optional[str]):
    """Analyze social sentiment and news impact for players."""
    logger.info("🔍 Running sentiment analysis...")
    
    try:
        from social_sentiment_analyzer import SocialSentimentAnalyzer, NewsImpactAnalyzer
        
        sentiment_analyzer = SocialSentimentAnalyzer()
        news_analyzer = NewsImpactAnalyzer()
        
        if player:
            # Analyze specific player
            logger.info(f"Analyzing sentiment for {player}")
            sentiment_data = sentiment_analyzer.analyze_player_sentiment(player)
            multiplier = sentiment_analyzer.get_sentiment_multiplier(player)
            
            print(f"\nSentiment Analysis for {player}:")
            print(f"  Sentiment Score: {sentiment_data.sentiment_score:.3f}")
            print(f"  Mention Volume: {sentiment_data.mention_volume}")
            print(f"  Positive/Negative/Neutral: {sentiment_data.positive_mentions}/{sentiment_data.negative_mentions}/{sentiment_data.neutral_mentions}")
            print(f"  Trending Topics: {', '.join(sentiment_data.trending_topics)}")
            print(f"  Sentiment Multiplier: {multiplier:.3f}")
            
            # Injury analysis
            injury_analysis = sentiment_analyzer.analyze_injury_sentiment(player)
            print(f"  Injury Concern Level: {injury_analysis['injury_concern_level']}")
        else:
            logger.info("Use --player option to analyze specific player sentiment")
            
        logger.info("✅ Sentiment analysis completed!")
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")


@cli.command()
@click.option('--player', help='Generate ultimate prediction for specific player')
@click.option('--opponent', help='Opponent team for enhanced analysis')
@click.option('--compare', help='Compare multiple players (comma-separated)')
@click.pass_context
def ultimate(ctx, player: Optional[str], opponent: Optional[str], compare: Optional[str]):
    """Generate ultimate enhanced predictions using all analytics."""
    logger.info("🚀 Running ultimate enhanced predictor...")
    
    try:
        from ultimate_enhanced_predictor import UltimateEnhancedPredictor
        
        predictor = UltimateEnhancedPredictor()
        
        if compare:
            # Compare multiple players
            player_list = [p.strip() for p in compare.split(',')]
            logger.info(f"Comparing players: {', '.join(player_list)}")
            
            comparison_df = predictor.compare_multiple_players(player_list)
            if not comparison_df.empty:
                print("\nPlayer Comparison:")
                print(comparison_df.to_string(index=False))
            else:
                logger.warning("No valid comparisons generated")
                
        elif player:
            # Analyze specific player
            logger.info(f"Generating ultimate prediction for {player}")
            predictor.display_ultimate_analysis(player, opponent)
        else:
            logger.info("Use --player or --compare options for ultimate analysis")
            
        logger.info("✅ Ultimate prediction completed!")
        
    except Exception as e:
        logger.error(f"❌ Ultimate prediction failed: {e}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and health check."""
    logger.info("📊 NFL Prediction System Status")
    logger.info("=" * 40)
    
    config = get_config()
    
    # Database connectivity check
    try:
        engine = create_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database: Connected")
    except Exception as e:
        logger.error(f"❌ Database: Connection failed - {e}")
        return
    
    # Data availability check
    try:
        Session = sessionmaker(bind=engine)
        with Session() as session:
            # Fix SQLAlchemy deprecation warning by using func.count with distinct
            from sqlalchemy import func
            player_count = session.query(func.count(func.distinct(PlayerGameStats.player_id))).scalar()
            game_count = session.query(func.count(Game.game_id)).scalar()
            
        logger.info(f"📊 Data: {player_count} players, {game_count} games")
        
    except Exception as e:
        logger.error(f"❌ Data check failed: {e}")
    
    # Check model availability
    model_dir = Path(config.models.model_directory)
    if model_dir.exists():
        model_files = list(model_dir.glob("*.pkl"))
        logger.info(f"🤖 Models: {len(model_files)} trained models found")
    else:
        logger.info("🤖 Models: No trained models found")
    
    logger.info(f"🔧 Configuration: {config.version} ({config.environment})")


if __name__ == "__main__":
    cli()
