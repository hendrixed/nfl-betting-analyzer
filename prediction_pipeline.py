"""
NFL Prediction Pipeline
Daily orchestration system that coordinates data collection, feature engineering, 
and model predictions for NFL player performance and game outcomes.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from sqlalchemy import create_engine, select, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import schedule
import pickle

# Import our modules (canonical)
from core.database_models import (
    Player, Team, Game, PlayerGameStats, BettingLine, 
    PlayerPrediction, GamePrediction, FeatureStore, ModelPerformance
)
from core.data.data_collector import NFLDataCollector, DataCollectionConfig
from core.models.feature_engineering import AdvancedFeatureEngineer, FeatureConfig
from ml_models import NFLPredictor, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_predictions.log'),
        logging.StreamHandler()
    ]
)


@dataclass 
class PipelineConfig:
    """Configuration for the prediction pipeline."""
    database_url: str
    
    # Data collection settings
    data_collection_enabled: bool = True
    live_data_update_interval: int = 300  # seconds
    
    # Feature engineering settings
    feature_engineering_enabled: bool = True
    feature_version: str = "v1.0"
    lookback_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    
    # Model settings
    model_retraining_enabled: bool = True
    model_retraining_frequency: str = "weekly"  # daily, weekly, monthly
    ensemble_method: str = "weighted_average"
    
    # Prediction settings
    prediction_horizon_days: int = 7
    confidence_threshold: float = 0.6
    
    # Performance settings
    max_workers: int = 4
    batch_size: int = 100
    
    # Scheduling
    daily_run_time: str = "08:00"  # UTC time
    enable_scheduler: bool = True


class NFLPredictionPipeline:
    """Main prediction pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the prediction pipeline."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.engine = create_engine(config.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Component initialization
        self.data_collector = None
        self.feature_engineer = None
        self.predictor = None
        
        # State tracking
        self.last_data_update = None
        self.last_model_training = None
        self.last_prediction_run = None
        
        # Performance metrics
        self.pipeline_metrics = {
            'data_collection_time': 0,
            'feature_engineering_time': 0,
            'prediction_time': 0,
            'total_predictions_made': 0,
            'average_confidence': 0,
            'errors': []
        }
        
    async def initialize(self):
        """Initialize all pipeline components."""
        try:
            self.logger.info("Initializing prediction pipeline...")
            
            # Initialize data collector
            if self.config.data_collection_enabled:
                data_config = DataCollectionConfig(
                    database_url=self.config.database_url,
                    api_keys={},  # Would be loaded from environment
                    data_sources={'nfl_data_py': True},
                    seasons=[2020, 2021, 2022, 2023, 2024],
                    current_season=2024,
                    current_week=self._get_current_week(),
                    enable_live_data=True
                )
                self.data_collector = NFLDataCollector(data_config)
                await self.data_collector.initialize_database()
                
            # Initialize feature engineer
            if self.config.feature_engineering_enabled:
                feature_config = FeatureConfig(
                    lookback_windows=self.config.lookback_windows,
                    rolling_windows=[4, 8],
                    min_games_threshold=3,
                    feature_version=self.config.feature_version,
                    scale_features=True,
                    handle_missing="impute"
                )
                self.feature_engineer = AdvancedFeatureEngineer(
                    self.config.database_url, 
                    feature_config
                )
                
            # Initialize ML predictor
            model_config = ModelConfig(
                model_types=['xgboost', 'lightgbm', 'random_forest'],
                ensemble_method=self.config.ensemble_method,
                hyperparameter_tuning=True,
                save_models=True
            )
            self.predictor = NFLPredictor(model_config)
            
            self.logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            raise
            
    async def run_daily_pipeline(self):
        """Run the complete daily prediction pipeline."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting daily prediction pipeline...")
            
            # Step 1: Update data
            if self.config.data_collection_enabled:
                await self._update_data()
                
            # Step 2: Engineer features for upcoming games
            if self.config.feature_engineering_enabled:
                await self._engineer_features()
                
            # Step 3: Retrain models if needed
            if self.config.model_retraining_enabled:
                await self._retrain_models_if_needed()
                
            # Step 4: Generate predictions
            await self._generate_predictions()
            
            # Step 5: Update performance metrics
            await self._update_pipeline_metrics()
            
            total_time = time.time() - start_time
            self.logger.info(f"Daily pipeline completed in {total_time:.2f} seconds")
            
            self.last_prediction_run = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Daily pipeline failed: {e}")
            self.pipeline_metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'pipeline_step': 'daily_run'
            })
            raise
            
    async def _update_data(self):
        """Update data from all sources."""
        start_time = time.time()
        
        try:
            self.logger.info("Updating data...")
            
            # Collect current season data
            await self.data_collector.collect_current_season()
            
            # Collect live data (injuries, line movements, etc.)
            await self.data_collector.collect_live_data()
            
            self.last_data_update = datetime.now()
            self.pipeline_metrics['data_collection_time'] = time.time() - start_time
            
            self.logger.info("Data update completed")
            
        except Exception as e:
            self.logger.error(f"Data update failed: {e}")
            raise
            
    async def _engineer_features(self):
        """Engineer features for upcoming games."""
        start_time = time.time()
        
        try:
            self.logger.info("Engineering features for upcoming games...")
            
            # Get upcoming games
            upcoming_games = self._get_upcoming_games()
            
            # Get active players
            active_players = self._get_active_players()
            
            # Engineer features in parallel
            feature_tasks = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                for game in upcoming_games:
                    for player in active_players:
                        if self._should_engineer_features(player, game):
                            task = executor.submit(
                                self._engineer_player_features,
                                player['player_id'],
                                game['game_id'],
                                player['position']
                            )
                            feature_tasks.append(task)
                            
                # Wait for completion
                for task in as_completed(feature_tasks):
                    try:
                        task.result()
                    except Exception as e:
                        self.logger.warning(f"Feature engineering task failed: {e}")
                        
            self.pipeline_metrics['feature_engineering_time'] = time.time() - start_time
            self.logger.info("Feature engineering completed")
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise
            
    async def _retrain_models_if_needed(self):
        """Retrain models based on frequency settings."""
        
        try:
            if not self._should_retrain_models():
                self.logger.info("Model retraining not needed")
                return
                
            self.logger.info("Starting model retraining...")
            
            # Get training data
            training_data = await self._prepare_training_data()
            
            # Retrain models for each position
            positions = ['QB', 'RB', 'WR', 'TE']
            
            for position in positions:
                self.logger.info(f"Retraining models for {position}")
                
                # Get position-specific data
                position_features = training_data['features'][
                    training_data['features']['position'] == position
                ]
                position_targets = training_data['targets'][
                    training_data['targets']['position'] == position
                ]
                
                if len(position_features) < 100:  # Minimum data requirement
                    self.logger.warning(f"Insufficient data for {position}: {len(position_features)} samples")
                    continue
                    
                # Train models
                results = self.predictor.train_models(
                    position_features,
                    position_targets,
                    position
                )
                
                # Log results
                for target, result in results.items():
                    self.logger.info(
                        f"{position} {target}: Best model {result['best_model']}, "
                        f"Score: {result['best_score']:.3f}"
                    )
                    
                # Save performance metrics
                await self._save_model_performance(position, results)
                
            self.last_model_training = datetime.now()
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            raise
            
    async def _generate_predictions(self):
        """Generate predictions for upcoming games."""
        start_time = time.time()
        
        try:
            self.logger.info("Generating predictions...")
            
            # Get upcoming games
            upcoming_games = self._get_upcoming_games()
            
            prediction_count = 0
            confidence_scores = []
            
            # Generate predictions for each game
            for game in upcoming_games:
                # Get players for this game
                game_players = self._get_game_players(game)
                
                for player in game_players:
                    try:
                        # Load features
                        features = self.feature_engineer.load_features_from_store(
                            player['player_id'],
                            game['game_id']
                        )
                        
                        if not features:
                            # Engineer features if not available
                            features = self.feature_engineer.engineer_player_features(
                                player['player_id'],
                                game['game_id'],
                                player['position']
                            )
                            
                        if not features:
                            self.logger.warning(f"No features available for {player['player_id']}")
                            continue
                            
                        # Generate predictions
                        predictions = self.predictor.predict(
                            features,
                            player['position'],
                            'ensemble'
                        )
                        
                        # Save predictions
                        await self._save_player_predictions(
                            player['player_id'],
                            game['game_id'],
                            predictions
                        )
                        
                        prediction_count += 1
                        
                        # Track confidence scores
                        for pred in predictions.values():
                            confidence_scores.append(pred['confidence'])
                            
                    except Exception as e:
                        self.logger.warning(f"Prediction failed for {player['player_id']}: {e}")
                        continue
                        
            # Generate game-level predictions
            await self._generate_game_predictions(upcoming_games)
            
            self.pipeline_metrics['prediction_time'] = time.time() - start_time
            self.pipeline_metrics['total_predictions_made'] = prediction_count
            
            if confidence_scores:
                self.pipeline_metrics['average_confidence'] = np.mean(confidence_scores)
                
            self.logger.info(f"Generated {prediction_count} player predictions")
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            raise
            
    async def _generate_game_predictions(self, games: List[Dict]):
        """Generate game-level outcome predictions."""
        
        try:
            self.logger.info("Generating game outcome predictions...")
            
            for game in games:
                # Aggregate player predictions for each team
                home_team_total = await self._aggregate_team_predictions(
                    game['home_team'], 
                    game['game_id']
                )
                away_team_total = await self._aggregate_team_predictions(
                    game['away_team'], 
                    game['game_id']
                )
                
                # Simple game prediction based on team totals
                predicted_home_score = home_team_total.get('total_fantasy_points', 0) * 1.5
                predicted_away_score = away_team_total.get('total_fantasy_points', 0) * 1.5
                
                # Calculate spread and totals
                predicted_spread = predicted_home_score - predicted_away_score
                predicted_total = predicted_home_score + predicted_away_score
                
                # Calculate win probabilities
                score_diff = predicted_home_score - predicted_away_score
                home_win_prob = 1 / (1 + np.exp(-score_diff / 7))  # Sigmoid function
                away_win_prob = 1 - home_win_prob
                
                # Save game prediction
                await self._save_game_prediction(
                    game['game_id'],
                    predicted_home_score,
                    predicted_away_score,
                    predicted_total,
                    predicted_spread,
                    home_win_prob,
                    away_win_prob
                )
                
        except Exception as e:
            self.logger.error(f"Game prediction generation failed: {e}")
            
    # Helper methods
    def _get_current_week(self) -> int:
        """Get current NFL week."""
        # Simplified logic - would need proper NFL calendar
        today = date.today()
        season_start = date(2024, 9, 5)  # Example season start
        days_since_start = (today - season_start).days
        week = min(18, max(1, days_since_start // 7 + 1))
        return week
        
    def _get_upcoming_games(self) -> List[Dict]:
        """Get upcoming games within prediction horizon."""
        
        with self.Session() as session:
            end_date = date.today() + timedelta(days=self.config.prediction_horizon_days)
            
            games = session.query(Game).filter(
                and_(
                    Game.game_date >= date.today(),
                    Game.game_date <= end_date,
                    Game.game_status == 'scheduled'
                )
            ).all()
            
            return [
                {
                    'game_id': game.game_id,
                    'game_date': game.game_date,
                    'home_team': game.home_team,
                    'away_team': game.away_team,
                    'week': game.week,
                    'season': game.season
                }
                for game in games
            ]
            
    def _get_active_players(self) -> List[Dict]:
        """Get active players."""
        
        with self.Session() as session:
            players = session.query(Player).filter(
                Player.is_active == True
            ).all()
            
            return [
                {
                    'player_id': player.player_id,
                    'position': player.position,
                    'current_team': player.current_team
                }
                for player in players
            ]
            
    def _get_game_players(self, game: Dict) -> List[Dict]:
        """Get players participating in a specific game."""
        
        with self.Session() as session:
            # Get players from both teams
            home_players = session.query(Player).filter(
                and_(
                    Player.current_team == game['home_team'],
                    Player.is_active == True
                )
            ).all()
            
            away_players = session.query(Player).filter(
                and_(
                    Player.current_team == game['away_team'],
                    Player.is_active == True
                )
            ).all()
            
            all_players = home_players + away_players
            
            return [
                {
                    'player_id': player.player_id,
                    'position': player.position,
                    'current_team': player.current_team
                }
                for player in all_players
                if player.position in ['QB', 'RB', 'WR', 'TE']  # Focus on skill positions
            ]
            
    def _should_engineer_features(self, player: Dict, game: Dict) -> bool:
        """Determine if features should be engineered for a player-game combination."""
        
        # Only engineer for players on teams playing in the game
        return player['current_team'] in [game['home_team'], game['away_team']]
        
    def _engineer_player_features(self, player_id: str, game_id: str, position: str):
        """Engineer features for a specific player-game combination."""
        
        try:
            features = self.feature_engineer.engineer_player_features(
                player_id, game_id, position
            )
            
            if features:
                self.feature_engineer.save_features_to_store(
                    player_id, game_id, features
                )
                
        except Exception as e:
            self.logger.warning(f"Feature engineering failed for {player_id}: {e}")
            
    def _should_retrain_models(self) -> bool:
        """Determine if models should be retrained."""
        
        if not self.last_model_training:
            return True
            
        if self.config.model_retraining_frequency == 'daily':
            return (datetime.now() - self.last_model_training).days >= 1
        elif self.config.model_retraining_frequency == 'weekly':
            return (datetime.now() - self.last_model_training).days >= 7
        elif self.config.model_retraining_frequency == 'monthly':
            return (datetime.now() - self.last_model_training).days >= 30
            
        return False
        
    async def _prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare training data for model retraining."""
        
        with self.Session() as session:
            # Get recent game stats (last 2 seasons)
            cutoff_date = date.today() - timedelta(days=730)
            
            # Features from feature store
            features_query = session.query(FeatureStore).join(
                Game, FeatureStore.game_id == Game.game_id
            ).filter(Game.game_date >= cutoff_date)
            
            features_data = []
            for feature_store in features_query.all():
                # Combine all feature categories
                features = {}
                for category in ['recent_form_features', 'seasonal_features', 
                               'opponent_features', 'contextual_features', 'advanced_features']:
                    category_features = getattr(feature_store, category, {})
                    if category_features:
                        features.update(category_features)
                        
                features['player_id'] = feature_store.player_id
                features['game_id'] = feature_store.game_id
                features_data.append(features)
                
            features_df = pd.DataFrame(features_data)
            
            # Targets from player game stats
            stats_query = session.query(PlayerGameStats).join(
                Game, PlayerGameStats.game_id == Game.game_id
            ).filter(Game.game_date >= cutoff_date)
            
            targets_data = []
            for stats in stats_query.all():
                targets_data.append({
                    'player_id': stats.player_id,
                    'game_id': stats.game_id,
                    'position': self._get_player_position(session, stats.player_id),
                    'passing_yards': stats.passing_yards,
                    'passing_touchdowns': stats.passing_touchdowns,
                    'passing_interceptions': stats.passing_interceptions,
                    'rushing_yards': stats.rushing_yards,
                    'rushing_touchdowns': stats.rushing_touchdowns,
                    'receptions': stats.receptions,
                    'receiving_yards': stats.receiving_yards,
                    'receiving_touchdowns': stats.receiving_touchdowns,
                    'targets': stats.targets,
                    'fantasy_points_ppr': stats.fantasy_points_ppr
                })
                
            targets_df = pd.DataFrame(targets_data)
            
            # Add position info to features
            features_df = pd.merge(
                features_df,
                targets_df[['player_id', 'game_id', 'position']],
                on=['player_id', 'game_id'],
                how='left'
            )
            
            return {
                'features': features_df,
                'targets': targets_df
            }
            
    def _get_player_position(self, session: Session, player_id: str) -> str:
        """Get player position."""
        player = session.query(Player).filter(Player.player_id == player_id).first()
        return player.position if player else 'Unknown'
        
    async def _save_player_predictions(
        self,
        player_id: str,
        game_id: str,
        predictions: Dict[str, Any]
    ):
        """Save player predictions to database."""
        
        try:
            with self.Session() as session:
                # Calculate overall confidence
                confidences = [pred['confidence'] for pred in predictions.values()]
                overall_confidence = np.mean(confidences) if confidences else 0.7
                
                # Create prediction record
                prediction = PlayerPrediction(
                    player_id=player_id,
                    game_id=game_id,
                    model_version="v1.0",
                    model_type="ensemble",
                    predicted_passing_yards=predictions.get('passing_yards', {}).get('predicted_value'),
                    predicted_passing_tds=predictions.get('passing_tds', {}).get('predicted_value'),
                    predicted_rushing_yards=predictions.get('rushing_yards', {}).get('predicted_value'),
                    predicted_rushing_tds=predictions.get('rushing_tds', {}).get('predicted_value'),
                    predicted_receiving_yards=predictions.get('receiving_yards', {}).get('predicted_value'),
                    predicted_receiving_tds=predictions.get('receiving_tds', {}).get('predicted_value'),
                    predicted_receptions=predictions.get('receptions', {}).get('predicted_value'),
                    predicted_fantasy_points=predictions.get('fantasy_points', {}).get('predicted_value'),
                    confidence_overall=overall_confidence,
                    prediction_timestamp=datetime.now()
                )
                
                session.merge(prediction)  # Use merge to handle duplicates
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving player predictions: {e}")
            
    async def _save_game_prediction(
        self,
        game_id: str,
        home_score: float,
        away_score: float,
        total_points: float,
        spread: float,
        home_win_prob: float,
        away_win_prob: float
    ):
        """Save game prediction to database."""
        
        try:
            with self.Session() as session:
                prediction = GamePrediction(
                    game_id=game_id,
                    model_version="v1.0",
                    model_type="ensemble",
                    predicted_home_score=home_score,
                    predicted_away_score=away_score,
                    predicted_total_points=total_points,
                    predicted_spread=spread,
                    home_win_probability=home_win_prob,
                    away_win_probability=away_win_prob,
                    confidence_score=0.75,  # Would calculate properly
                    prediction_timestamp=datetime.now()
                )
                
                session.merge(prediction)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving game prediction: {e}")
            
    async def _aggregate_team_predictions(self, team: str, game_id: str) -> Dict[str, float]:
        """Aggregate player predictions for a team."""
        
        with self.Session() as session:
            # Get player predictions for this team and game
            players_query = session.query(Player).filter(
                Player.current_team == team
            )
            
            team_totals = {
                'total_fantasy_points': 0,
                'total_passing_yards': 0,
                'total_rushing_yards': 0,
                'total_receiving_yards': 0,
                'player_count': 0
            }
            
            for player in players_query.all():
                prediction = session.query(PlayerPrediction).filter(
                    and_(
                        PlayerPrediction.player_id == player.player_id,
                        PlayerPrediction.game_id == game_id
                    )
                ).first()
                
                if prediction:
                    team_totals['total_fantasy_points'] += prediction.predicted_fantasy_points or 0
                    team_totals['total_passing_yards'] += prediction.predicted_passing_yards or 0
                    team_totals['total_rushing_yards'] += prediction.predicted_rushing_yards or 0
                    team_totals['total_receiving_yards'] += prediction.predicted_receiving_yards or 0
                    team_totals['player_count'] += 1
                    
            return team_totals
            
    async def _save_model_performance(self, position: str, results: Dict[str, Any]):
        """Save model performance metrics."""
        
        try:
            with self.Session() as session:
                performance = ModelPerformance(
                    model_version="v1.0",
                    model_type="ensemble",
                    evaluation_period=f"{position}_retrain_{datetime.now().strftime('%Y%m%d')}",
                    accuracy_metrics=results,
                    total_predictions=sum(r.get('n_samples', 0) for r in results.values()),
                    evaluation_start_date=date.today() - timedelta(days=730),
                    evaluation_end_date=date.today()
                )
                
                session.add(performance)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving model performance: {e}")
            
    async def _update_pipeline_metrics(self):
        """Update pipeline performance metrics."""
        
        # Save metrics to file
        metrics_file = Path("pipeline_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.pipeline_metrics, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving pipeline metrics: {e}")
            
    # Scheduling methods
    def start_scheduler(self):
        """Start the prediction pipeline scheduler."""
        
        if not self.config.enable_scheduler:
            self.logger.info("Scheduler disabled")
            return
            
        self.logger.info(f"Starting scheduler - daily run at {self.config.daily_run_time}")
        
        # Schedule daily run
        schedule.every().day.at(self.config.daily_run_time).do(
            lambda: asyncio.run(self.run_daily_pipeline())
        )
        
        # Schedule live data updates
        schedule.every(self.config.live_data_update_interval).seconds.do(
            lambda: asyncio.run(self._update_live_data())
        )
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    async def _update_live_data(self):
        """Update live data during the day."""
        
        try:
            if self.data_collector:
                await self.data_collector.collect_live_data()
                
        except Exception as e:
            self.logger.warning(f"Live data update failed: {e}")


# Main execution
async def main():
    """Main function to run the prediction pipeline."""
    
    # Configuration
    config = PipelineConfig(
        database_url="postgresql://user:password@localhost/nfl_predictions",
        data_collection_enabled=True,
        feature_engineering_enabled=True,
        model_retraining_enabled=True,
        prediction_horizon_days=7,
        max_workers=4,
        enable_scheduler=True,
        daily_run_time="08:00"
    )
    
    # Initialize pipeline
    pipeline = NFLPredictionPipeline(config)
    await pipeline.initialize()
    
    # Run once manually for testing
    await pipeline.run_daily_pipeline()
    
    # Start scheduler for continuous operation
    # pipeline.start_scheduler()  # Uncomment for production


if __name__ == "__main__":
    asyncio.run(main())