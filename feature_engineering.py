"""
Advanced Feature Engineering Pipeline for NFL Player Performance Prediction
Creates comprehensive features from raw game data for machine learning models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from sqlalchemy import create_engine, select, and_, or_, desc, asc
from sqlalchemy.orm import sessionmaker, Session
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# Import our models
from database_models import (
    Player, Team, Game, PlayerGameStats, BettingLine, FeatureStore
)

# Configure logging
logging.basicConfig(level=logging.INFO)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    lookback_windows: List[int]
    rolling_windows: List[int]
    min_games_threshold: int
    feature_version: str
    scale_features: bool
    handle_missing: str  # 'drop', 'impute', or 'flag'
    

class AdvancedFeatureEngineer:
    """Advanced feature engineering for NFL player predictions."""
    
    def __init__(self, database_url: str, config: FeatureConfig):
        """Initialize the feature engineer."""
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Scalers for different feature types
        self.scalers = {
            'numerical': StandardScaler(),
            'rate': MinMaxScaler(),
            'categorical': LabelEncoder()
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Position-specific configurations
        self.position_configs = self._initialize_position_configs()
        
    def _initialize_position_configs(self) -> Dict[str, Dict]:
        """Initialize position-specific feature configurations."""
        return {
            'QB': {
                'primary_stats': ['passing_yards', 'passing_touchdowns', 'interceptions', 'rushing_yards'],
                'efficiency_metrics': ['completion_percentage', 'yards_per_attempt', 'passer_rating'],
                'context_factors': ['pressure_rate', 'time_to_throw', 'red_zone_attempts'],
                'target_features': ['passing_yards', 'passing_touchdowns', 'interceptions', 'rushing_yards']
            },
            'RB': {
                'primary_stats': ['rushing_yards', 'rushing_touchdowns', 'receptions', 'receiving_yards'],
                'efficiency_metrics': ['yards_per_carry', 'yards_per_reception', 'broken_tackle_rate'],
                'context_factors': ['carry_share', 'red_zone_carries', 'goal_line_carries'],
                'target_features': ['rushing_yards', 'rushing_touchdowns', 'receptions', 'receiving_yards']
            },
            'WR': {
                'primary_stats': ['receptions', 'receiving_yards', 'receiving_touchdowns', 'targets'],
                'efficiency_metrics': ['catch_rate', 'yards_per_reception', 'yards_per_target'],
                'context_factors': ['target_share', 'air_yards_share', 'red_zone_targets'],
                'target_features': ['receptions', 'receiving_yards', 'receiving_touchdowns', 'targets']
            },
            'TE': {
                'primary_stats': ['receptions', 'receiving_yards', 'receiving_touchdowns', 'targets'],
                'efficiency_metrics': ['catch_rate', 'yards_per_reception', 'yards_after_catch'],
                'context_factors': ['target_share', 'blocking_snaps', 'route_diversity'],
                'target_features': ['receptions', 'receiving_yards', 'receiving_touchdowns', 'targets']
            }
        }
        
    def engineer_player_features(
        self,
        player_id: str,
        target_game_id: str,
        position: str
    ) -> Dict[str, Any]:
        """Engineer comprehensive features for a specific player and game."""
        
        try:
            self.logger.info(f"Engineering features for player {player_id}, game {target_game_id}")
            
            # Get historical data
            historical_data = self._get_historical_data(player_id, target_game_id)
            
            if historical_data.empty:
                self.logger.warning(f"No historical data found for player {player_id}")
                return {}
                
            # Get game context
            game_context = self._get_game_context(target_game_id)
            
            # Engineer different types of features
            features = {}
            
            # 1. Recent Performance Features
            features.update(self._create_recent_performance_features(historical_data, position))
            
            # 2. Seasonal Trend Features
            features.update(self._create_seasonal_trend_features(historical_data, position))
            
            # 3. Opponent-Adjusted Features
            features.update(self._create_opponent_adjusted_features(historical_data, game_context, position))
            
            # 4. Situational Features
            features.update(self._create_situational_features(historical_data, game_context, position))
            
            # 5. Consistency Features
            features.update(self._create_consistency_features(historical_data, position))
            
            # 6. Advanced Analytics Features
            features.update(self._create_advanced_features(historical_data, position))
            
            # 7. Context Features
            features.update(self._create_context_features(game_context))
            
            # 8. Momentum Features
            features.update(self._create_momentum_features(historical_data, position))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering features for {player_id}: {e}")
            return {}
            
    def _get_historical_data(self, player_id: str, target_game_id: str) -> pd.DataFrame:
        """Get historical game data for a player up to target game."""
        
        with self.Session() as session:
            # Get target game date
            target_game = session.query(Game).filter(Game.game_id == target_game_id).first()
            if not target_game:
                return pd.DataFrame()
                
            # Get historical stats before target game
            query = session.query(
                PlayerGameStats,
                Game.game_date,
                Game.home_team,
                Game.away_team,
                Game.weather_temperature,
                Game.weather_wind_speed
            ).join(
                Game, PlayerGameStats.game_id == Game.game_id
            ).filter(
                and_(
                    PlayerGameStats.player_id == player_id,
                    Game.game_date < target_game.game_date
                )
            ).order_by(desc(Game.game_date))
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
                
            # Convert to DataFrame
            data = []
            for stats, game_date, home_team, away_team, temp, wind in results:
                row = {
                    'game_date': game_date,
                    'team': stats.team,
                    'opponent': stats.opponent,
                    'is_home': stats.is_home,
                    'passing_attempts': stats.passing_attempts,
                    'passing_completions': stats.passing_completions,
                    'passing_yards': stats.passing_yards,
                    'passing_touchdowns': stats.passing_touchdowns,
                    'passing_interceptions': stats.passing_interceptions,
                    'rushing_attempts': stats.rushing_attempts,
                    'rushing_yards': stats.rushing_yards,
                    'rushing_touchdowns': stats.rushing_touchdowns,
                    'targets': stats.targets,
                    'receptions': stats.receptions,
                    'receiving_yards': stats.receiving_yards,
                    'receiving_touchdowns': stats.receiving_touchdowns,
                    'fantasy_points_ppr': stats.fantasy_points_ppr,
                    'fantasy_points_standard': stats.fantasy_points_standard,
                    'weather_temperature': temp,
                    'weather_wind_speed': wind
                }
                data.append(row)
                
            return pd.DataFrame(data)
            
    def _get_game_context(self, game_id: str) -> Dict[str, Any]:
        """Get context information for the target game."""
        
        with self.Session() as session:
            game = session.query(Game).filter(Game.game_id == game_id).first()
            
            if not game:
                return {}
                
            return {
                'game_id': game.game_id,
                'season': game.season,
                'week': game.week,
                'game_type': game.game_type,
                'home_team': game.home_team,
                'away_team': game.away_team,
                'stadium': game.stadium,
                'weather_temperature': game.weather_temperature,
                'weather_humidity': game.weather_humidity,
                'weather_wind_speed': game.weather_wind_speed,
                'weather_conditions': game.weather_conditions,
                'game_date': game.game_date
            }
            
    def _create_recent_performance_features(self, df: pd.DataFrame, position: str) -> Dict[str, float]:
        """Create features based on recent performance."""
        features = {}
        
        position_stats = self.position_configs[position]['primary_stats']
        
        for window in self.config.lookback_windows:
            if len(df) < window:
                continue
                
            recent_df = df.head(window)
            prefix = f"last_{window}_games"
            
            for stat in position_stats:
                if stat in recent_df.columns:
                    values = recent_df[stat].values
                    
                    # Basic statistics
                    features[f"{prefix}_{stat}_mean"] = float(np.mean(values))
                    features[f"{prefix}_{stat}_std"] = float(np.std(values))
                    features[f"{prefix}_{stat}_max"] = float(np.max(values))
                    features[f"{prefix}_{stat}_min"] = float(np.min(values))
                    features[f"{prefix}_{stat}_sum"] = float(np.sum(values))
                    
                    # Trend features
                    if len(values) > 1:
                        features[f"{prefix}_{stat}_trend"] = self._calculate_trend(values)
                        features[f"{prefix}_{stat}_consistency"] = self._calculate_consistency(values)
                        
            # Game-level features
            features[f"{prefix}_fantasy_points_mean"] = float(recent_df['fantasy_points_ppr'].mean())
            features[f"{prefix}_fantasy_points_std"] = float(recent_df['fantasy_points_ppr'].std())
            features[f"{prefix}_home_games"] = float(recent_df['is_home'].sum())
            
        return features
        
    def _create_seasonal_trend_features(self, df: pd.DataFrame, position: str) -> Dict[str, float]:
        """Create seasonal trend and progression features."""
        features = {}
        
        if len(df) < 4:  # Need minimum games for trends
            return features
            
        position_stats = self.position_configs[position]['primary_stats']
        
        for stat in position_stats:
            if stat in df.columns:
                values = df[stat].values
                
                # Overall season trend
                features[f"season_{stat}_trend"] = self._calculate_trend(values)
                
                # Early vs late season comparison
                mid_point = len(values) // 2
                early_season = values[-mid_point:]  # More recent is earlier in reverse order
                late_season = values[:mid_point]
                
                if len(early_season) > 0 and len(late_season) > 0:
                    features[f"{stat}_early_vs_late_improvement"] = float(
                        np.mean(late_season) - np.mean(early_season)
                    )
                    
        # Weekly progression features
        if len(df) >= 3:
            recent_fantasy = df.head(3)['fantasy_points_ppr'].values
            features['recent_fantasy_progression'] = self._calculate_trend(recent_fantasy)
            
        return features
        
    def _create_opponent_adjusted_features(
        self,
        df: pd.DataFrame,
        game_context: Dict[str, Any],
        position: str
    ) -> Dict[str, float]:
        """Create features adjusted for opponent strength."""
        features = {}
        
        target_opponent = game_context.get('away_team') if game_context.get('home_team') else game_context.get('home_team')
        
        if not target_opponent:
            return features
            
        # Get performance against this specific opponent
        opponent_games = df[df['opponent'] == target_opponent]
        
        if len(opponent_games) > 0:
            position_stats = self.position_configs[position]['primary_stats']
            
            for stat in position_stats:
                if stat in opponent_games.columns:
                    features[f"vs_{target_opponent}_{stat}_avg"] = float(opponent_games[stat].mean())
                    
            features[f"vs_{target_opponent}_games_played"] = float(len(opponent_games))
            features[f"vs_{target_opponent}_fantasy_avg"] = float(opponent_games['fantasy_points_ppr'].mean())
            
        # Division opponent features
        division_opponents = self._get_division_teams(game_context.get('home_team', ''))
        if target_opponent in division_opponents:
            division_games = df[df['opponent'].isin(division_opponents)]
            
            if len(division_games) > 0:
                features['vs_division_fantasy_avg'] = float(division_games['fantasy_points_ppr'].mean())
                features['vs_division_games'] = float(len(division_games))
                
        return features
        
    def _create_situational_features(
        self,
        df: pd.DataFrame,
        game_context: Dict[str, Any],
        position: str
    ) -> Dict[str, float]:
        """Create situational and contextual features."""
        features = {}
        
        # Home/Away splits
        home_games = df[df['is_home'] == True]
        away_games = df[df['is_home'] == False]
        
        target_is_home = game_context.get('home_team') is not None
        
        if target_is_home and len(home_games) > 0:
            features['home_fantasy_avg'] = float(home_games['fantasy_points_ppr'].mean())
            features['home_games_played'] = float(len(home_games))
        elif not target_is_home and len(away_games) > 0:
            features['away_fantasy_avg'] = float(away_games['fantasy_points_ppr'].mean())
            features['away_games_played'] = float(len(away_games))
            
        # Weather-based features
        if 'weather_temperature' in df.columns:
            target_temp = game_context.get('weather_temperature')
            if target_temp is not None:
                # Performance in similar weather
                temp_range = 10  # +/- 10 degrees
                similar_weather = df[
                    abs(df['weather_temperature'] - target_temp) <= temp_range
                ]
                
                if len(similar_weather) > 0:
                    features['similar_weather_fantasy_avg'] = float(
                        similar_weather['fantasy_points_ppr'].mean()
                    )
                    
        # Game type features (regular season vs playoffs)
        game_type = game_context.get('game_type', 'REG')
        features['is_playoff_game'] = float(game_type != 'REG')
        
        return features
        
    def _create_consistency_features(self, df: pd.DataFrame, position: str) -> Dict[str, float]:
        """Create consistency and volatility features."""
        features = {}
        
        if len(df) < 3:
            return features
            
        position_stats = self.position_configs[position]['primary_stats']
        
        for stat in position_stats:
            if stat in df.columns:
                values = df[stat].values
                
                # Coefficient of variation
                if np.mean(values) > 0:
                    features[f"{stat}_coefficient_variation"] = float(np.std(values) / np.mean(values))
                    
                # Boom/bust rate (games significantly above/below average)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                boom_threshold = mean_val + std_val
                bust_threshold = mean_val - std_val
                
                boom_games = np.sum(values > boom_threshold)
                bust_games = np.sum(values < bust_threshold)
                
                features[f"{stat}_boom_rate"] = float(boom_games / len(values))
                features[f"{stat}_bust_rate"] = float(bust_games / len(values))
                
        # Fantasy points consistency
        fantasy_values = df['fantasy_points_ppr'].values
        features['fantasy_points_consistency'] = self._calculate_consistency(fantasy_values)
        
        # Floor and ceiling
        features['fantasy_floor_10th_percentile'] = float(np.percentile(fantasy_values, 10))
        features['fantasy_ceiling_90th_percentile'] = float(np.percentile(fantasy_values, 90))
        
        return features
        
    def _create_advanced_features(self, df: pd.DataFrame, position: str) -> Dict[str, float]:
        """Create advanced analytical features."""
        features = {}
        
        if len(df) < 3:
            return features
            
        # Efficiency metrics specific to position
        if position == 'QB':
            features.update(self._create_qb_efficiency_features(df))
        elif position == 'RB':
            features.update(self._create_rb_efficiency_features(df))
        elif position in ['WR', 'TE']:
            features.update(self._create_receiver_efficiency_features(df))
            
        # Usage patterns
        features.update(self._create_usage_features(df, position))
        
        # Performance patterns
        features.update(self._create_performance_patterns(df))
        
        return features
        
    def _create_context_features(self, game_context: Dict[str, Any]) -> Dict[str, float]:
        """Create features from game context."""
        features = {}
        
        # Time-based features
        game_date = game_context.get('game_date')
        if game_date:
            features['week'] = float(game_context.get('week', 0))
            features['is_early_season'] = float(game_context.get('week', 0) <= 4)
            features['is_late_season'] = float(game_context.get('week', 0) >= 14)
            
        # Weather features
        temp = game_context.get('weather_temperature')
        if temp is not None:
            features['temperature'] = float(temp)
            features['is_cold_weather'] = float(temp < 40)
            features['is_hot_weather'] = float(temp > 80)
            
        wind = game_context.get('weather_wind_speed')
        if wind is not None:
            features['wind_speed'] = float(wind)
            features['is_windy'] = float(wind > 15)
            
        # Stadium features
        stadium = game_context.get('stadium', '')
        features['is_dome'] = float(self._is_dome_stadium(stadium))
        
        return features
        
    def _create_momentum_features(self, df: pd.DataFrame, position: str) -> Dict[str, float]:
        """Create momentum and hot/cold streak features."""
        features = {}
        
        if len(df) < 3:
            return features
            
        # Recent momentum (last 3 games vs previous 3 games)
        if len(df) >= 6:
            recent_3 = df.head(3)['fantasy_points_ppr'].mean()
            previous_3 = df.iloc[3:6]['fantasy_points_ppr'].mean()
            
            features['momentum_recent_vs_previous'] = float(recent_3 - previous_3)
            features['momentum_ratio'] = float(recent_3 / previous_3) if previous_3 > 0 else 0.0
            
        # Streak features
        fantasy_values = df['fantasy_points_ppr'].values
        mean_fantasy = np.mean(fantasy_values)
        
        # Current streak (above or below average)
        current_streak = 0
        streak_type = 'neutral'
        
        for value in fantasy_values:
            if value > mean_fantasy:
                if streak_type == 'positive' or streak_type == 'neutral':
                    current_streak += 1
                    streak_type = 'positive'
                else:
                    break
            elif value < mean_fantasy:
                if streak_type == 'negative' or streak_type == 'neutral':
                    current_streak += 1
                    streak_type = 'negative'
                else:
                    break
            else:
                break
                
        features['current_streak_length'] = float(current_streak)
        features['current_streak_positive'] = float(streak_type == 'positive')
        features['current_streak_negative'] = float(streak_type == 'negative')
        
        return features
        
    def _create_qb_efficiency_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create QB-specific efficiency features."""
        features = {}
        
        # Completion percentage
        attempts = df['passing_attempts'].sum()
        completions = df['passing_completions'].sum()
        
        if attempts > 0:
            features['completion_percentage'] = float(completions / attempts)
            
        # Yards per attempt
        yards = df['passing_yards'].sum()
        if attempts > 0:
            features['yards_per_attempt'] = float(yards / attempts)
            
        # TD to INT ratio
        tds = df['passing_touchdowns'].sum()
        ints = df['passing_interceptions'].sum()
        
        if ints > 0:
            features['td_int_ratio'] = float(tds / ints)
        else:
            features['td_int_ratio'] = float(tds)  # No INTs is good
            
        return features
        
    def _create_rb_efficiency_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create RB-specific efficiency features."""
        features = {}
        
        # Yards per carry
        carries = df['rushing_attempts'].sum()
        yards = df['rushing_yards'].sum()
        
        if carries > 0:
            features['yards_per_carry'] = float(yards / carries)
            
        # Reception efficiency
        targets = df['targets'].sum()
        receptions = df['receptions'].sum()
        
        if targets > 0:
            features['catch_rate'] = float(receptions / targets)
            
        # Yards per touch (rushing + receiving)
        total_touches = carries + receptions
        total_yards = yards + df['receiving_yards'].sum()
        
        if total_touches > 0:
            features['yards_per_touch'] = float(total_yards / total_touches)
            
        return features
        
    def _create_receiver_efficiency_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create receiver-specific efficiency features."""
        features = {}
        
        # Catch rate
        targets = df['targets'].sum()
        receptions = df['receptions'].sum()
        
        if targets > 0:
            features['catch_rate'] = float(receptions / targets)
            
        # Yards per reception
        yards = df['receiving_yards'].sum()
        if receptions > 0:
            features['yards_per_reception'] = float(yards / receptions)
            
        # Yards per target
        if targets > 0:
            features['yards_per_target'] = float(yards / targets)
            
        return features
        
    def _create_usage_features(self, df: pd.DataFrame, position: str) -> Dict[str, float]:
        """Create usage pattern features."""
        features = {}
        
        # This would require team-level data to calculate shares
        # For now, create basic usage features
        
        if position == 'RB':
            features['avg_carries_per_game'] = float(df['rushing_attempts'].mean())
            features['avg_targets_per_game'] = float(df['targets'].mean())
            
        elif position in ['WR', 'TE']:
            features['avg_targets_per_game'] = float(df['targets'].mean())
            
        elif position == 'QB':
            features['avg_attempts_per_game'] = float(df['passing_attempts'].mean())
            
        return features
        
    def _create_performance_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create performance pattern features."""
        features = {}
        
        # Day of week patterns (would need game day data)
        # Time of day patterns (would need game time data)
        # Rest patterns (would need days between games)
        
        # For now, create basic pattern features
        fantasy_values = df['fantasy_points_ppr'].values
        
        # Performance volatility
        if len(fantasy_values) > 1:
            features['performance_volatility'] = float(np.std(fantasy_values))
            
        # Recency-weighted average (more weight to recent games)
        weights = np.exp(-0.1 * np.arange(len(fantasy_values)))
        weighted_avg = np.average(fantasy_values, weights=weights)
        features['recency_weighted_fantasy'] = float(weighted_avg)
        
        return features
        
    # Utility methods
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
        
    def _calculate_consistency(self, values: np.ndarray) -> float:
        """Calculate consistency score (1 - coefficient of variation)."""
        if len(values) < 2 or np.mean(values) == 0:
            return 0.0
            
        cv = np.std(values) / np.mean(values)
        return float(max(0, 1 - cv))
        
    def _get_division_teams(self, team: str) -> List[str]:
        """Get teams in the same division."""
        divisions = {
            'AFC_North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC_South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC_East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC_West': ['DEN', 'KC', 'LAC', 'LV'],
            'NFC_North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC_South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC_East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC_West': ['ARI', 'LAR', 'SF', 'SEA']
        }
        
        for division, teams in divisions.items():
            if team in teams:
                return [t for t in teams if t != team]
        return []
        
    def _is_dome_stadium(self, stadium: str) -> bool:
        """Check if stadium is a dome."""
        dome_stadiums = {
            'Mercedes-Benz Superdome', 'AT&T Stadium', 'Ford Field',
            'Lucas Oil Stadium', 'NRG Stadium', 'U.S. Bank Stadium',
            'State Farm Stadium', 'Allegiant Stadium', 'Mercedes-Benz Stadium'
        }
        return stadium in dome_stadiums
        
    def save_features_to_store(
        self,
        player_id: str,
        game_id: str,
        features: Dict[str, Any]
    ) -> bool:
        """Save engineered features to the feature store."""
        
        try:
            with self.Session() as session:
                # Organize features by category
                feature_categories = {
                    'recent_form_features': {},
                    'seasonal_features': {},
                    'opponent_features': {},
                    'contextual_features': {},
                    'advanced_features': {}
                }
                
                # Categorize features based on prefixes
                for feature_name, value in features.items():
                    if any(prefix in feature_name for prefix in ['last_', 'recent_']):
                        feature_categories['recent_form_features'][feature_name] = value
                    elif any(prefix in feature_name for prefix in ['season_', 'trend']):
                        feature_categories['seasonal_features'][feature_name] = value
                    elif any(prefix in feature_name for prefix in ['vs_', 'opponent']):
                        feature_categories['opponent_features'][feature_name] = value
                    elif any(prefix in feature_name for prefix in ['weather', 'stadium', 'week']):
                        feature_categories['contextual_features'][feature_name] = value
                    else:
                        feature_categories['advanced_features'][feature_name] = value
                
                # Create feature store entry
                feature_store = FeatureStore(
                    player_id=player_id,
                    game_id=game_id,
                    recent_form_features=feature_categories['recent_form_features'],
                    seasonal_features=feature_categories['seasonal_features'],
                    opponent_features=feature_categories['opponent_features'],
                    contextual_features=feature_categories['contextual_features'],
                    advanced_features=feature_categories['advanced_features'],
                    feature_version=self.config.feature_version
                )
                
                session.merge(feature_store)  # Use merge to handle duplicates
                session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving features to store: {e}")
            return False
            
    def load_features_from_store(
        self,
        player_id: str,
        game_id: str
    ) -> Dict[str, Any]:
        """Load features from the feature store."""
        
        try:
            with self.Session() as session:
                feature_store = session.query(FeatureStore).filter(
                    and_(
                        FeatureStore.player_id == player_id,
                        FeatureStore.game_id == game_id,
                        FeatureStore.feature_version == self.config.feature_version
                    )
                ).first()
                
                if not feature_store:
                    return {}
                    
                # Combine all feature categories
                features = {}
                for category in ['recent_form_features', 'seasonal_features', 
                               'opponent_features', 'contextual_features', 'advanced_features']:
                    category_features = getattr(feature_store, category, {})
                    if category_features:
                        features.update(category_features)
                        
                return features
                
        except Exception as e:
            self.logger.error(f"Error loading features from store: {e}")
            return {}


# Example usage
def main():
    """Example usage of the feature engineering system."""
    
    config = FeatureConfig(
        lookback_windows=[3, 5, 10],
        rolling_windows=[4, 8],
        min_games_threshold=3,
        feature_version="v1.0",
        scale_features=True,
        handle_missing="impute"
    )
    
    engineer = AdvancedFeatureEngineer(
        database_url="postgresql://user:password@localhost/nfl_predictions",
        config=config
    )
    
    # Engineer features for a specific player and game
    features = engineer.engineer_player_features(
        player_id="mahomes_patrick_qb",
        target_game_id="2024_10_KC_BUF",
        position="QB"
    )
    
    print(f"Engineered {len(features)} features")
    for key, value in list(features.items())[:10]:  # Show first 10
        print(f"{key}: {value}")
        
    # Save to feature store
    engineer.save_features_to_store(
        player_id="mahomes_patrick_qb",
        game_id="2024_10_KC_BUF",
        features=features
    )
    
    print("Features saved to feature store!")


if __name__ == "__main__":
    main()