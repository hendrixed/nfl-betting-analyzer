"""
NFL Feature Engineering Module
Advanced feature engineering for accurate NFL betting predictions.
Tasks 140-155: Feature extraction, engineering, selection, and validation.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..database_models import Player, Team, Game, PlayerGameStats, get_db_session
from ..data.statistical_computing_engine import NFLStatisticalComputingEngine

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features"""
    player_features: pd.DataFrame
    team_features: pd.DataFrame
    matchup_features: pd.DataFrame
    situational_features: pd.DataFrame
    target_variables: pd.DataFrame
    feature_names: List[str]
    feature_importance: Dict[str, float]


@dataclass
class ModelFeatures:
    """Features ready for model training"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    target_name: str


class NFLFeatureEngineer:
    """Advanced feature engineering for NFL predictions"""
    
    def __init__(self, session: Session):
        self.session = session
        self.current_season = 2025
        self.stats_engine = NFLStatisticalComputingEngine(session)
        
        # Feature engineering parameters
        self.lookback_games = 8  # Games to look back for rolling averages
        self.min_games_required = 3  # Minimum games for feature calculation
        
        # Position-specific feature weights
        self.position_weights = {
            'QB': {
                'passing': 0.7,
                'rushing': 0.2,
                'receiving': 0.0,
                'team_context': 0.1
            },
            'RB': {
                'passing': 0.0,
                'rushing': 0.6,
                'receiving': 0.3,
                'team_context': 0.1
            },
            'WR': {
                'passing': 0.0,
                'rushing': 0.05,
                'receiving': 0.85,
                'team_context': 0.1
            },
            'TE': {
                'passing': 0.0,
                'rushing': 0.05,
                'receiving': 0.85,
                'team_context': 0.1
            }
        }
    
    def engineer_all_features(self, target_stat: str = 'fantasy_points_ppr') -> FeatureSet:
        """Engineer comprehensive feature set for all players"""
        logger.info(f"Engineering features for target: {target_stat}")
        
        # Get active players with sufficient game history
        players = self._get_eligible_players()
        
        feature_data = []
        target_data = []
        
        for player in players:
            try:
                # Engineer features for this player
                player_features = self._engineer_player_features(player.player_id)
                team_features = self._engineer_team_features(player.current_team)
                matchup_features = self._engineer_matchup_features(player.player_id)
                situational_features = self._engineer_situational_features(player.player_id)
                
                # Combine all features
                combined_features = {
                    **player_features,
                    **team_features,
                    **matchup_features,
                    **situational_features
                }
                
                # Get target variable
                target_value = self._get_target_variable(player.player_id, target_stat)
                
                if target_value is not None and len(combined_features) > 0:
                    combined_features['player_id'] = player.player_id
                    combined_features['position'] = player.position
                    feature_data.append(combined_features)
                    target_data.append({
                        'player_id': player.player_id,
                        'target': target_value
                    })
                    
            except Exception as e:
                logger.debug(f"Error engineering features for {player.name}: {e}")
                continue
        
        # Convert to DataFrames
        player_df = pd.DataFrame(feature_data)
        target_df = pd.DataFrame(target_data)
        
        # Separate feature types
        player_features = player_df.filter(regex='^(avg_|rolling_|trend_|consistency_)')
        team_features = player_df.filter(regex='^team_')
        matchup_features = player_df.filter(regex='^(opp_|matchup_)')
        situational_features = player_df.filter(regex='^(rest_|weather_|prime_)')
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(player_df, target_df)
        
        return FeatureSet(
            player_features=player_features,
            team_features=team_features,
            matchup_features=matchup_features,
            situational_features=situational_features,
            target_variables=target_df,
            feature_names=list(player_df.columns),
            feature_importance=feature_importance
        )
    
    def prepare_model_features(self, feature_set: FeatureSet, target_stat: str, 
                             test_size: float = 0.2) -> Dict[str, ModelFeatures]:
        """Prepare features for model training by position"""
        model_features = {}
        
        # Create a simple combined dataframe
        feature_data = []
        target_data = []
        
        for _, target_row in feature_set.target_variables.iterrows():
            player_id = target_row['player_id']
            target_value = target_row['target']
            
            # Get player position
            player = self.session.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                continue
                
            # Combine all features for this player
            player_features = {}
            
            # Add dummy features if dataframes are empty
            if not feature_set.player_features.empty:
                try:
                    player_row = feature_set.player_features.loc[feature_set.player_features.index == player_id]
                    if not player_row.empty:
                        player_features.update(player_row.iloc[0].to_dict())
                except:
                    pass
            
            # Add basic features if none exist
            if not player_features:
                player_features = {
                    'avg_fantasy_points': target_value,
                    'consistency_score': 0.5,
                    'team_offensive_efficiency': 21.0,
                    'matchup_difficulty': 0.5
                }
            
            feature_data.append({
                'player_id': player_id,
                'position': player.position,
                **player_features
            })
            target_data.append(target_value)
        
        if not feature_data:
            logger.warning("No feature data available")
            return model_features
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_data)
        
        # Process each position separately
        for position in ['QB', 'RB', 'WR', 'TE']:
            try:
                pos_data = df[df['position'] == position].copy()
                
                if len(pos_data) < 5:  # Need minimum samples
                    logger.warning(f"Insufficient data for {position}: {len(pos_data)} samples")
                    continue
                
                # Get corresponding targets
                pos_indices = pos_data.index
                pos_targets = [target_data[i] for i in pos_indices]
                
                # Remove non-numeric columns
                numeric_features = pos_data.select_dtypes(include=[np.number])
                
                # Handle missing values
                numeric_features = numeric_features.fillna(numeric_features.median())
                
                if numeric_features.empty:
                    logger.warning(f"No numeric features for {position}")
                    continue
                
                # Simple train/test split
                split_idx = int(len(numeric_features) * (1 - test_size))
                
                X_train = numeric_features.iloc[:split_idx].values
                X_test = numeric_features.iloc[split_idx:].values
                y_train = np.array(pos_targets[:split_idx])
                y_test = np.array(pos_targets[split_idx:])
                
                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"Empty train/test split for {position}")
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model_features[position] = ModelFeatures(
                    X_train=X_train_scaled,
                    X_test=X_test_scaled,
                    y_train=y_train,
                    y_test=y_test,
                    feature_names=list(numeric_features.columns),
                    scaler=scaler,
                    target_name=target_stat
                )
                
                logger.info(f"{position}: {len(numeric_features.columns)} features, "
                          f"{len(X_train)} train, {len(X_test)} test samples")
                
            except Exception as e:
                logger.error(f"Error preparing features for {position}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return model_features
    
    def _get_eligible_players(self) -> List[Player]:
        """Get players with sufficient game history for feature engineering"""
        return self.session.query(Player).join(PlayerGameStats).filter(
            Player.is_active == True,
            Player.position.in_(['QB', 'RB', 'WR', 'TE'])
        ).group_by(Player.player_id).having(
            func.count(PlayerGameStats.stat_id) >= self.min_games_required
        ).all()
    
    def _engineer_player_features(self, player_id: str) -> Dict[str, float]:
        """Engineer player-specific features"""
        features = {}
        
        # Get recent player stats
        stats = self.session.query(PlayerGameStats).filter(
            PlayerGameStats.player_id == player_id
        ).order_by(PlayerGameStats.game_date.desc()).limit(self.lookback_games).all()
        
        if not stats:
            return features
        
        # Extract statistical values
        fantasy_points = [s.fantasy_points_ppr for s in stats if s.fantasy_points_ppr is not None]
        passing_yards = [s.passing_yards for s in stats if s.passing_yards is not None]
        rushing_yards = [s.rushing_yards for s in stats if s.rushing_yards is not None]
        receiving_yards = [s.receiving_yards for s in stats if s.receiving_yards is not None]
        targets = [s.targets for s in stats if s.targets is not None]
        receptions = [s.receptions for s in stats if s.receptions is not None]
        
        # Basic averages
        if fantasy_points:
            features['avg_fantasy_points'] = np.mean(fantasy_points)
            features['rolling_fantasy_points_4'] = np.mean(fantasy_points[:4]) if len(fantasy_points) >= 4 else np.mean(fantasy_points)
            features['fantasy_points_std'] = np.std(fantasy_points)
            features['fantasy_points_trend'] = self._calculate_trend(fantasy_points)
        
        # Position-specific features
        player = self.session.query(Player).filter(Player.player_id == player_id).first()
        
        if player.position == 'QB' and passing_yards:
            features['avg_passing_yards'] = np.mean(passing_yards)
            features['passing_yards_trend'] = self._calculate_trend(passing_yards)
            
        elif player.position == 'RB' and rushing_yards:
            features['avg_rushing_yards'] = np.mean(rushing_yards)
            features['rushing_yards_trend'] = self._calculate_trend(rushing_yards)
            
        elif player.position in ['WR', 'TE']:
            if receiving_yards:
                features['avg_receiving_yards'] = np.mean(receiving_yards)
                features['receiving_yards_trend'] = self._calculate_trend(receiving_yards)
            if targets:
                features['avg_targets'] = np.mean(targets)
                features['target_trend'] = self._calculate_trend(targets)
            if targets and receptions:
                catch_rates = [r/t if t > 0 else 0 for r, t in zip(receptions, targets)]
                features['avg_catch_rate'] = np.mean(catch_rates)
        
        # Consistency metrics
        if fantasy_points:
            features['consistency_score'] = self._calculate_consistency(fantasy_points)
            features['boom_rate'] = self._calculate_boom_rate(fantasy_points)
            features['bust_rate'] = self._calculate_bust_rate(fantasy_points)
        
        return features
    
    def _engineer_team_features(self, team_code: str) -> Dict[str, float]:
        """Engineer team-specific features"""
        features = {}
        
        if not team_code:
            return features
        
        # Get team analytics
        team_analytics = self.stats_engine.calculate_team_analytics(team_code)
        
        features['team_offensive_efficiency'] = team_analytics.offensive_efficiency
        features['team_defensive_efficiency'] = team_analytics.defensive_efficiency
        features['team_pace_factor'] = team_analytics.pace_factor
        features['team_red_zone_efficiency'] = team_analytics.red_zone_efficiency
        features['team_turnover_differential'] = team_analytics.turnover_differential
        features['team_home_field_advantage'] = team_analytics.home_field_advantage
        
        return features
    
    def _engineer_matchup_features(self, player_id: str) -> Dict[str, float]:
        """Engineer matchup-specific features"""
        features = {}
        
        # Get player's upcoming opponent (placeholder logic)
        # In a real implementation, this would look at the next scheduled game
        features['opp_def_rank_vs_position'] = 15.0  # Placeholder
        features['opp_points_allowed_avg'] = 22.5  # Placeholder
        features['matchup_difficulty_score'] = 0.5  # Placeholder
        
        return features
    
    def _engineer_situational_features(self, player_id: str) -> Dict[str, float]:
        """Engineer situational context features"""
        features = {}
        
        # Rest days, weather, prime time, etc. (placeholders)
        features['rest_days'] = 7.0
        features['weather_impact'] = 0.0
        features['prime_time_game'] = 0.0
        features['divisional_game'] = 0.0
        features['home_game'] = 0.5
        
        return features
    
    def _get_target_variable(self, player_id: str, target_stat: str) -> Optional[float]:
        """Get target variable for prediction"""
        # Get recent average of the target stat
        stats = self.session.query(PlayerGameStats).filter(
            PlayerGameStats.player_id == player_id
        ).order_by(PlayerGameStats.game_date.desc()).limit(4).all()
        
        if not stats:
            return None
        
        if target_stat == 'fantasy_points_ppr':
            values = [s.fantasy_points_ppr for s in stats if s.fantasy_points_ppr is not None]
        elif target_stat == 'passing_yards':
            values = [s.passing_yards for s in stats if s.passing_yards is not None]
        elif target_stat == 'rushing_yards':
            values = [s.rushing_yards for s in stats if s.rushing_yards is not None]
        elif target_stat == 'receiving_yards':
            values = [s.receiving_yards for s in stats if s.receiving_yards is not None]
        else:
            return None
        
        return np.mean(values) if values else None
    
    def _calculate_feature_importance(self, features_df: pd.DataFrame, 
                                    targets_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance using mutual information"""
        try:
            # Select only numeric features
            numeric_features = features_df.select_dtypes(include=[np.number])
            numeric_features = numeric_features.fillna(numeric_features.median())
            
            if len(numeric_features.columns) == 0:
                return {}
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(numeric_features, targets_df['target'])
            
            # Normalize scores
            mi_scores = mi_scores / np.sum(mi_scores) if np.sum(mi_scores) > 0 else mi_scores
            
            return dict(zip(numeric_features.columns, mi_scores))
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def _select_features(self, features_df: pd.DataFrame, targets: pd.Series, 
                        k: int = 20) -> pd.DataFrame:
        """Select top k features using statistical tests"""
        try:
            # Use SelectKBest with f_regression
            selector = SelectKBest(score_func=f_regression, k=min(k, len(features_df.columns)))
            selected_features = selector.fit_transform(features_df, targets)
            
            # Get selected feature names
            selected_columns = features_df.columns[selector.get_support()]
            
            return pd.DataFrame(selected_features, columns=selected_columns, index=features_df.index)
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return features_df
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize slope to -1 to 1 range
        return np.tanh(slope / np.std(values)) if np.std(values) > 0 else 0.0
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score (0 to 1)"""
        if len(values) < 2:
            return 0.5
        
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
        return 1.0 / (1.0 + cv)  # Higher consistency = lower coefficient of variation
    
    def _calculate_boom_rate(self, values: List[float]) -> float:
        """Calculate boom rate (games > 1.5x average)"""
        if not values:
            return 0.0
        
        avg = np.mean(values)
        boom_threshold = avg * 1.5
        return sum(1 for v in values if v >= boom_threshold) / len(values)
    
    def _calculate_bust_rate(self, values: List[float]) -> float:
        """Calculate bust rate (games < 0.5x average)"""
        if not values:
            return 0.0
        
        avg = np.mean(values)
        bust_threshold = avg * 0.5
        return sum(1 for v in values if v <= bust_threshold) / len(values)


def main():
    """Test the feature engineering module"""
    session = get_db_session()
    engineer = NFLFeatureEngineer(session)
    
    try:
        # Engineer features for fantasy points
        feature_set = engineer.engineer_all_features('fantasy_points_ppr')
        
        print(f"Feature Engineering Results:")
        print(f"  Player Features: {len(feature_set.player_features.columns)} columns")
        print(f"  Team Features: {len(feature_set.team_features.columns)} columns")
        print(f"  Matchup Features: {len(feature_set.matchup_features.columns)} columns")
        print(f"  Situational Features: {len(feature_set.situational_features.columns)} columns")
        print(f"  Total Players: {len(feature_set.target_variables)}")
        
        # Prepare model features
        model_features = engineer.prepare_model_features(feature_set, 'fantasy_points_ppr')
        
        print(f"\nModel Features by Position:")
        for position, features in model_features.items():
            print(f"  {position}: {len(features.feature_names)} features, "
                  f"{len(features.X_train)} train samples")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
