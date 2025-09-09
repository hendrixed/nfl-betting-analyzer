"""
Unit tests for NFL Feature Engineering module
Tests comprehensive feature engineering functionality with mock data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from core.models.feature_engineering import (
    NFLFeatureEngineer, 
    FeatureSet, 
    ModelFeatures
)
from core.database_models import Player, Team, Game, PlayerGameStats
from core.data.ingestion_adapters import CacheManager


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session"""
    session = Mock(spec=Session)
    return session


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager"""
    cache_manager = Mock(spec=CacheManager)
    return cache_manager


@pytest.fixture
def feature_engineer(mock_session, mock_cache_manager):
    """Create a feature engineer instance with mocked dependencies"""
    return NFLFeatureEngineer(mock_session, mock_cache_manager)


@pytest.fixture
def sample_player():
    """Create a sample player for testing"""
    player = Mock(spec=Player)
    player.player_id = "test_player_001"
    player.name = "Test Player"
    player.position = "WR"
    player.current_team = "KC"
    return player


@pytest.fixture
def sample_game_stats():
    """Create sample game statistics"""
    stats = []
    for week in range(1, 9):  # 8 weeks of data
        stat = Mock(spec=PlayerGameStats)
        stat.player_id = "test_player_001"
        stat.season = 2024
        stat.week = week
        stat.passing_yards = None
        stat.passing_touchdowns = None
        stat.passing_attempts = None
        stat.passing_completions = None
        stat.interceptions = None
        stat.rushing_yards = 15 if week % 3 == 0 else 0  # Occasional rushing
        stat.rushing_touchdowns = 0
        stat.rushing_attempts = 2 if week % 3 == 0 else 0
        stat.receiving_yards = 80 + (week * 5)  # Trending up
        stat.receiving_touchdowns = 1 if week % 2 == 0 else 0
        stat.receptions = 6 + week
        stat.targets = 8 + week
        stat.snap_percentage = 0.75 + (week * 0.02)  # Increasing usage
        stats.append(stat)
    return stats


@pytest.fixture
def sample_games():
    """Create sample games for matchup analysis"""
    games = []
    for week in range(1, 10):
        game = Mock(spec=Game)
        game.game_id = f"2024_{week:02d}_KC_OPP"
        game.season = 2024
        game.week = week
        game.home_team = "KC" if week % 2 == 0 else "OPP"
        game.away_team = "OPP" if week % 2 == 0 else "KC"
        game.home_score = 24 + week
        game.away_score = 20 + (week % 3)
        games.append(game)
    return games


class TestNFLFeatureEngineer:
    """Test suite for NFL Feature Engineer"""
    
    def test_initialization(self, mock_session, mock_cache_manager):
        """Test proper initialization of feature engineer"""
        engineer = NFLFeatureEngineer(mock_session, mock_cache_manager)
        
        assert engineer.session == mock_session
        assert engineer.cache_manager == mock_cache_manager
        assert len(engineer.lookback_windows) == 4
        assert 'skill' in engineer.position_groups
        assert 'QB' in engineer.position_weights
    
    @pytest.mark.asyncio
    async def test_engineer_player_features_complete_flow(
        self, 
        feature_engineer, 
        sample_player, 
        sample_game_stats
    ):
        """Test complete feature engineering flow for a player"""
        
        # Mock database queries with proper chaining
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_player
        mock_query.filter.return_value.scalar.return_value = "KC"
        mock_query.join.return_value.filter.return_value.order_by.return_value.all.return_value = sample_game_stats
        mock_query.join.return_value.filter.return_value.all.return_value = sample_game_stats
        mock_query.filter.return_value.all.return_value = []  # For opponent stats
        feature_engineer.session.query.return_value = mock_query
        
        # Mock cache data
        feature_engineer.cache_manager.load_from_cache.side_effect = [
            {"temperature": 65.0, "wind_speed": 5.0, "precipitation": 0.0, "conditions": "clear"},
            [{"player_id": "test_player_001", "injury_status": "probable"}]
        ]
        
        features = await feature_engineer.engineer_player_features(
            "test_player_001", 2024, 9
        )
        
        # Verify feature categories are present
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for historical features (rolling averages)
        historical_features = [k for k in features.keys() if '_avg_' in k or '_std_' in k]
        assert len(historical_features) > 0
        
        # Check for situational features
        assert 'temperature' in features
        assert 'injury_status' in features
        
        # Verify feature values are reasonable
        assert 0 <= features['injury_status'] <= 1.0
        assert features['temperature'] > 0
    
    @pytest.mark.asyncio
    async def test_compute_historical_features(
        self, 
        feature_engineer, 
        sample_game_stats
    ):
        """Test historical feature computation"""
        
        # Mock the database query
        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.order_by.return_value.all.return_value = sample_game_stats[:5]
        feature_engineer.session.query.return_value = mock_query
        
        features = await feature_engineer._compute_historical_features(
            "test_player_001", 2024, 9, "WR"
        )
        
        # Check for rolling averages
        assert 'receiving_yards_avg_3g' in features
        assert 'targets_avg_3g' in features
        assert 'snap_percentage_avg_3g' in features
        
        # Check for standard deviations
        assert 'receiving_yards_std_3g' in features
        
        # Check for trends
        trend_features = [k for k in features.keys() if '_trend_' in k]
        assert len(trend_features) > 0
        
        # Verify values are reasonable
        assert features['receiving_yards_avg_3g'] > 0
        assert features['snap_percentage_avg_3g'] > 0
    
    @pytest.mark.asyncio
    async def test_compute_matchup_features(
        self, 
        feature_engineer, 
        sample_games
    ):
        """Test matchup-specific feature computation"""
        
        # Mock game query
        feature_engineer.session.query.return_value.filter.return_value.first.return_value = sample_games[8]  # Week 9 game
        feature_engineer.session.query.return_value.filter.return_value.scalar.return_value = "KC"
        feature_engineer.session.query.return_value.filter.return_value.all.return_value = sample_games[:8]
        
        # Mock opponent stats query
        mock_opp_stats = []
        for i in range(3):  # 3 games of opponent data
            stat = Mock()
            stat.passing_yards = 250
            stat.rushing_yards = 100
            stat.receiving_yards = 200
            stat.passing_touchdowns = 2
            stat.rushing_touchdowns = 1
            stat.receiving_touchdowns = 1
            mock_opp_stats.append(stat)
        
        feature_engineer.session.query.return_value.filter.return_value.all.return_value = mock_opp_stats
        
        features = await feature_engineer._compute_matchup_features(
            "test_player_001", 2024, 9
        )
        
        # Check for home/away indicator
        assert 'is_home' in features
        assert features['is_home'] in [0.0, 1.0]
        
        # Check for opponent defensive stats
        opp_features = [k for k in features.keys() if k.startswith('opp_')]
        assert len(opp_features) > 0
    
    @pytest.mark.asyncio
    async def test_compute_situational_features(self, feature_engineer):
        """Test situational feature computation"""
        
        # Mock weather data
        weather_data = {
            "temperature": 45.0,
            "wind_speed": 15.0,
            "precipitation": 0.1,
            "conditions": "rain"
        }
        
        # Mock injury data
        injury_data = [
            {"player_id": "test_player_001", "injury_status": "questionable"}
        ]
        
        feature_engineer.cache_manager.load_from_cache.side_effect = [weather_data, injury_data]
        
        features = await feature_engineer._compute_situational_features(
            "test_player_001", 2024, 9
        )
        
        # Check weather features
        assert features['temperature'] == 45.0
        assert features['wind_speed'] == 15.0
        assert features['precipitation'] == 0.1
        assert features['is_dome'] == 0.0
        
        # Check injury feature
        assert features['injury_status'] == 0.75  # questionable = 0.75
    
    @pytest.mark.asyncio
    async def test_compute_team_context_features(
        self, 
        feature_engineer, 
        sample_games
    ):
        """Test team context feature computation"""
        
        # Mock recent games query
        feature_engineer.session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = sample_games[:4]
        
        features = await feature_engineer._compute_team_context_features(
            "KC", 2024, 9
        )
        
        # Check team performance features
        assert 'team_win_pct_4g' in features
        assert 'team_ppg_4g' in features
        assert 'team_papg_4g' in features
        assert 'team_point_diff_4g' in features
        
        # Verify values are reasonable
        assert 0 <= features['team_win_pct_4g'] <= 1.0
        assert features['team_ppg_4g'] > 0
        assert features['team_papg_4g'] > 0
    
    @pytest.mark.asyncio
    async def test_compute_advanced_metrics_qb(self, feature_engineer):
        """Test advanced metrics computation for QB"""
        
        # Create QB stats
        qb_stats = []
        for i in range(5):
            stat = Mock()
            stat.passing_attempts = 35
            stat.passing_completions = 25
            stat.passing_yards = 280
            stat.passing_touchdowns = 2
            stat.interceptions = 1
            stat.snap_percentage = 0.98
            qb_stats.append(stat)
        
        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.all.return_value = qb_stats
        feature_engineer.session.query.return_value = mock_query
        
        features = await feature_engineer._compute_advanced_metrics(
            "qb_001", 2024, 9, "QB"
        )
        
        # Check QB-specific metrics
        assert 'completion_pct' in features
        assert 'yards_per_attempt' in features
        assert 'td_rate' in features
        assert 'int_rate' in features
        assert 'avg_snap_pct' in features
        
        # Verify calculations
        assert abs(features['completion_pct'] - (25/35)) < 0.01
        assert abs(features['yards_per_attempt'] - (280/35)) < 0.01
    
    @pytest.mark.asyncio
    async def test_compute_advanced_metrics_wr(self, feature_engineer):
        """Test advanced metrics computation for WR"""
        
        # Create WR stats
        wr_stats = []
        for i in range(5):
            stat = Mock()
            stat.targets = 8
            stat.receptions = 6
            stat.receiving_yards = 85
            stat.snap_percentage = 0.75
            wr_stats.append(stat)
        
        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.all.return_value = wr_stats
        feature_engineer.session.query.return_value = mock_query
        
        features = await feature_engineer._compute_advanced_metrics(
            "wr_001", 2024, 9, "WR"
        )
        
        # Check WR-specific metrics
        assert 'catch_rate' in features
        assert 'yards_per_target' in features
        assert 'target_share' in features
        
        # Verify calculations
        assert abs(features['catch_rate'] - (6/8)) < 0.01
        assert abs(features['yards_per_target'] - (85/8)) < 0.01
    
    def test_prepare_training_data(self, feature_engineer):
        """Test training data preparation"""
        
        # Create sample feature data
        feature_data = [
            {
                'receiving_yards_avg_3g': 85.0,
                'targets_avg_3g': 8.5,
                'snap_percentage_avg_3g': 0.75,
                'is_home': 1.0,
                'temperature': 70.0,
                'fantasy_points_ppr': 18.5
            },
            {
                'receiving_yards_avg_3g': 92.0,
                'targets_avg_3g': 9.2,
                'snap_percentage_avg_3g': 0.82,
                'is_home': 0.0,
                'temperature': 65.0,
                'fantasy_points_ppr': 21.2
            },
            {
                'receiving_yards_avg_3g': 78.0,
                'targets_avg_3g': 7.8,
                'snap_percentage_avg_3g': 0.68,
                'is_home': 1.0,
                'temperature': 75.0,
                'fantasy_points_ppr': 16.8
            }
        ]
        
        model_features = feature_engineer.prepare_training_data(feature_data)
        
        # Check structure
        assert isinstance(model_features, ModelFeatures)
        assert model_features.X_train.shape[0] > 0
        assert model_features.X_test.shape[0] > 0
        assert len(model_features.y_train) > 0
        assert len(model_features.y_test) > 0
        assert len(model_features.feature_names) == 5  # 5 features
        assert model_features.target_name == 'fantasy_points_ppr'
        
        # Check scaling (should have mean ~0, std ~1)
        train_means = np.mean(model_features.X_train, axis=0)
        train_stds = np.std(model_features.X_train, axis=0)
        assert np.allclose(train_means, 0, atol=1e-10)
        assert np.allclose(train_stds, 1, atol=1e-10)
    
    def test_select_features(self, feature_engineer):
        """Test feature selection functionality"""
        
        # Create sample model features with many features
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        X_train = np.random.randn(n_samples, n_features)
        X_test = np.random.randn(20, n_features)
        y_train = np.random.randn(n_samples)
        y_test = np.random.randn(20)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        model_features = ModelFeatures(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            scaler=Mock(),
            target_name='target'
        )
        
        # Select top 10 features
        selected_features = feature_engineer.select_features(model_features, k=10)
        
        # Check that features were selected
        assert selected_features.X_train.shape[1] == 10
        assert selected_features.X_test.shape[1] == 10
        assert len(selected_features.feature_names) == 10
        
        # Check that feature names are subset of original
        assert all(name in feature_names for name in selected_features.feature_names)
    
    def test_error_handling_no_player(self, feature_engineer):
        """Test error handling when player is not found"""
        
        # Mock empty query result
        feature_engineer.session.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ValueError, match="Player .* not found"):
            import asyncio
            asyncio.run(feature_engineer.engineer_player_features("nonexistent", 2024, 9))
    
    def test_error_handling_no_feature_data(self, feature_engineer):
        """Test error handling with empty feature data"""
        
        with pytest.raises(ValueError, match="No feature data provided"):
            feature_engineer.prepare_training_data([])
    
    def test_error_handling_missing_target(self, feature_engineer):
        """Test error handling when target column is missing"""
        
        feature_data = [{'feature1': 1.0, 'feature2': 2.0}]
        
        with pytest.raises(ValueError, match="Target column .* not found"):
            feature_engineer.prepare_training_data(feature_data, target_column='missing_target')
    
    @pytest.mark.parametrize("position,expected_weights", [
        ("QB", {"passing": 0.7, "rushing": 0.2, "receiving": 0.0, "team_context": 0.1}),
        ("RB", {"passing": 0.0, "rushing": 0.6, "receiving": 0.3, "team_context": 0.1}),
        ("WR", {"passing": 0.0, "rushing": 0.05, "receiving": 0.85, "team_context": 0.1}),
        ("TE", {"passing": 0.0, "rushing": 0.05, "receiving": 0.75, "team_context": 0.2})
    ])
    def test_position_weights(self, feature_engineer, position, expected_weights):
        """Test position-specific feature weights"""
        
        weights = feature_engineer.position_weights[position]
        assert weights == expected_weights
        
        # Verify weights sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-10


class TestFeatureIntegration:
    """Integration tests for feature engineering with real-like data"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_feature_pipeline(self, mock_session, mock_cache_manager):
        """Test complete feature engineering pipeline"""
        
        engineer = NFLFeatureEngineer(mock_session, mock_cache_manager)
        
        # Mock comprehensive data
        player = Mock(spec=Player)
        player.player_id = "integration_test"
        player.position = "WR"
        player.current_team = "KC"
        
        # Create realistic game stats
        game_stats = []
        for week in range(1, 9):
            stat = Mock()
            stat.receiving_yards = 70 + np.random.randint(-20, 30)
            stat.receptions = 5 + np.random.randint(-2, 4)
            stat.targets = 8 + np.random.randint(-3, 5)
            stat.snap_percentage = 0.7 + np.random.uniform(-0.1, 0.2)
            game_stats.append(stat)
        
        # Mock all database queries
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = player
        mock_query.filter.return_value.scalar.return_value = "KC"
        mock_query.join.return_value.filter.return_value.order_by.return_value.all.return_value = game_stats
        mock_query.join.return_value.filter.return_value.all.return_value = game_stats
        mock_query.filter.return_value.all.return_value = []
        mock_session.query.return_value = mock_query
        
        # Mock cache data
        mock_cache_manager.load_from_cache.side_effect = [
            {"temperature": 72.0, "wind_speed": 8.0, "conditions": "clear"},
            [{"player_id": "integration_test", "injury_status": "probable"}]
        ]
        
        # Run feature engineering
        features = await engineer.engineer_player_features("integration_test", 2024, 9)
        
        # Verify comprehensive feature set
        assert len(features) >= 10  # Should have many features
        
        # Check feature categories
        historical_features = [k for k in features.keys() if any(window in k for window in ['3g', '5g', '8g'])]
        situational_features = [k for k in features.keys() if k in ['temperature', 'wind_speed', 'injury_status']]
        
        assert len(historical_features) > 0
        assert len(situational_features) > 0
        
        # Verify feature quality
        for feature_name, value in features.items():
            assert isinstance(value, (int, float, np.number))
            assert not np.isnan(value)
            assert not np.isinf(value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
