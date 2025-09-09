"""
Comprehensive unit tests for NFL Data Validation module
Tests data quality validation, stats validation, and validation reporting.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock
from sqlalchemy.orm import Session

from core.data.data_validator import (
    DataQualityValidator,
    StatsValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity
)
from core.data.data_foundation import (
    MasterPlayer,
    WeeklyRosterSnapshot,
    PlayerRole
)
from core.database_models import PlayerGameStats


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session"""
    return Mock(spec=Session)


@pytest.fixture
def data_quality_validator(mock_session):
    """Create a data quality validator instance"""
    return DataQualityValidator(mock_session)


@pytest.fixture
def stats_validator(mock_session):
    """Create a stats validator instance"""
    return StatsValidator(mock_session)


@pytest.fixture
def sample_master_player():
    """Create a sample master player for testing"""
    return MasterPlayer(
        nfl_id="12345",
        name="Test Player",
        position="WR",
        team="KC",
        jersey_number=87,
        height=72,
        weight=200,
        college="Test University",
        years_pro=3,
        birth_date=datetime(1995, 5, 15),
        is_active=True
    )


class TestDataQualityValidator:
    """Test suite for Data Quality Validator"""
    
    def test_initialization(self, mock_session):
        """Test proper initialization of data quality validator"""
        validator = DataQualityValidator(mock_session)
        
        assert validator.session == mock_session
        assert len(validator.validation_rules) > 0
    
    def test_validate_player_completeness_valid(self, data_quality_validator, sample_master_player):
        """Test player completeness validation with valid data"""
        result = data_quality_validator.validate_player_completeness(sample_master_player)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.severity == ValidationSeverity.INFO
        assert result.score >= 0.9
    
    def test_validate_player_completeness_missing_fields(self, data_quality_validator):
        """Test player completeness validation with missing fields"""
        incomplete_player = MasterPlayer(
            nfl_id="12345",
            name="Test Player",
            position="WR",
            team="KC",
            jersey_number=None,
            height=None,
            weight=200,
            college=None,
            years_pro=3,
            birth_date=None,
            is_active=True
        )
        
        result = data_quality_validator.validate_player_completeness(incomplete_player)
        
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert result.score < 0.8


class TestStatsValidator:
    """Test suite for Stats Validator"""
    
    def test_initialization(self, mock_session):
        """Test proper initialization of stats validator"""
        validator = StatsValidator(mock_session)
        
        assert validator.session == mock_session
        assert len(validator.position_stat_ranges) > 0
    
    def test_validate_fantasy_points_calculation_correct(self, stats_validator):
        """Test fantasy points calculation validation with correct calculation"""
        stat = Mock(spec=PlayerGameStats)
        stat.receiving_yards = 100
        stat.receptions = 8
        stat.receiving_touchdowns = 1
        stat.rushing_yards = 0
        stat.rushing_touchdowns = 0
        stat.passing_yards = 0
        stat.passing_touchdowns = 0
        stat.interceptions = 0
        stat.fumbles_lost = 0
        stat.fantasy_points_ppr = 24.0  # Correct: 100*0.1 + 8*1 + 1*6 = 24
        
        result = stats_validator.validate_fantasy_points_calculation(stat)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.severity == ValidationSeverity.INFO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
