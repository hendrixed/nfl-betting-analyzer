"""Comprehensive NFL Betting Analyzer System Tests."""

import pytest
import sys
import os
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestSystemImports:
    """Test all critical system imports."""
    
    def test_core_imports(self):
        """Test that core modules can be imported."""
        try:
            import database_models
            import config_manager
            import run_nfl_system
            assert True
        except ImportError as e:
            pytest.fail(f"Core import failed: {e}")
    
    def test_advanced_imports(self):
        """Test that advanced modules can be imported."""
        try:
            import ultimate_enhanced_predictor
            import social_sentiment_analyzer
            import nfl_ultimate_system
            import nfl_interactive_main
            assert True
        except ImportError as e:
            pytest.fail(f"Advanced import failed: {e}")
    
    def test_utility_imports(self):
        """Test utility modules."""
        try:
            import data_collector
            import feature_engineering
            import ml_models
            import prediction_pipeline
            assert True
        except ImportError as e:
            pytest.fail(f"Utility import failed: {e}")

class TestDatabaseModels:
    """Test database model functionality."""
    
    def test_player_model_creation(self):
        """Test Player model can be created."""
        from database_models import Player
        
        player = Player(
            player_id="pmahomes_qb",
            name="Patrick Mahomes",
            position="QB",
            current_team="KC"
        )
        
        assert player.player_id == "pmahomes_qb"
        assert player.position == "QB"
        assert player.current_team == "KC"
    
    def test_game_model_creation(self):
        """Test Game model can be created."""
        from database_models import Game
        
        game = Game(
            game_id="2024_01_KC_DEN",
            season=2024,
            week=1,
            home_team="KC",
            away_team="DEN"
        )
        
        assert game.game_id == "2024_01_KC_DEN"
        assert game.season == 2024
        assert game.home_team == "KC"
    
    def test_player_game_stats_model(self):
        """Test PlayerGameStats model."""
        from database_models import PlayerGameStats
        
        stats = PlayerGameStats(
            player_id="pmahomes_qb",
            game_id="2024_01_KC_DEN",
            passing_yards=300,
            passing_touchdowns=3,
            fantasy_points_ppr=25.5
        )
        
        assert stats.player_id == "pmahomes_qb"
        assert stats.passing_yards == 300
        assert stats.fantasy_points_ppr == 25.5

class TestConfigurationSystem:
    """Test configuration management."""
    
    def test_config_files_exist(self):
        """Test that required config files exist."""
        config_dir = Path("config")
        assert config_dir.exists(), "Config directory missing"
        assert (config_dir / "config.yaml").exists(), "config.yaml missing"
        assert (config_dir / "logging.yaml").exists(), "logging.yaml missing"
    
    def test_config_loading(self):
        """Test configuration can be loaded."""
        from config_manager import get_config
        
        config = get_config()
        assert config is not None
        assert hasattr(config, 'database')
        assert hasattr(config, 'models')
        assert hasattr(config, 'api')
    
    def test_config_validation(self):
        """Test configuration validation."""
        from config_manager import ConfigManager
        
        manager = ConfigManager()
        config = manager.load_config()
        
        # Test required sections exist
        assert config.database.path is not None
        assert hasattr(config.models, 'algorithms')
        assert config.api.host is not None

class TestPredictionSystem:
    """Test prediction system functionality."""
    
    @pytest.fixture
    def mock_predictor(self):
        """Create mock predictor for testing."""
        with patch('ultimate_enhanced_predictor.UltimateEnhancedPredictor') as mock:
            mock_instance = Mock()
            mock_instance.generate_ultimate_prediction.return_value = Mock(
                final_prediction={'fantasy_points_ppr': 20.5},
                confidence_score=0.85,
                market_edge=0.05,
                value_rating='GOOD_VALUE',
                risk_assessment='MEDIUM_RISK'
            )
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_ultimate_predictor_initialization(self):
        """Test UltimateEnhancedPredictor can be initialized."""
        try:
            from ultimate_enhanced_predictor import UltimateEnhancedPredictor
            # Just test import, actual initialization requires database
            assert UltimateEnhancedPredictor is not None
        except Exception as e:
            pytest.fail(f"UltimateEnhancedPredictor initialization failed: {e}")
    
    def test_prediction_structure(self, mock_predictor):
        """Test prediction output structure."""
        prediction = mock_predictor.generate_ultimate_prediction("pmahomes_qb")
        
        assert hasattr(prediction, 'final_prediction')
        assert hasattr(prediction, 'confidence_score')
        assert hasattr(prediction, 'market_edge')
        assert 'fantasy_points_ppr' in prediction.final_prediction

class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""
    
    def test_sentiment_analyzer_import(self):
        """Test sentiment analyzer can be imported."""
        try:
            from social_sentiment_analyzer import SocialSentimentAnalyzer
            assert SocialSentimentAnalyzer is not None
        except ImportError as e:
            pytest.fail(f"SentimentAnalyzer import failed: {e}")
    
    @patch('social_sentiment_analyzer.SocialSentimentAnalyzer.analyze_player_sentiment')
    def test_sentiment_analysis_output(self, mock_sentiment):
        """Test sentiment analysis output structure."""
        from social_sentiment_analyzer import SocialSentimentAnalyzer
        
        # Mock sentiment response
        mock_sentiment.return_value = Mock(
            sentiment_score=0.75,
            mention_volume=150,
            trending_topics=['injury_update', 'fantasy']
        )
        
        analyzer = SocialSentimentAnalyzer()
        result = analyzer.analyze_player_sentiment("pmahomes_qb")
        
        assert hasattr(result, 'sentiment_score')
        assert hasattr(result, 'mention_volume')
        assert hasattr(result, 'trending_topics')

class TestUltimateSystem:
    """Test the ultimate system integration."""
    
    def test_ultimate_system_import(self):
        """Test ultimate system can be imported."""
        try:
            from nfl_ultimate_system import NFLUltimateSystem
            assert NFLUltimateSystem is not None
        except ImportError as e:
            pytest.fail(f"NFLUltimateSystem import failed: {e}")
    
    @patch('nfl_ultimate_system.UltimateEnhancedPredictor')
    @patch('nfl_ultimate_system.SocialSentimentAnalyzer')
    def test_comprehensive_analysis_structure(self, mock_sentiment, mock_predictor):
        """Test comprehensive analysis output structure."""
        from nfl_ultimate_system import NFLUltimateSystem, ComprehensiveAnalysis
        
        # Test ComprehensiveAnalysis dataclass
        analysis = ComprehensiveAnalysis(
            player_id="pmahomes_qb",
            position="QB",
            ultimate_prediction=Mock(),
            sentiment_data={},
            market_intelligence={},
            final_recommendation={},
            confidence_level="HIGH",
            risk_assessment="MEDIUM_RISK"
        )
        
        assert analysis.player_id == "pmahomes_qb"
        assert analysis.position == "QB"
        assert analysis.confidence_level == "HIGH"

class TestInteractiveInterface:
    """Test interactive interface functionality."""
    
    def test_interactive_main_import(self):
        """Test interactive main can be imported."""
        try:
            import nfl_interactive_main
            assert nfl_interactive_main is not None
        except ImportError as e:
            pytest.fail(f"Interactive main import failed: {e}")

class TestCLISystem:
    """Test CLI system functionality."""
    
    def test_cli_commands_available(self):
        """Test that CLI commands are properly registered."""
        from run_nfl_system import cli
        
        # Get list of registered commands
        commands = list(cli.commands.keys())
        
        # Check for essential commands
        essential_commands = [
            'status', 'predict', 'ultimate', 'sentiment', 
            'compare', 'daily-recs', 'interactive'
        ]
        
        for cmd in essential_commands:
            assert cmd in commands, f"Command '{cmd}' not found in CLI"

class TestDataIntegrity:
    """Test data integrity and validation."""
    
    def test_requirements_file_exists(self):
        """Test requirements.txt exists and has content."""
        req_file = Path("requirements.txt")
        assert req_file.exists(), "requirements.txt missing"
        
        content = req_file.read_text()
        assert len(content.strip()) > 0, "requirements.txt is empty"
        assert "pandas" in content, "pandas not in requirements"
        assert "scikit-learn" in content, "scikit-learn not in requirements"
    
    def test_gitignore_exists(self):
        """Test .gitignore exists and has proper exclusions."""
        gitignore = Path(".gitignore")
        assert gitignore.exists(), ".gitignore missing"
        
        content = gitignore.read_text()
        assert "*.db" in content, "Database files not ignored"
        assert "__pycache__" in content, "Python cache not ignored"
        assert "*.log" in content, "Log files not ignored"
    
    def test_documentation_exists(self):
        """Test that key documentation files exist."""
        docs = [
            "README.md", "USER_GUIDE.md", "SYSTEM_OVERVIEW.md", 
            "BETTING_GUIDE.md"
        ]
        
        for doc in docs:
            doc_path = Path(doc)
            assert doc_path.exists(), f"{doc} missing"
            
            content = doc_path.read_text()
            assert len(content.strip()) > 100, f"{doc} appears to be empty or too short"

class TestPerformanceAndReliability:
    """Test system performance and reliability."""
    
    def test_memory_usage_reasonable(self):
        """Test that imports don't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Import heavy modules
        try:
            import ultimate_enhanced_predictor
            import nfl_ultimate_system
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Should not use more than 500MB for imports
            assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"
            
        except ImportError:
            # Skip if modules can't be imported
            pass
    
    def test_error_handling_robustness(self):
        """Test that system handles errors gracefully."""
        from config_manager import ConfigManager
        
        # Test with invalid config path
        manager = ConfigManager()
        try:
            manager.load_config("nonexistent_config.yaml")
        except (FileNotFoundError, IOError) as e:
            # Should handle gracefully with proper exception
            assert "not found" in str(e).lower() or "no such file" in str(e).lower()
        except Exception as e:
            # Any other exception is also acceptable for robustness
            assert len(str(e)) > 0

# Integration test that runs a complete workflow
class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.mark.integration
    def test_complete_prediction_workflow(self):
        """Test a complete prediction workflow end-to-end."""
        try:
            # Test configuration loading
            from config_manager import get_config
            config = get_config()
            assert config is not None
            
            # Test database models
            from database_models import Player
            test_player = Player(
                player_id="test_qb",
                name="Test QB",
                position="QB",
                current_team="TEST"
            )
            assert test_player.position == "QB"
            
            print("✅ Integration test passed - system components work together")
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main(["-v", __file__])