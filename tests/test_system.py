"""Basic system tests."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    try:
        import database_models
        import data_collector
        import feature_engineering
        import ml_models
        import prediction_pipeline
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_database_models():
    """Test database models can be created."""
    from database_models import Player, Game, PlayerGameStats
    
    # Test model creation
    player = Player(
        player_id="test_player",
        name="Test Player",
        position="QB",
        current_team="TEST"
    )
    
    assert player.player_id == "test_player"
    assert player.position == "QB"

def test_config_files():
    """Test that config files exist."""
    config_dir = Path("config")
    assert config_dir.exists()
    assert (config_dir / "config.yaml").exists()