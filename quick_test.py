#!/usr/bin/env python3
"""
Quick Test Suite for Enhanced NFL Betting Analyzer
Fast validation without heavy ML training.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prediction_targets():
    """Test prediction targets functionality."""
    print("🎯 Testing Prediction Targets...")
    
    try:
        from enhanced_prediction_targets import (
            PREDICTION_TARGETS, get_targets_for_position, StatCategory
        )
        
        # Test basic functionality
        qb_targets = get_targets_for_position('QB')
        assert len(qb_targets) > 0, "No QB targets found"
        
        rb_targets = get_targets_for_position('RB')
        assert len(rb_targets) > 0, "No RB targets found"
        
        all_targets = PREDICTION_TARGETS.get_all_targets()
        assert len(all_targets) > 50, f"Expected 50+ targets, got {len(all_targets)}"
        
        print(f"   ✅ Found {len(all_targets)} total prediction targets")
        print(f"   ✅ QB: {len(qb_targets)} targets")
        print(f"   ✅ RB: {len(rb_targets)} targets")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_database_connection():
    """Test database connectivity."""
    print("💾 Testing Database Connection...")
    
    try:
        from sqlalchemy import create_engine, text
        
        engine = create_engine("sqlite:///data/nfl_predictions.db")
        
        with engine.connect() as conn:
            # Test basic queries
            result = conn.execute(text("SELECT COUNT(*) FROM players"))
            player_count = result.scalar()
            
            result = conn.execute(text("SELECT COUNT(*) FROM player_game_stats"))
            stats_count = result.scalar()
            
            print(f"   ✅ Players: {player_count:,}")
            print(f"   ✅ Game Stats: {stats_count:,}")
            
            return stats_count > 0
            
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering without heavy computation."""
    print("🔧 Testing Feature Engineering...")
    
    try:
        from feature_engineering import AdvancedFeatureEngineer, FeatureConfig
        
        config = FeatureConfig(
            lookback_windows=[3, 5],
            rolling_windows=[4],
            min_games_threshold=3,
            feature_version="test_v1.0",
            scale_features=True,
            handle_missing="impute"
        )
        
        engineer = AdvancedFeatureEngineer("sqlite:///data/nfl_predictions.db", config)
        
        # Test initialization
        assert engineer.config is not None
        assert len(engineer.position_configs) > 0
        
        print("   ✅ Feature engineer initialized")
        print(f"   ✅ Position configs: {list(engineer.position_configs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature engineering error: {e}")
        return False

def test_ensemble_models_basic():
    """Test ensemble model initialization without training."""
    print("🤖 Testing Ensemble Models (Basic)...")
    
    try:
        from enhanced_ensemble_models import EnhancedEnsembleModel, ComprehensivePredictor
        
        config = {
            'model_types': ['random_forest'],  # Only RF for speed
            'ensemble_method': 'weighted_average'
        }
        
        # Test initialization
        ensemble = EnhancedEnsembleModel(config)
        assert ensemble.config is not None
        
        predictor = ComprehensivePredictor(config)
        assert predictor.config is not None
        assert predictor.model_dir.exists()
        
        print("   ✅ Ensemble model initialized")
        print("   ✅ Comprehensive predictor initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ensemble model error: {e}")
        return False

def test_data_validation():
    """Test data validation without heavy processing."""
    print("🔍 Testing Data Validation...")
    
    try:
        from data_validation_pipeline import DataValidator, ValidationRule
        
        validator = DataValidator("sqlite:///data/nfl_predictions.db")
        
        # Test validation rules initialization
        rules = validator.validation_rules
        assert len(rules) > 0, "No validation rules found"
        
        print(f"   ✅ Validation rules: {len(rules)}")
        print("   ✅ Data validator initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data validation error: {e}")
        return False

def test_comprehensive_analyzer():
    """Test comprehensive analyzer initialization."""
    print("📊 Testing Comprehensive Analyzer...")
    
    try:
        from comprehensive_betting_analyzer import ComprehensiveBettingAnalyzer
        
        config = {
            'model_types': ['random_forest'],
            'confidence_threshold': 0.6,
            'min_games_threshold': 3
        }
        
        analyzer = ComprehensiveBettingAnalyzer(config)
        
        # Test basic functionality
        assert analyzer.config is not None
        assert analyzer.feature_engineer is not None
        assert analyzer.predictor is not None
        
        print("   ✅ Comprehensive analyzer initialized")
        print("   ✅ All components connected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Comprehensive analyzer error: {e}")
        return False

def run_quick_tests():
    """Run all quick tests."""
    print("🧪 QUICK TEST SUITE - Enhanced NFL Betting Analyzer")
    print("=" * 60)
    
    tests = [
        ("Prediction Targets", test_prediction_targets),
        ("Database Connection", test_database_connection),
        ("Feature Engineering", test_feature_engineering),
        ("Ensemble Models", test_ensemble_models_basic),
        ("Data Validation", test_data_validation),
        ("Comprehensive Analyzer", test_comprehensive_analyzer),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
        print()
    
    print("📊 TEST RESULTS:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All quick tests passed! System components are working.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
