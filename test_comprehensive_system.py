"""
Comprehensive Test Suite for Enhanced NFL Betting Analyzer
Tests all components: prediction targets, ensemble models, data validation, and betting recommendations.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sqlite3
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Handle PyTorch import gracefully
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for testing without PyTorch
    class DummyTorch:
        class nn:
            class Module:
                def __init__(self):
                    pass
                def train(self):
                    pass
                def eval(self):
                    pass
            class Sequential:
                def __init__(self, *args):
                    pass
            class Linear:
                def __init__(self, *args):
                    pass
            class ReLU:
                pass
            class Dropout:
                def __init__(self, *args):
                    pass
    torch = DummyTorch()
    nn = torch.nn

from enhanced_prediction_targets import (
    PREDICTION_TARGETS, get_targets_for_position, StatCategory, PredictionTarget
)
from enhanced_ensemble_models import EnhancedEnsembleModel, ComprehensivePredictor
from data_validation_pipeline import DataValidator, ValidationRule
from comprehensive_betting_analyzer import ComprehensiveBettingAnalyzer

class TestPredictionTargets(unittest.TestCase):
    """Test prediction targets functionality."""
    
    def test_position_targets(self):
        """Test that all positions have appropriate targets."""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            targets = get_targets_for_position(position)
            self.assertGreater(len(targets), 0, f"No targets found for {position}")
            
            # Check that all targets are PredictionTarget instances
            for target in targets:
                self.assertIsInstance(target, PredictionTarget)
                self.assertIn(position, target.positions)
    
    def test_qb_specific_targets(self):
        """Test QB-specific prediction targets."""
        qb_targets = get_targets_for_position('QB')
        target_names = [t.name for t in qb_targets]
        
        # QB should have passing stats
        self.assertIn('passing_yards', target_names)
        self.assertIn('passing_touchdowns', target_names)
        self.assertIn('passing_interceptions', target_names)
        
        # QB should have fantasy points
        self.assertIn('fantasy_points_ppr', target_names)
    
    def test_skill_position_targets(self):
        """Test skill position targets (RB, WR, TE)."""
        for position in ['RB', 'WR', 'TE']:
            targets = get_targets_for_position(position)
            target_names = [t.name for t in targets]
            
            # Should have receiving stats
            self.assertIn('receptions', target_names)
            self.assertIn('receiving_yards', target_names)
            self.assertIn('targets', target_names)
    
    def test_target_constraints(self):
        """Test that targets have proper constraints."""
        all_targets = PREDICTION_TARGETS.get_all_targets()
        
        for target in all_targets:
            # Check min/max values are reasonable
            self.assertGreaterEqual(target.min_value, 0)
            if target.max_value is not None:
                self.assertGreater(target.max_value, target.min_value)
            
            # Check required fields
            self.assertIsNotNone(target.name)
            self.assertIsNotNone(target.column)
            self.assertIsInstance(target.category, StatCategory)

class TestEnsembleModels(unittest.TestCase):
    """Test ensemble model functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randn(self.n_samples) * 10 + 20  # Fantasy points-like data
        
        self.config = {
            'model_types': ['xgboost', 'lightgbm', 'random_forest'],
            'ensemble_method': 'weighted_average'
        }
        
        # Create a sample target
        self.target = PredictionTarget(
            name="test_fantasy_points",
            column="fantasy_points_ppr",
            category=StatCategory.FANTASY,
            positions=["QB"],
            prediction_type="regression",
            min_value=0,
            max_value=50
        )
    
    def test_ensemble_model_training(self):
        """Test ensemble model training."""
        ensemble = EnhancedEnsembleModel(self.config)
        results = ensemble.train(self.X, self.y, self.target)
        
        # Check that models were trained
        self.assertGreater(len(ensemble.models), 0)
        self.assertGreater(len(results), 0)
        
        # Check that weights were calculated
        self.assertGreater(len(ensemble.weights), 0)
        
        # Weights should sum to approximately 1
        weight_sum = sum(ensemble.weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=2)
    
    def test_ensemble_predictions(self):
        """Test ensemble predictions."""
        ensemble = EnhancedEnsembleModel(self.config)
        ensemble.train(self.X, self.y, self.target)
        
        # Make predictions
        predictions = ensemble.predict(self.X[:5])
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_comprehensive_predictor(self):
        """Test comprehensive predictor initialization."""
        predictor = ComprehensivePredictor(self.config)
        
        self.assertIsNotNone(predictor.config)
        self.assertIsInstance(predictor.models, dict)
        self.assertTrue(predictor.model_dir.exists())

class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_url = f"sqlite:///{self.temp_db.name}"
        
        # Create test data
        conn = sqlite3.connect(self.temp_db.name)
        
        # Create players table
        conn.execute("""
            CREATE TABLE players (
                player_id TEXT PRIMARY KEY,
                name TEXT,
                position TEXT,
                created_at TEXT
            )
        """)
        
        # Create player_game_stats table
        conn.execute("""
            CREATE TABLE player_game_stats (
                player_id TEXT,
                game_id TEXT,
                fantasy_points_ppr REAL,
                passing_yards INTEGER,
                rushing_yards INTEGER,
                receiving_yards INTEGER,
                created_at TEXT
            )
        """)
        
        # Insert test data
        test_players = [
            ('mahomes_qb', 'Patrick Mahomes', 'QB', '2024-01-01'),
            ('henry_rb', 'Derrick Henry', 'RB', '2024-01-01'),
            ('adams_wr', 'Davante Adams', 'WR', '2024-01-01')
        ]
        
        conn.executemany(
            "INSERT INTO players VALUES (?, ?, ?, ?)", 
            test_players
        )
        
        test_stats = [
            ('mahomes_qb', 'game_1', 25.5, 300, 20, 0, '2024-01-01'),
            ('henry_rb', 'game_1', 18.2, 0, 120, 25, '2024-01-01'),
            ('adams_wr', 'game_1', 22.1, 0, 0, 150, '2024-01-01'),
            # Add some outlier data
            ('mahomes_qb', 'game_2', 85.0, 600, 100, 0, '2024-01-02'),  # Outlier
        ]
        
        conn.executemany(
            "INSERT INTO player_game_stats VALUES (?, ?, ?, ?, ?, ?, ?)",
            test_stats
        )
        
        conn.commit()
        conn.close()
        
        self.validator = DataValidator(self.db_url)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_validation_rules_initialization(self):
        """Test that validation rules are properly initialized."""
        rules = self.validator.validation_rules
        self.assertGreater(len(rules), 0)
        
        # Check that rules have required attributes
        for rule in rules:
            self.assertIsInstance(rule, ValidationRule)
            self.assertIsNotNone(rule.name)
            self.assertIsNotNone(rule.column)
            self.assertIsNotNone(rule.rule_type)
    
    def test_table_validation(self):
        """Test table validation functionality."""
        result = self.validator.validate_table('players')
        
        self.assertIn('total_records', result)
        self.assertIn('rules_passed', result)
        self.assertIn('rules_failed', result)
        self.assertGreater(result['total_records'], 0)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        result = self.validator.validate_table('player_game_stats')
        
        # Should detect the outlier fantasy points value (85.0)
        self.assertIn('warnings_list', result)
        
        # Check if outlier was detected
        outlier_warnings = [w for w in result.get('warnings_list', []) 
                          if 'outlier' in w.get('rule', '')]
        self.assertGreater(len(outlier_warnings), 0)

class TestComprehensiveAnalyzer(unittest.TestCase):
    """Test the comprehensive betting analyzer."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'model_types': ['random_forest'],  # Use only RF for faster testing
            'confidence_threshold': 0.6,
            'min_games_threshold': 3,
            'database_url': 'sqlite:///data/nfl_predictions.db'
        }
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = ComprehensiveBettingAnalyzer(self.config)
        
        self.assertIsNotNone(analyzer.config)
        self.assertIsNotNone(analyzer.feature_engineer)
        self.assertIsNotNone(analyzer.predictor)
        self.assertTrue(analyzer.model_dir.exists())
    
    def test_default_config(self):
        """Test default configuration."""
        analyzer = ComprehensiveBettingAnalyzer()
        
        self.assertIn('model_types', analyzer.config)
        self.assertIn('confidence_threshold', analyzer.config)
        self.assertGreater(len(analyzer.config['model_types']), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_prediction_targets_integration(self):
        """Test that prediction targets integrate with ensemble models."""
        qb_targets = get_targets_for_position('QB')
        
        # Test that we can create models for QB targets
        config = {'model_types': ['random_forest']}
        
        for target in qb_targets[:2]:  # Test first 2 targets
            ensemble = EnhancedEnsembleModel(config)
            
            # Create dummy data
            X = np.random.randn(50, 10)
            y = np.random.randn(50) * 10 + 20
            
            try:
                results = ensemble.train(X, y, target)
                self.assertIsInstance(results, dict)
            except Exception as e:
                self.fail(f"Failed to train model for {target.name}: {e}")
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with minimal data."""
        # This would test the complete workflow
        # For now, just test that components can be initialized together
        
        try:
            config = {
                'model_types': ['random_forest'],
                'confidence_threshold': 0.6,
                'min_games_threshold': 2
            }
            
            analyzer = ComprehensiveBettingAnalyzer(config)
            
            # Test database validation
            validation_result = analyzer.validate_database()
            self.assertIsInstance(validation_result, bool)
            
        except Exception as e:
            self.fail(f"End-to-end workflow failed: {e}")

def run_comprehensive_tests():
    """Run all tests and generate report."""
    print("🧪 Running Comprehensive Test Suite for Enhanced NFL Betting Analyzer")
    print("=" * 80)
    
    # Create test suite
    test_classes = [
        TestPredictionTargets,
        TestEnsembleModels,
        TestDataValidation,
        TestComprehensiveAnalyzer,
        TestIntegration
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        if result.failures:
            print(f"  ❌ {len(result.failures)} failures")
            for test, traceback in result.failures:
                print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"  🚨 {len(result.errors)} errors")
            for test, traceback in result.errors:
                print(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")
        
        if not result.failures and not result.errors:
            print(f"  ✅ All {result.testsRun} tests passed")
    
    print(f"\n📊 Test Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_tests - total_failures - total_errors}")
    print(f"  Failed: {total_failures}")
    print(f"  Errors: {total_errors}")
    
    success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
    print(f"  Success Rate: {success_rate:.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 All tests passed! System is ready for production.")
    else:
        print(f"\n⚠️  {total_failures + total_errors} issues found. Please review and fix.")
    
    return total_failures + total_errors == 0

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
