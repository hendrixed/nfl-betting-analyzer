#!/usr/bin/env python3
"""
Enhanced NFL Betting System - Main Entry Point
Comprehensive production-ready system with all enhanced features.
"""

import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_nfl_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the enhanced NFL betting system."""
    parser = argparse.ArgumentParser(description='Enhanced NFL Betting Analyzer')
    parser.add_argument('--mode', choices=['analyze', 'train', 'validate', 'test'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🏈 ENHANCED NFL BETTING ANALYZER")
    print("=" * 60)
    print(f"🚀 Mode: {args.mode.upper()}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        if args.mode == 'analyze':
            run_comprehensive_analysis()
        elif args.mode == 'train':
            run_model_training()
        elif args.mode == 'validate':
            run_data_validation()
        elif args.mode == 'test':
            run_system_tests()
            
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        sys.exit(1)

def run_comprehensive_analysis():
    """Run the comprehensive betting analysis."""
    try:
        from comprehensive_betting_analyzer import ComprehensiveBettingAnalyzer
        
        config = {
            'model_types': ['xgboost', 'lightgbm', 'random_forest'],
            'confidence_threshold': 0.65,
            'prop_bet_threshold': 0.7,
            'min_games_threshold': 5,
            'backtesting_enabled': True
        }
        
        analyzer = ComprehensiveBettingAnalyzer(config)
        analyzer.display_comprehensive_analysis()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ Missing dependencies. Please install requirements:")
        print("   pip install -r requirements_enhanced.txt")
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        print(f"❌ Analysis failed: {e}")

def run_model_training():
    """Run comprehensive model training."""
    try:
        from comprehensive_betting_analyzer import ComprehensiveBettingAnalyzer
        
        print("🤖 Starting comprehensive model training...")
        
        config = {
            'model_types': ['xgboost', 'lightgbm', 'random_forest', 'neural_network'],
            'confidence_threshold': 0.65,
            'min_games_threshold': 5
        }
        
        analyzer = ComprehensiveBettingAnalyzer(config)
        analyzer.train_comprehensive_models()
        
        print("✅ Model training completed!")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        print(f"❌ Training failed: {e}")

def run_data_validation():
    """Run comprehensive data validation."""
    try:
        from data_validation_pipeline import DataValidator, AutomatedWorkflowManager
        
        print("🔍 Running comprehensive data validation...")
        
        validator = DataValidator("sqlite:///data/nfl_predictions.db")
        results = validator.validate_all_tables()
        
        print(f"Validation Status: {results['overall_status']}")
        print(f"Tables Validated: {results['tables_validated']}")
        
        for table, result in results["table_results"].items():
            if "error" not in result:
                print(f"\n{table}:")
                print(f"  Records: {result['total_records']:,}")
                print(f"  Rules Passed: {result['rules_passed']}")
                print(f"  Rules Failed: {result['rules_failed']}")
                print(f"  Warnings: {result['warnings']}")
        
        # Run automated workflow
        print("\n🔄 Running automated workflow...")
        workflow_manager = AutomatedWorkflowManager(
            "sqlite:///data/nfl_predictions.db", 
            {"retrain_threshold": 100}
        )
        workflow_result = workflow_manager.run_daily_workflow()
        
        print(f"Workflow Status: {workflow_result['overall_status']}")
        print(f"Steps Completed: {len(workflow_result['steps'])}")
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        print(f"❌ Validation failed: {e}")

def run_system_tests():
    """Run comprehensive system tests."""
    try:
        from test_comprehensive_system import run_comprehensive_tests
        
        print("🧪 Running comprehensive system tests...")
        success = run_comprehensive_tests()
        
        if success:
            print("\n🎉 All tests passed! System is production-ready.")
        else:
            print("\n⚠️  Some tests failed. Please review and fix issues.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Testing error: {e}")
        print(f"❌ Testing failed: {e}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'xgboost', 'lightgbm', 
        'sqlalchemy', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install -r requirements_enhanced.txt")
        return False
    
    return True

def display_system_info():
    """Display system information and capabilities."""
    print("📊 SYSTEM CAPABILITIES:")
    print("-" * 40)
    
    try:
        from enhanced_prediction_targets import PREDICTION_TARGETS
        
        summary = PREDICTION_TARGETS.get_position_summary()
        total_targets = len(PREDICTION_TARGETS.get_all_targets())
        
        print(f"🎯 Total Prediction Targets: {total_targets}")
        
        for position, categories in summary.items():
            target_count = sum(categories.values())
            print(f"   {position}: {target_count} targets")
        
        print(f"\n🤖 Model Types: XGBoost, LightGBM, Random Forest, Neural Networks")
        print(f"🔧 Features: Advanced ensemble methods, prop bets, backtesting")
        print(f"📈 Positions: QB, RB, WR, TE (+ DEF, K support)")
        
    except ImportError:
        print("⚠️  Enhanced modules not available")

if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Display system info
    display_system_info()
    print()
    
    # Run main function
    main()
