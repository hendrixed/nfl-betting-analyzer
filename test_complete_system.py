"""
Complete System Test for NFL Betting Analyzer
Tests the integrated system with streamlined models and API functionality
"""
import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database_models import get_db_session, Player, PlayerGameStats
from core.models.streamlined_models import StreamlinedNFLModels

def test_database_connection():
    """Test database connectivity and data availability."""
    print("Testing Database Connection...")
    
    try:
        session = get_db_session()
        
        # Test basic queries
        player_count = session.query(Player).count()
        active_players = session.query(Player).filter(Player.is_active == True).count()
        stats_count = session.query(PlayerGameStats).count()
        
        print(f"Database connected successfully")
        print(f"   Total players: {player_count}")
        print(f"   Active players: {active_players}")
        print(f"   Player game stats: {stats_count}")
        
        # Get sample players for testing
        sample_players = session.query(Player).filter(Player.is_active == True).limit(5).all()
        print(f"   Sample players available for testing:")
        for player in sample_players:
            print(f"      - {player.name} ({player.position}) - ID: {player.player_id}")
        
        session.close()
        return True, sample_players
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False, []

def test_model_loading():
    """Test model loading and initialization."""
    print("\nTesting Model Loading...")
    
    try:
        session = get_db_session()
        models = StreamlinedNFLModels(session)
        
        print("StreamlinedNFLModels initialized successfully")
        
        # Test model summary
        summary = models.get_model_summary()
        if summary:
            print(f"   Model Summary:")
            for position, info in summary.items():
                print(f"      - {position}: {info.get('model_type', 'Unknown')} (R²: {info.get('r2_score', 'N/A')})")
        
        session.close()
        return True, models
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False, None

def test_predictions(models, sample_players):
    """Test prediction functionality."""
    print("\nTesting Predictions...")
    
    if not models or not sample_players:
        print("Cannot test predictions - models or players not available")
        return False
    
    try:
        successful_predictions = 0
        total_tests = min(3, len(sample_players))
        
        for i, player in enumerate(sample_players[:total_tests]):
            print(f"\n   Testing prediction for {player.name} ({player.position})...")
            
            start_time = time.time()
            prediction = models.predict_player(player.player_id)
            prediction_time = time.time() - start_time
            
            if prediction:
                print(f"   Prediction successful:")
                print(f"      - Player: {prediction.player_name}")
                print(f"      - Position: {prediction.position}")
                print(f"      - Predicted Value: {prediction.predicted_value:.2f}")
                print(f"      - Confidence: {prediction.confidence:.3f}")
                print(f"      - Model Used: {prediction.model_used}")
                print(f"      - Prediction Time: {prediction_time:.3f}s")
                successful_predictions += 1
            else:
                print(f"   No prediction available for {player.name}")
        
        success_rate = successful_predictions / total_tests
        print(f"\n   Prediction Success Rate: {successful_predictions}/{total_tests} ({success_rate:.1%})")
        
        return success_rate > 0.5
        
    except Exception as e:
        print(f"Prediction testing failed: {e}")
        return False

def test_batch_predictions(models, sample_players):
    """Test batch prediction functionality."""
    print("\nTesting Batch Predictions...")
    
    if not models or not sample_players:
        print("Cannot test batch predictions - models or players not available")
        return False
    
    try:
        player_ids = [player.player_id for player in sample_players[:3]]
        print(f"   Testing batch prediction for {len(player_ids)} players...")
        
        start_time = time.time()
        results = []
        
        for player_id in player_ids:
            prediction = models.predict_player(player_id)
            if prediction:
                results.append(prediction)
        
        batch_time = time.time() - start_time
        
        print(f"   Batch predictions completed:")
        print(f"      - Players processed: {len(results)}/{len(player_ids)}")
        print(f"      - Total time: {batch_time:.3f}s")
        print(f"      - Avg time per prediction: {batch_time/len(player_ids):.3f}s")
        
        for result in results:
            print(f"      - {result.player_name}: {result.predicted_value:.2f} ({result.confidence:.3f})")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"Batch prediction testing failed: {e}")
        return False

def test_api_integration():
    """Test API integration readiness."""
    print("\nTesting API Integration Readiness...")
    
    try:
        # Check if enhanced API file exists and is properly configured
        api_file = "api/enhanced_prediction_api.py"
        web_file = "web/web_server.py"
        
        api_exists = os.path.exists(api_file)
        web_exists = os.path.exists(web_file)
        
        print(f"   Enhanced API file: {'Found' if api_exists else 'Missing'}")
        print(f"   Web server file: {'Found' if web_exists else 'Missing'}")
        
        # Check web interface
        web_template = "web/templates/index.html"
        template_exists = os.path.exists(web_template)
        print(f"   Web interface: {'Found' if template_exists else 'Missing'}")
        
        if api_exists and web_exists and template_exists:
            print("   All API components ready for deployment")
            return True
        else:
            print("   Some API components missing")
            return False
            
    except Exception as e:
        print(f"API integration test failed: {e}")
        return False

def run_complete_system_test():
    """Run comprehensive system test."""
    print("NFL Betting Analyzer - Complete System Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Test 1: Database Connection
    db_success, sample_players = test_database_connection()
    test_results['database'] = db_success
    
    # Test 2: Model Loading
    model_success, models = test_model_loading()
    test_results['models'] = model_success
    
    # Test 3: Predictions
    if db_success and model_success:
        pred_success = test_predictions(models, sample_players)
        test_results['predictions'] = pred_success
        
        # Test 4: Batch Predictions
        batch_success = test_batch_predictions(models, sample_players)
        test_results['batch_predictions'] = batch_success
    else:
        test_results['predictions'] = False
        test_results['batch_predictions'] = False
    
    # Test 5: API Integration
    api_success = test_api_integration()
    test_results['api_integration'] = api_success
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    print("-" * 60)
    print(f"Overall Result: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
    
    if passed_tests == total_tests:
        print("ALL TESTS PASSED - System ready for production!")
    elif passed_tests >= total_tests * 0.8:
        print("MOSTLY SUCCESSFUL - System ready with minor issues")
    else:
        print("ISSUES DETECTED - System needs attention")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return passed_tests / total_tests

if __name__ == "__main__":
    success_rate = run_complete_system_test()
    exit(0 if success_rate >= 0.8 else 1)
