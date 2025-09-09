#!/usr/bin/env python3
"""
System Validation Script
Tests all critical fixes implemented for the NFL Betting Analyzer
"""

import sys
import os
import traceback
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_database_connections():
    """Test database connections and session management."""
    print("🔍 Testing Database Connections...")
    try:
        from core.database_models import get_db_session, Player, Game
        
        session = get_db_session()
        player_count = session.query(Player).count()
        game_count = session.query(Game).count()
        active_players = session.query(Player).filter(Player.is_active == True).count()
        
        session.close()
        
        print(f"✅ Database connection successful")
        print(f"   📊 Total players: {player_count}")
        print(f"   🎮 Total games: {game_count}")
        print(f"   🏃 Active players: {active_players}")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_streamlined_models():
    """Test streamlined models initialization and predictions."""
    print("\n🤖 Testing Streamlined Models...")
    try:
        from core.database_models import get_db_session, Player
        from core.models.streamlined_models import StreamlinedNFLModels
        
        session = get_db_session()
        models = StreamlinedNFLModels(session)
        
        # Test with a few active players
        active_players = session.query(Player).filter(Player.is_active == True).limit(3).all()
        
        predictions_made = 0
        for player in active_players:
            try:
                result = models.predict_player(player.player_id)
                if result:
                    predictions_made += 1
                    print(f"   ✅ {player.name} ({player.position}): {result.predicted_value:.2f} points")
            except Exception as e:
                print(f"   ⚠️  {player.name}: {e}")
        
        session.close()
        
        if predictions_made > 0:
            print(f"✅ Streamlined models working - {predictions_made} predictions generated")
            return True
        else:
            print("❌ No predictions generated")
            return False
            
    except Exception as e:
        print(f"❌ Streamlined models failed: {e}")
        return False

def test_prediction_bounds():
    """Test prediction bounds validation system."""
    print("\n📏 Testing Prediction Bounds Validation...")
    try:
        from core.prediction_bounds import PredictionBoundsValidator
        
        validator = PredictionBoundsValidator()
        
        # Test various predictions
        test_cases = [
            {'position': 'QB', 'value': 20.5, 'expected': 'valid'},
            {'position': 'QB', 'value': 100.0, 'expected': 'invalid'},
            {'position': 'RB', 'value': 15.2, 'expected': 'valid'},
            {'position': 'WR', 'value': -5.0, 'expected': 'invalid'}
        ]
        
        passed_tests = 0
        for test in test_cases:
            result = validator.validate_fantasy_prediction(test['position'], test['value'])
            is_valid = result['is_valid']
            
            if (test['expected'] == 'valid' and is_valid) or (test['expected'] == 'invalid' and not is_valid):
                passed_tests += 1
                print(f"   ✅ {test['position']} {test['value']} -> {'Valid' if is_valid else 'Invalid'}")
            else:
                print(f"   ❌ {test['position']} {test['value']} -> Expected {test['expected']}, got {'valid' if is_valid else 'invalid'}")
        
        if passed_tests == len(test_cases):
            print("✅ Prediction bounds validation working correctly")
            return True
        else:
            print(f"❌ Prediction bounds validation failed {len(test_cases) - passed_tests} tests")
            return False
            
    except Exception as e:
        print(f"❌ Prediction bounds validation failed: {e}")
        return False

def test_api_imports():
    """Test that API files can be imported without errors."""
    print("\n📦 Testing API Import Compatibility...")
    
    tests = [
        ('Unified API', 'api.app'),
        ('Prediction Bounds', 'core.prediction_bounds')
    ]
    
    passed_imports = 0
    for name, module_path in tests:
        try:
            __import__(module_path)
            print(f"   ✅ {name} imports successfully")
            passed_imports += 1
        except Exception as e:
            print(f"   ❌ {name} import failed: {e}")
    
    if passed_imports == len(tests):
        print("✅ All API imports working correctly")
        return True
    else:
        print(f"❌ {len(tests) - passed_imports} API import failures")
        return False

def test_retired_players():
    """Test that retired players are correctly marked as inactive."""
    print("\n👴 Testing Retired Players Status...")
    try:
        from core.database_models import get_db_session, Player
        
        session = get_db_session()
        
        # Check known retired players
        retired_names = ['Tom Brady', 'Matt Ryan', 'Ben Roethlisberger']
        all_inactive = True
        
        for name in retired_names:
            players = session.query(Player).filter(Player.name.like(f'%{name}%')).all()
            for player in players:
                if player.is_active:
                    print(f"   ❌ {player.name} is still marked as active")
                    all_inactive = False
                else:
                    print(f"   ✅ {player.name} correctly marked as inactive")
        
        session.close()
        
        if all_inactive:
            print("✅ Retired players correctly marked as inactive")
            return True
        else:
            print("❌ Some retired players still marked as active")
            return False
            
    except Exception as e:
        print(f"❌ Retired players test failed: {e}")
        return False

def test_data_integrity():
    """Test basic data integrity checks."""
    print("\n🔍 Testing Data Integrity...")
    try:
        from core.database_models import get_db_session, Player
        
        session = get_db_session()
        
        # Check for active players with NULL teams
        null_teams = session.query(Player).filter(
            Player.current_team.is_(None), 
            Player.is_active == True
        ).count()
        
        # Check for reasonable data distribution
        total_active = session.query(Player).filter(Player.is_active == True).count()
        qb_count = session.query(Player).filter(Player.position == 'QB', Player.is_active == True).count()
        
        session.close()
        
        issues = []
        if null_teams > 0:
            issues.append(f"{null_teams} active players with NULL teams")
        if qb_count == 0:
            issues.append("No active QBs found")
        if total_active < 100:
            issues.append(f"Only {total_active} active players (seems low)")
        
        if not issues:
            print("✅ Data integrity checks passed")
            print(f"   📊 {total_active} active players, {qb_count} active QBs")
            return True
        else:
            print("❌ Data integrity issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ Data integrity test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 NFL Betting Analyzer - System Validation")
    print("=" * 50)
    print(f"Validation started at: {datetime.now()}")
    print()
    
    tests = [
        test_database_connections,
        test_streamlined_models,
        test_prediction_bounds,
        test_api_imports,
        test_retired_players,
        test_data_integrity
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
        print("✅ System is ready for Phase 2 enhancements")
    else:
        print("⚠️  Some tests failed - review issues above")
        print("🔧 Additional fixes may be needed")
    
    print(f"\nValidation completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
