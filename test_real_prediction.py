#!/usr/bin/env python3
"""
EVIDENCE-BASED PREDICTION TEST
Test actual prediction functionality with real code and output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database_models import get_db_session, Player
from core.models.streamlined_models import StreamlinedNFLModels

def test_real_prediction():
    """Test actual prediction with Patrick Mahomes"""
    try:
        print("=== REAL PREDICTION TEST ===")
        
        # Connect to actual database
        session = get_db_session("sqlite:///nfl_predictions.db")
        print("Database connected")
        
        # Find Patrick Mahomes
        mahomes = session.query(Player).filter(
            Player.name.ilike('%Patrick Mahomes%')
        ).first()
        
        if not mahomes:
            print("Patrick Mahomes not found in database")
            # Try Aaron Rodgers instead
            mahomes = session.query(Player).filter(
                Player.name.ilike('%Aaron Rodgers%')
            ).first()
            if not mahomes:
                print("No QB found for testing")
                return False
        
        print(f"Found player: {mahomes.name} (ID: {mahomes.player_id}, Position: {mahomes.position})")
        
        # Load models
        models = StreamlinedNFLModels(session)
        print("Models loaded")
        
        # Make actual prediction
        print(f"\n--- Testing prediction for {mahomes.name} ---")
        result = models.predict_player(
            player_id=mahomes.player_id,
            target_stat='fantasy_points_ppr'
        )
        
        if result:
            print(f"PREDICTION SUCCESS:")
            print(f"  Player: {mahomes.name}")
            print(f"  Predicted Value: {result.predicted_value}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Model Used: {result.model_used}")
            return True
        else:
            print("Prediction returned None")
            return False
            
    except Exception as e:
        print(f"PREDICTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    success = test_real_prediction()
    if success:
        print("\nSUCCESS: Prediction system works")
        sys.exit(0)
    else:
        print("\nFAILURE: Prediction system broken")
        sys.exit(1)
