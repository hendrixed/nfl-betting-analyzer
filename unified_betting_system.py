#!/usr/bin/env python3
"""
Unified NFL Betting System
Combines the reliability of the working system with enhanced capabilities.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedBettingSystem:
    """Unified system combining working predictor with enhanced features."""
    
    def __init__(self):
        self.working_predictor = None
        self.streamlined_predictor = None
        self.initialize_systems()
    
    def initialize_systems(self):
        """Initialize both prediction systems."""
        try:
            # Initialize working predictor
            from working_betting_predictor import WorkingBettingPredictor
            self.working_predictor = WorkingBettingPredictor()
            logger.info("✅ Working predictor initialized")
            
            # Initialize streamlined enhanced predictor
            from streamlined_enhanced_system import StreamlinedEnhancedPredictor
            self.streamlined_predictor = StreamlinedEnhancedPredictor()
            self.streamlined_predictor.load_models()
            logger.info("✅ Streamlined enhanced predictor initialized")
            
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'working_system': {
                'available': self.working_predictor is not None,
                'models_loaded': 0,
                'database_connected': False
            },
            'enhanced_system': {
                'available': self.streamlined_predictor is not None,
                'models_loaded': 0,
                'database_connected': False
            }
        }
        
        # Check working system
        if self.working_predictor:
            try:
                status['working_system']['database_connected'] = self.working_predictor.test_database_connection()
                status['working_system']['models_loaded'] = len(self.working_predictor.models)
            except:
                pass
        
        # Check enhanced system
        if self.streamlined_predictor:
            try:
                status['enhanced_system']['models_loaded'] = len(self.streamlined_predictor.models)
                # Test database connection
                with self.streamlined_predictor.engine.connect() as conn:
                    conn.execute("SELECT 1")
                status['enhanced_system']['database_connected'] = True
            except:
                pass
        
        return status
    
    def get_combined_recommendations(self, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Get recommendations from both systems and combine them."""
        recommendations = []
        
        # Get working system recommendations
        if self.working_predictor:
            try:
                working_recs = self.working_predictor.get_betting_recommendations()
                for rec in working_recs:
                    rec['source'] = 'working_system'
                    rec['system_confidence'] = 0.8  # High confidence in working system
                    recommendations.append(rec)
                logger.info(f"Got {len(working_recs)} recommendations from working system")
            except Exception as e:
                logger.warning(f"Working system recommendations failed: {e}")
        
        # Get enhanced system recommendations
        if self.streamlined_predictor:
            try:
                enhanced_recs = self.streamlined_predictor.generate_recommendations(min_confidence)
                for rec in enhanced_recs:
                    rec['source'] = 'enhanced_system'
                    rec['system_confidence'] = rec.get('confidence', 0.5)
                    recommendations.append(rec)
                logger.info(f"Got {len(enhanced_recs)} recommendations from enhanced system")
            except Exception as e:
                logger.warning(f"Enhanced system recommendations failed: {e}")
        
        # Sort by combined confidence score
        for rec in recommendations:
            rec['combined_score'] = rec.get('system_confidence', 0.5) * rec.get('confidence', rec.get('edge', 0.5))
        
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return recommendations
    
    def train_enhanced_models(self) -> Dict[str, Any]:
        """Train enhanced models if needed."""
        if not self.streamlined_predictor:
            return {'error': 'Enhanced system not available'}
        
        try:
            results = self.streamlined_predictor.train_all_models()
            logger.info(f"Enhanced model training completed: {results['summary']['total']['trained']} models")
            return results
        except Exception as e:
            logger.error(f"Enhanced model training failed: {e}")
            return {'error': str(e)}
    
    def validate_predictions(self, player_id: str) -> Dict[str, Any]:
        """Validate predictions across both systems."""
        results = {
            'player_id': player_id,
            'working_system': {},
            'enhanced_system': {},
            'comparison': {}
        }
        
        # Get working system predictions
        if self.working_predictor:
            try:
                working_pred = self.working_predictor.predict_player_performance(player_id)
                results['working_system'] = working_pred or {}
            except Exception as e:
                results['working_system'] = {'error': str(e)}
        
        # Get enhanced system predictions
        if self.streamlined_predictor:
            try:
                # Try multiple targets
                enhanced_preds = {}
                position = None
                for pos in ['QB', 'RB', 'WR', 'TE']:
                    if player_id.endswith(f'_{pos.lower()}'):
                        position = pos
                        break
                
                if position and position in self.streamlined_predictor.prediction_targets:
                    for target in self.streamlined_predictor.prediction_targets[position]:
                        pred = self.streamlined_predictor.predict(player_id, target)
                        if pred:
                            enhanced_preds[target] = pred
                
                results['enhanced_system'] = enhanced_preds
            except Exception as e:
                results['enhanced_system'] = {'error': str(e)}
        
        return results

def print_system_status(status: Dict[str, Any]):
    """Print formatted system status."""
    print("🏈 UNIFIED NFL BETTING SYSTEM STATUS")
    print("=" * 50)
    
    # Working System
    ws = status['working_system']
    print(f"🔧 Working System:")
    print(f"  Available: {'✅' if ws['available'] else '❌'}")
    print(f"  Database: {'✅' if ws['database_connected'] else '❌'}")
    print(f"  Models: {ws['models_loaded']}")
    
    # Enhanced System
    es = status['enhanced_system']
    print(f"🚀 Enhanced System:")
    print(f"  Available: {'✅' if es['available'] else '❌'}")
    print(f"  Database: {'✅' if es['database_connected'] else '❌'}")
    print(f"  Models: {es['models_loaded']}")
    
    print()

def print_recommendations(recommendations: List[Dict[str, Any]], limit: int = 15):
    """Print formatted recommendations."""
    if not recommendations:
        print("❌ No recommendations available")
        return
    
    print(f"💡 TOP {min(limit, len(recommendations))} UNIFIED BETTING RECOMMENDATIONS:")
    print("-" * 80)
    
    working_count = sum(1 for r in recommendations if r.get('source') == 'working_system')
    enhanced_count = sum(1 for r in recommendations if r.get('source') == 'enhanced_system')
    
    print(f"📊 Sources: Working System ({working_count}) | Enhanced System ({enhanced_count})")
    print()
    
    for i, rec in enumerate(recommendations[:limit], 1):
        source_icon = "🔧" if rec.get('source') == 'working_system' else "🚀"
        
        if rec.get('source') == 'working_system':
            player_name = rec.get('player_name', rec.get('player_id', 'Unknown')).replace('_', ' ').title()
            target = rec.get('target', 'Unknown')
            prediction = rec.get('prediction', 0)
            confidence = rec.get('confidence', 0)
            
            print(f"{i:2d}. {source_icon} {player_name}")
            print(f"    📊 {target}: {prediction:.1f}")
            print(f"    🎯 Confidence: {confidence:.1%}")
            
        else:  # Enhanced system
            player_name = rec.get('player_id', 'Unknown').replace('_', ' ').title()
            target = rec.get('target', 'Unknown')
            prediction = rec.get('prediction', 0)
            recommendation = rec.get('recommendation', 'HOLD')
            confidence = rec.get('confidence', 0)
            edge = rec.get('edge', 0)
            
            print(f"{i:2d}. {source_icon} {player_name} ({rec.get('position', 'UNK')})")
            print(f"    📊 {target}: {recommendation} {prediction:.1f}")
            print(f"    🎯 Confidence: {confidence:.1%} | Edge: {edge:.1%}")
        
        print()

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Unified NFL Betting System")
    parser.add_argument('--mode', choices=['status', 'recommend', 'train', 'validate'], 
                       default='recommend', help='Operation mode')
    parser.add_argument('--player', type=str, help='Player ID for validation')
    parser.add_argument('--confidence', type=float, default=0.6, 
                       help='Minimum confidence threshold')
    parser.add_argument('--limit', type=int, default=15, 
                       help='Maximum number of recommendations')
    
    args = parser.parse_args()
    
    print("🏈 UNIFIED NFL BETTING SYSTEM")
    print("=" * 60)
    print("🔧 Combining Working System + 🚀 Enhanced Capabilities")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize unified system
        system = UnifiedBettingSystem()
        
        if args.mode == 'status':
            status = system.get_system_status()
            print_system_status(status)
            
        elif args.mode == 'train':
            print("🤖 Training enhanced models...")
            results = system.train_enhanced_models()
            if 'error' in results:
                print(f"❌ Training failed: {results['error']}")
            else:
                print(f"✅ Training completed: {results['summary']['total']['trained']} models")
                
        elif args.mode == 'validate':
            if not args.player:
                print("❌ Player ID required for validation mode")
                return
            
            print(f"🔍 Validating predictions for: {args.player}")
            results = system.validate_predictions(args.player)
            
            print("\n🔧 Working System Predictions:")
            for key, value in results['working_system'].items():
                print(f"  {key}: {value}")
            
            print("\n🚀 Enhanced System Predictions:")
            for key, value in results['enhanced_system'].items():
                print(f"  {key}: {value}")
                
        else:  # recommend mode
            print("🎯 Generating unified betting recommendations...")
            recommendations = system.get_combined_recommendations(args.confidence)
            print_recommendations(recommendations, args.limit)
        
        print("\n✅ Unified system operation completed successfully!")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
