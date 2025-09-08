#!/usr/bin/env python3
"""
Prediction Bounds Validation System
Ensures all predictions fall within realistic statistical ranges
"""

from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PredictionBoundsValidator:
    """Validates prediction values are within realistic bounds for each position."""
    
    # Position-specific bounds for fantasy points (min, max, typical_range)
    FANTASY_BOUNDS = {
        'QB': {'min': -5.0, 'max': 50.0, 'typical': (8.0, 35.0)},
        'RB': {'min': -2.0, 'max': 40.0, 'typical': (4.0, 25.0)},
        'WR': {'min': -2.0, 'max': 45.0, 'typical': (3.0, 28.0)},
        'TE': {'min': -2.0, 'max': 35.0, 'typical': (2.0, 20.0)},
        'K': {'min': 0.0, 'max': 25.0, 'typical': (5.0, 15.0)},
        'DST': {'min': -10.0, 'max': 30.0, 'typical': (2.0, 18.0)}
    }
    
    # Statistical bounds for individual stats
    STAT_BOUNDS = {
        'passing_yards': {'min': 0, 'max': 600, 'typical': (180, 350)},
        'passing_tds': {'min': 0, 'max': 7, 'typical': (0, 4)},
        'rushing_yards': {'min': -10, 'max': 300, 'typical': (0, 150)},
        'rushing_tds': {'min': 0, 'max': 5, 'typical': (0, 2)},
        'receiving_yards': {'min': 0, 'max': 250, 'typical': (0, 120)},
        'receiving_tds': {'min': 0, 'max': 4, 'typical': (0, 2)},
        'receptions': {'min': 0, 'max': 20, 'typical': (0, 12)}
    }
    
    def __init__(self):
        self.validation_errors = []
        self.warnings = []
    
    def validate_fantasy_prediction(self, position: str, predicted_value: float) -> Dict:
        """Validate fantasy points prediction for a position."""
        result = {
            'is_valid': True,
            'is_typical': True,
            'warnings': [],
            'corrected_value': predicted_value
        }
        
        if position not in self.FANTASY_BOUNDS:
            result['warnings'].append(f"Unknown position: {position}")
            return result
        
        bounds = self.FANTASY_BOUNDS[position]
        
        # Check hard bounds
        if predicted_value < bounds['min']:
            result['is_valid'] = False
            result['corrected_value'] = bounds['min']
            result['warnings'].append(f"Value {predicted_value:.2f} below minimum {bounds['min']}")
        elif predicted_value > bounds['max']:
            result['is_valid'] = False
            result['corrected_value'] = bounds['max']
            result['warnings'].append(f"Value {predicted_value:.2f} above maximum {bounds['max']}")
        
        # Check typical range
        typical_min, typical_max = bounds['typical']
        if not (typical_min <= predicted_value <= typical_max):
            result['is_typical'] = False
            if predicted_value < typical_min:
                result['warnings'].append(f"Value {predicted_value:.2f} below typical range {typical_min}-{typical_max}")
            else:
                result['warnings'].append(f"Value {predicted_value:.2f} above typical range {typical_min}-{typical_max}")
        
        return result
    
    def validate_stat_prediction(self, stat_name: str, predicted_value: float) -> Dict:
        """Validate individual statistical prediction."""
        result = {
            'is_valid': True,
            'is_typical': True,
            'warnings': [],
            'corrected_value': predicted_value
        }
        
        if stat_name not in self.STAT_BOUNDS:
            result['warnings'].append(f"Unknown stat: {stat_name}")
            return result
        
        bounds = self.STAT_BOUNDS[stat_name]
        
        # Check hard bounds
        if predicted_value < bounds['min']:
            result['is_valid'] = False
            result['corrected_value'] = bounds['min']
            result['warnings'].append(f"Stat {stat_name} value {predicted_value:.2f} below minimum {bounds['min']}")
        elif predicted_value > bounds['max']:
            result['is_valid'] = False
            result['corrected_value'] = bounds['max']
            result['warnings'].append(f"Stat {stat_name} value {predicted_value:.2f} above maximum {bounds['max']}")
        
        # Check typical range
        typical_min, typical_max = bounds['typical']
        if not (typical_min <= predicted_value <= typical_max):
            result['is_typical'] = False
            if predicted_value < typical_min:
                result['warnings'].append(f"Stat {stat_name} value {predicted_value:.2f} below typical range {typical_min}-{typical_max}")
            else:
                result['warnings'].append(f"Stat {stat_name} value {predicted_value:.2f} above typical range {typical_min}-{typical_max}")
        
        return result
    
    def validate_prediction_batch(self, predictions: list) -> Dict:
        """Validate a batch of predictions."""
        results = {
            'total_predictions': len(predictions),
            'valid_predictions': 0,
            'typical_predictions': 0,
            'corrections_made': 0,
            'warnings': [],
            'corrected_predictions': []
        }
        
        for pred in predictions:
            if 'position' in pred and 'predicted_value' in pred:
                validation = self.validate_fantasy_prediction(pred['position'], pred['predicted_value'])
                
                corrected_pred = pred.copy()
                corrected_pred['predicted_value'] = validation['corrected_value']
                corrected_pred['validation_warnings'] = validation['warnings']
                
                results['corrected_predictions'].append(corrected_pred)
                
                if validation['is_valid']:
                    results['valid_predictions'] += 1
                else:
                    results['corrections_made'] += 1
                
                if validation['is_typical']:
                    results['typical_predictions'] += 1
                
                results['warnings'].extend(validation['warnings'])
        
        return results
    
    def get_position_bounds(self, position: str) -> Optional[Dict]:
        """Get bounds information for a position."""
        return self.FANTASY_BOUNDS.get(position)
    
    def get_stat_bounds(self, stat_name: str) -> Optional[Dict]:
        """Get bounds information for a stat."""
        return self.STAT_BOUNDS.get(stat_name)
    
    def log_validation_summary(self, validation_result: Dict):
        """Log a summary of validation results."""
        total = validation_result['total_predictions']
        valid = validation_result['valid_predictions']
        typical = validation_result['typical_predictions']
        corrections = validation_result['corrections_made']
        
        logger.info(f"Prediction Validation Summary:")
        logger.info(f"  Total predictions: {total}")
        logger.info(f"  Valid predictions: {valid} ({valid/total*100:.1f}%)")
        logger.info(f"  Typical predictions: {typical} ({typical/total*100:.1f}%)")
        logger.info(f"  Corrections made: {corrections}")
        
        if validation_result['warnings']:
            logger.warning(f"  Validation warnings: {len(validation_result['warnings'])}")
