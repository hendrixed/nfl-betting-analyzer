"""
NFL Betting Analyzer - Core Data Module
Data collection, validation, and processing components.
"""

# Import only modules that don't have heavy external dependencies
try:
    # Backward-compatible alias: expose DataQualityValidator as DataValidator
    from .data_validator import DataQualityValidator as DataValidator, StatsValidator
except ImportError:
    pass

try:
    from .nfl_2025_data_collector import NFL2025DataCollector
except ImportError:
    pass
