"""
NFL Betting Analyzer - Core Data Module
Data collection, validation, and processing components.
"""

# Import only modules that don't have external dependencies
try:
    from .data_validator import DataValidator
except ImportError:
    pass

try:
    from .nfl_2025_data_collector import NFL2025DataCollector
except ImportError:
    pass
