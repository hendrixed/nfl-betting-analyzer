# NFL Betting Analyzer Refactor Plan

Based on the audit report analysis, this document outlines the KEEP/REFACTOR/REMOVE decisions for the repository refactoring.

## Summary of Current State
- **80 Python modules** with significant duplication and fragmentation
- **16 groups of duplicate function bodies** 
- **176 potentially unused definitions**
- **4 files with `if __name__ == '__main__'`** (multiple CLI entrypoints)
- **3 FastAPI applications** that need consolidation
- Multiple backup directories with redundant code

## KEEP (Core Components to Preserve)

### Core Data Infrastructure
- `core/data/data_foundation.py` - Essential data structures (MasterPlayer, WeeklyRosterSnapshot, etc.)
- `core/data/enhanced_data_collector.py` - Enhanced NFL data collection with role-based stats
- `core/data/data_validator.py` - Comprehensive validation logic
- `core/database_models.py` - SQLAlchemy models (consolidate with enhancements)

### Core Models
- `core/models/feature_engineering.py` - Feature engineering pipeline
- `core/models/prediction_models.py` - Core prediction models
- `core/models/streamlined_models.py` - Streamlined model implementations

### Configuration & Logging
- `config/config_manager.py` - Centralized configuration management
- `enhanced_logging.py` - Structured logging system
- `.env` and `requirements.txt` - Environment configuration

### Historical Data Components
- `historical_data_standardizer.py` - Data standardization (from memory)
- `stat_terminology_mapper.py` - Statistical naming consistency
- `player_identity_resolver.py` - Player identity resolution

## REFACTOR (Consolidate and Modernize)

### API Consolidation
**Target**: Single FastAPI app at `api/app.py`
**Merge from**:
- `api/prediction_api.py` (770 LOC) - Core prediction endpoints
- `api/enhanced_prediction_api.py` (761 LOC) - Enhanced features, WebSocket support
- `api/web_app.py` (365 LOC) - Web interface endpoints

**Key endpoints to preserve**:
- GET /health, /players/{id}, /games/{id}
- GET /props (market analysis)
- POST /sim/run (Monte Carlo simulation)
- WebSocket support for real-time updates

### CLI Consolidation
**Target**: Single `nfl_cli.py` with Typer
**Current CLI files to merge**:
- `nfl_cli.py` (485 LOC) - Existing CLI with multiple commands
- `backup_20250907_212630/production_cli.py` (453 LOC) - Production CLI features

**Commands to implement**:
- fetch, features build, train, project, sim, backtest, run-api

### Data Processing Pipeline
**Target**: Streamlined `core/data/` modules
**Consolidate**:
- `core/data/data_processing_pipeline.py` (459 LOC)
- `core/data/data_validation_pipeline.py` (505 LOC)
- Multiple data collectors into unified interface

### Model Infrastructure
**Target**: Unified model system
**Consolidate**:
- `ml_models.py` (923 LOC) - Multiple model implementations
- `advanced_ml_models.py` (512 LOC) - Advanced features
- `prediction_pipeline.py` (831 LOC) - Pipeline orchestration

### Analytics Modules
**Consolidate advanced analytics**:
- `advanced_analytics.py` (661 LOC)
- `advanced_feature_engineering.py` (507 LOC)
- `advanced_game_context.py` (451 LOC)
- `advanced_market_analytics.py` (499 LOC)
- `advanced_situational_analytics.py` (551 LOC)

## REMOVE (Delete or Archive)

### Backup Directories
- `backup_20250907_212630/` - Archive entire directory
- `models_backup/` - Consolidate with `models/performance/`

### Duplicate Files
**Exact duplicates identified in audit**:
- `backup_20250907_212630/verify_production_readiness.py` (duplicate of main file)
- All duplicate function bodies (16 groups) - keep canonical versions

### Legacy/Unused Scripts
- `start_nfl_system.py` (208 LOC) - Replace with CLI
- `nfl_browser.py` (454 LOC) - Flask-based browser (superseded by FastAPI)
- `deploy_production.py` (345 LOC) - Ad-hoc deployment script
- Multiple test files with overlapping functionality

### Potentially Unused Classes (176 identified)
**High-confidence removals**:
- `APIConfig`, `DatabaseConfig` (replaced by centralized config)
- `TestSystemImports`, `TestDatabaseModels` (redundant test classes)
- Multiple analyzer classes with no references

### Ad-hoc Scripts
- `fix_player_count.py` (95 LOC)
- `expand_data_coverage.py` (407 LOC)
- `load_nfl_schedule.py` (324 LOC)
- Various validation and verification scripts

## TARGET STRUCTURE

```
nfl-betting-analyzer/
├── core/
│   ├── data/
│   │   ├── data_collector.py           # Unified data collection
│   │   ├── enhanced_data_collector.py  # Role-based collection
│   │   ├── data_processing_pipeline.py # Processing pipeline
│   │   ├── data_validation_pipeline.py # Validation pipeline
│   │   ├── data_validator.py           # Validation logic
│   │   └── data_foundation.py          # Core data structures
│   ├── models/
│   │   ├── feature_engineering.py      # Feature engineering
│   │   ├── prediction_models.py        # Core models
│   │   ├── streamlined_models.py       # Streamlined implementations
│   │   └── prediction_bounds.py        # Bounds validation
│   └── database_models.py              # SQLAlchemy models
├── api/
│   └── app.py                          # Single FastAPI application
├── web/                                # Optional HTML frontend
├── scripts/                            # Thin wrappers only
│   ├── fetch.py
│   ├── train.py
│   ├── project.py
│   ├── simulate.py
│   ├── backtest.py
│   └── run_api.py
├── tests/                              # Consolidated test suite
├── models/                             # Model artifacts
├── config/                             # Configuration
├── nfl_cli.py                          # Single CLI entrypoint
├── enhanced_logging.py                 # Logging system
├── requirements.txt
└── .env
```

## IMPLEMENTATION PRIORITY

### Phase 1: Foundation (Steps 1-3)
1. Create target directory structure
2. Consolidate API endpoints into single FastAPI app
3. Create unified CLI with Typer

### Phase 2: Core Systems (Steps 4-6)
4. Implement unified data ingestion with caching
5. Consolidate feature engineering pipeline
6. Unify model training and prediction systems

### Phase 3: Integration (Steps 7-8)
7. Wire CLI and API with end-to-end tests
8. Implement backtesting framework

### Phase 4: Cleanup (Steps 9-10)
9. Remove duplicates and unused code
10. Documentation and final validation

## QUALITY GATES

- [ ] Single CLI entrypoint (`nfl_cli.py`)
- [ ] Single FastAPI app (`api/app.py`)
- [ ] All tests passing (`pytest -q`)
- [ ] Static analysis clean (`ruff check`, `mypy`)
- [ ] End-to-end pipeline working (fetch → train → project → sim)
- [ ] Backtest report generated
- [ ] No duplicate function bodies
- [ ] Comprehensive documentation

## RISK MITIGATION

1. **Preserve working behavior** - Test each consolidation step
2. **Incremental approach** - Small commits with clear rationales
3. **Backup validation** - Ensure no critical functionality lost
4. **Type safety** - Add type hints throughout refactoring
5. **Configuration preservation** - Maintain existing .env and requirements.txt
