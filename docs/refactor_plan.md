# NFL Betting Analyzer Refactor Plan

## Status Update (2025-09-09 local)

- Canonicalization (Phase B Steps 1–3) completed:
  - All imports now use `core/*`.
  - Unique legacy feature engineering interfaces (`AdvancedFeatureEngineer`, `FeatureConfig`) ported into `core/models/feature_engineering.py`.
  - Root duplicates removed: `database_models.py`, `data_collector.py`, `data_foundation.py`, `feature_engineering.py`.
  - Evidence: `pytest -q` => 86 passed, 11 skipped, 3 warnings.
- Hard Rule alignment:
  - Single entrypoints retained: `api/app.py`, `nfl_cli.py`.
  - Model gating aligned to `models/streamlined/*.pkl` in API (`/health`, predictions) and CLI (`project`, `sim`).
  - Snapshot verification (`nfl_cli.py snapshot-verify`) now fails on missing files/headers.

This plan adheres to HARD RULES and focuses on Phase B canonicalization with a concise 5-commit sequence to fold logic into `core/*` without feature changes.

## Phase B — Canonicalization: 5-commit sequence

1) Canonicalize Imports (repo-wide)
   - Replace all imports from root duplicates to `core/*`:
     - `feature_engineering` → `core.models.feature_engineering`
     - `database_models` → `core.database_models`
     - `data_collector` → `core.data.data_collector`
     - `data_foundation` → `core.data.data_foundation`
   - Ensure `api/app.py` and `nfl_cli.py` only import from `core/*`.
   - Evidence: grep diff showing only `core/*` imports; `pytest -q` green.

2) Fold Unique Logic from Root Duplicates
   - Compare root-level files to canonical `core/*` and copy any unique helpers/docstrings:
     - `feature_engineering.py` → merge helpers into `core/models/feature_engineering.py` (append-only, no behavior change).
     - `database_models.py` → verify schema parity with `core/database_models.py`; migrate any new fields.
     - `data_collector.py`, `data_foundation.py` → ensure parity with `core/data/*`; merge missing enums/validators.
   - Evidence: short code diff snippets embedded in commit; `pytest -q` green.

3) Delete Root Duplicates (post-green)
   - Remove root: `feature_engineering.py`, `database_models.py`, `data_collector.py`, `data_foundation.py`.
   - Update docs and references where necessary (none expected after step 1).
   - Evidence: tree snapshot, `pytest -q` green; imports only from `core/*`.

4) Enforce Model Gating Everywhere
   - Verify API and CLI both require `models/streamlined/*.pkl`:
     - `api/app.py` → `models_available()` gate already checks `models/streamlined`.
     - `nfl_cli.py` → ensure projection/prediction paths check presence of `models/streamlined/*.pkl` (already present for projections/fantasy).
   - Add a lightweight check in any code paths still missing gating.
   - Evidence: `/health` reflects `models_loaded`, CLI `project` fails gracefully when missing; `tests/test_api_contract.py` to be added in later phase.

5) Final Canonicalization Sweep
   - Run repo-wide search to ensure no remaining imports from root duplicates.
   - Remove stale references in docs where applicable; confirm only single API/CLI remain.
   - Evidence: grep results attached in commit; `pytest -q` green.

Acceptance for Phase B: tests green after each commit; only `core/*` imports remain.

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
