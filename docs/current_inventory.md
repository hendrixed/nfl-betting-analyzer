# Current Inventory - NFL Betting Analyzer

## Fresh Baseline (2025-09-09 16:31)

After Claude's consolidation commits, current repository state:

**ENTRYPOINTS**: 
- Single unified API: `api/app.py` 
- Single unified CLI: `nfl_cli.py` 
- Legacy APIs: REMOVED 

**CLEANUP STATUS**:
- Duplicates: 3 function/method groups (reduced from 18)
- Unused Definitions: 202 possibly unused definitions (reduced from 214)
- Files with __main__: 1 remaining (down from 3)

## Data Files Present (33 total)

### Snapshots (data/snapshots/2025-09-09/)
- `reference_teams.csv`
- `reference_stadiums.csv` 
- `reference_players.csv`
- `rosters.csv`
- `depth_charts.csv`
- `injuries.csv`
- `schedules.csv`
- `snaps.csv`
- `pbp.csv`
- `weather.csv`
- `odds.csv`

### Models Directory
- Multiple .pkl files in models/ subdirectories
- Model artifacts for different positions and markets

### Other Data
- Database files (.db)
- Configuration files (.yaml, .env)

## Python Modules Present (89 total)

### CORE ENTRYPOINTS (KEEP)
- `api/app.py` - Unified FastAPI server (784 LOC)
- `nfl_cli.py` - Unified Typer CLI

### CORE ARCHITECTURE (KEEP)
- `core/database_models.py` - SQLAlchemy models
- `core/models/feature_engineering.py` - Feature pipeline
- `core/models/prediction_models.py` - ML models
- `core/data/ingestion_adapters.py` - Data adapters
- `core/data/market_mapping.py` - Market normalization

### DATA PIPELINE (KEEP)
- `core/data/data_collector.py`
- `core/data/data_foundation.py` 
- `core/data/data_validator.py`
- `core/data/enhanced_data_collector.py`
- `core/data/odds_snapshot.py`

### Data Processing (25 modules)
- `core/data/data_collector.py` - Enhanced data collection
- `core/data/data_foundation.py` - Foundation data structures
- `core/data/odds_snapshot.py` - Odds data handling

### Model Training & Evaluation (15 modules)
- `core/models/feature_engineering.py` - Feature engineering
- `core/models/streamlined_models.py` - Streamlined ML models
- `core/models/prediction_bounds.py` - Prediction validation

### Testing Framework (25 modules)
- `tests/` - Comprehensive test suite (87 passing tests)
- Schema validation, feature engineering, model training tests

### Configuration & Utilities (9 modules)
- `config/config_manager.py` - Configuration management
- `config/config.yaml` - System configuration
- Various utility scripts and helpers

## Data Assets

### Processed Data (33 files)
- `data/snapshots/2025-09-09/` - Current snapshot data
- `data/processed/` - Processed datasets
- `data/features/` - Feature matrices
- `models/` - Trained model artifacts (33 .pkl files)

### Reports & Coverage
- `reports/coverage/` - Coverage matrices and validation
- `reports/backtests/` - Backtest results and metrics

## Removed During Cleanup (Phase 6)
- 30+ legacy files including duplicates and unused modules
- Broken test files referencing deleted modules
- Obsolete configuration and backup files

- core/
  - `core/__init__.py`
  - data/
    - `core/data/__init__.py`
    - `core/data/data_collector.py` (loc=662)
    - `core/data/data_foundation.py` (loc=289)
    - `core/data/data_processing_pipeline.py` (loc=459)
    - `core/data/data_validation_pipeline.py` (loc=505)
    - `core/data/data_validator.py` (loc=425)
    - `core/data/enhanced_data_collector.py` (loc=685)
    - `core/data/ingestion_adapters.py` (loc=678)
    - `core/data/nfl_2025_data_collector.py` (loc=391)
    - `core/data/statistical_computing_engine.py` (loc=541)
  - models/
    - `core/models/__init__.py`
    - `core/models/feature_engineering.py` (loc=955)
    - `core/models/prediction_models.py` (loc=783)
    - `core/models/streamlined_models.py` (loc=485)
  - `core/prediction_bounds.py` (loc=167)
  - `core/database_models.py` (loc=419)

- tests/
  - `tests/test_complete_system.py` (loc=350)
  - `tests/test_data_validation.py` (loc=137)
  - `tests/test_enhanced_realtime.py` (loc=295)
  - `tests/test_enhanced_system.py` (loc=280)
  - `tests/test_feature_engineering.py` (loc=534)
  - `tests/test_feature_engineering_simple.py` (loc=163)
  - `tests/test_historical_standardizer.py` (loc=254)
  - `tests/test_prediction_models.py` (loc=481)
  - `tests/test_system.py` (loc=370)

- web/
  - `web/__init__.py` (loc=4)
  - `web/web_server.py` (loc=254)

- scripts/
  - `scripts/backtest.py`
  - `scripts/download_initial_data.py`
  - `scripts/fetch.py`
  - `scripts/make_predictions.py`
  - `scripts/project.py`
  - `scripts/run_api.py`
  - `scripts/run_pipeline.py`
  - `scripts/setup_database.py`
  - `scripts/simulate.py`
  - `scripts/start_api.py`
  - `scripts/train.py`
  - `scripts/train_models.py`

- config/
  - `config/config_manager.py` (loc=431)

- top-level modules (selection)
  - `nfl_cli.py` (loc=591) [Typer]
  - `audit_repo.py` (loc=362)
  - `advanced_analytics.py` (loc=661)
  - `advanced_feature_engineering.py` (loc=507)
  - `advanced_game_context.py` (loc=451)
  - `advanced_market_analytics.py` (loc=499)
  - `advanced_ml_models.py` (loc=512)
  - `advanced_situational_analytics.py` (loc=551)
  - `api_integrations.py` (loc=643)
  - `comprehensive_matchup_analyzer.py` (loc=624)
  - `comprehensive_stats_engine.py` (loc=556)
  - `deploy_production.py` (loc=345)
  - `enhanced_logging.py` (loc=445)
  - `expand_data_coverage.py` (loc=407)
  - `feature_engineering.py` (loc=806)
  - `fix_player_count.py` (loc=95)
  - `historical_data_standardizer.py` (loc=759)
  - `historical_trend_analyzer.py` (loc=625)
  - `historical_validation_lite.py` (loc=353)
  - `injury_data_integration.py` (loc=367)
  - `load_nfl_schedule.py` (loc=324)
  - `ml_models.py` (loc=923)
  - `model_evaluation.py` (loc=826)
  - `nfl_browser.py` (loc=454)
  - `opponent_defensive_analysis.py` (loc=479)
  - `player_comparison_optimizer.py` (loc=606)
  - `player_identity_resolver.py` (loc=651)
  - `prediction_pipeline.py` (loc=831)
  - `real_time_data_integration.py` (loc=537)
  - `real_time_market_data.py` (loc=400)
  - `run_full_standardization.py` (loc=273)
  - `social_sentiment_analyzer.py` (loc=450)
  - `start_nfl_system.py` (loc=208)
  - `stat_terminology_mapper.py` (loc=286)
  - `team_matchup_analyzer.py` (loc=421)
  - `validate_system_fixes.py` (loc=256)
  - `validation_backtesting_framework.py` (loc=681)
  - `verify_environment.py` (loc=104)
  - `verify_production_readiness.py` (loc=651)

Note: Backups under `backup_*/` are excluded from the active module list, but are covered in duplicate and removal plans below.

## Data Files (active; excludes models_backup/ and backup_*/)

- data/
  - `data/market_data.db`
  - `data/nfl_predictions.db`
  - `data/sentiment_data.db`

- models/performance/
  - `models/performance/QB_fantasy_points_model_performance.json`
  - `models/performance/QB_passing_touchdowns_model_performance.json`
  - `models/performance/QB_passing_yards_model_performance.json`
  - `models/performance/RB_fantasy_points_model_performance.json`
  - `models/performance/RB_receiving_yards_model_performance.json`
  - `models/performance/RB_rushing_touchdowns_model_performance.json`
  - `models/performance/RB_rushing_yards_model_performance.json`
  - `models/performance/TE_fantasy_points_model_performance.json`
  - `models/performance/TE_receiving_yards_model_performance.json`
  - `models/performance/TE_receptions_model_performance.json`
  - `models/performance/WR_fantasy_points_model_performance.json`
  - `models/performance/WR_receiving_touchdowns_model_performance.json`
  - `models/performance/WR_receiving_yards_model_performance.json`
  - `models/performance/WR_receptions_model_performance.json`

- models/ (root)
  - `models/QB_fantasy_points_ppr_basic.pkl`
  - `models/QB_passing_yards_basic.pkl`
  - `models/RB_fantasy_points_ppr_basic.pkl`
  - `models/RB_receiving_yards_basic.pkl`
  - `models/RB_rushing_yards_basic.pkl`
  - `models/TE_fantasy_points_ppr_basic.pkl`
  - `models/TE_receiving_yards_basic.pkl`
  - `models/TE_rushing_yards_basic.pkl`
  - `models/WR_fantasy_points_ppr_basic.pkl`
  - `models/WR_receiving_yards_basic.pkl`
  - `models/WR_rushing_yards_basic.pkl`

- models/streamlined/
  - `models/streamlined/QB_fantasy_points_ppr_ridge.pkl`
  - `models/streamlined/QB_fantasy_points_ppr_streamlined.pkl`
  - `models/streamlined/QB_passing_attempts_streamlined.pkl`
  - `models/streamlined/QB_passing_touchdowns_streamlined.pkl`
  - `models/streamlined/QB_passing_yards_streamlined.pkl`
  - `models/streamlined/RB_fantasy_points_ppr_ridge.pkl`
  - `models/streamlined/RB_fantasy_points_ppr_streamlined.pkl`
  - `models/streamlined/RB_rushing_attempts_streamlined.pkl`
  - `models/streamlined/RB_rushing_touchdowns_streamlined.pkl`
  - `models/streamlined/RB_rushing_yards_streamlined.pkl`
  - `models/streamlined/TE_fantasy_points_ppr_ridge.pkl`
  - `models/streamlined/TE_fantasy_points_ppr_streamlined.pkl`
  - `models/streamlined/TE_receiving_touchdowns_streamlined.pkl`
  - `models/streamlined/TE_receiving_yards_streamlined.pkl`
  - `models/streamlined/TE_receptions_streamlined.pkl`
  - `models/streamlined/TE_targets_streamlined.pkl`
  - `models/streamlined/WR_fantasy_points_ppr_streamlined.pkl`
  - `models/streamlined/WR_receiving_touchdowns_streamlined.pkl`
  - `models/streamlined/WR_receiving_yards_streamlined.pkl`
  - `models/streamlined/WR_receptions_streamlined.pkl`
  - `models/streamlined/WR_targets_streamlined.pkl`

- root and tests (active DBs)
  - `nfl_predictions.db`
  - `test_enhanced_nfl.db`
  - `test_enhanced_realtime.db`
  - `test_historical_standardizer.db`

- Other assets
  - `audit_report.json`
  - `api_demo.html`

Approximate active data file count (excluding backups and archives): ~56

## Duplicate Groups (from audit) and Canonical Selection

- Hash b67a381c55…
  - Duplicates: `advanced_ml_models.py:ModelComparison.__init__`, `player_comparison_optimizer.py:AdvancedAnalytics.__init__`
  - Canonical: Refactor into `core/models/prediction_models.py` and remove duplicated initializers in advanced modules.

- 11a101f24a…
  - Duplicates: `nfl_browser.py:find_available_port`, `start_nfl_system.py:find_available_port`
  - Canonical: Remove standalone copies; we will rely on `nfl_cli.py run-api` (uvicorn) and drop custom port finders.

- a19e229c0b… through 085aca5ebf… (series in `verify_production_readiness.py` and `backup_20250907_212630/verify_production_readiness.py`)
  - Canonical: Keep `verify_production_readiness.py`; mark the backup copy for archive/removal.

- 743f6aeb3c… (`get_db` duplicates)
  - Duplicates: `api/app.py:get_db`, `api/enhanced_prediction_api.py:get_db`, `api/prediction_api.py:get_db`, `api/web_app.py:get_db`
  - Canonical: `api/app.py:get_db`. Old API files will be removed after tests pass.

Note: The full duplicate list (16 groups) is preserved in `audit_report.json`; all groups will be resolved by consolidating to the chosen canonical targets above, preferring implementations under `core/` and the unified `api/app.py`.

## Files with __main__ and Consolidation Plan

Primary non-test `__main__` entrypoints:
- `audit_repo.py` — Plan: callable via `nfl_cli.py status` or a future `nfl_cli.py project --audit` option; remove `__main__`.
- `nfl_browser.py` — Plan: retire; overlapping with FastAPI web endpoints in `api/app.py` and `web/templates/`.
- `web/web_server.py` — Plan: retire; web routes now served by `api/app.py` (`/web`).

Additional scripts under `scripts/` contain minimal `__main__` wrappers that already defer to `nfl_cli.py` commands. Plan: remove these after confirming all pathways exist in `nfl_cli.py` and CI/tests are updated.

Test files with `__main__` (to allow direct execution) do not count as entrypoints; no action required beyond standardizing on `pytest`.

## Notes

- A unified API already exists at `api/app.py` and includes: health, players, predictions, betting insights/props, simulations, models, a web landing page, and WebSocket support.
- The single CLI already exists at `nfl_cli.py` and exposes: `fetch`, `features`, `train`, `project`, `sim`, `backtest`, `run-api`, `status`.
- Old API modules remain present and will be removed in Phase 1 after verification.
