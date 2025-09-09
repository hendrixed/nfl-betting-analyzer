# KEEP / REFACTOR / REMOVE-ARCHIVE Plan (Phase 0)

This plan adheres to the non-negotiable rules:
- Reuse-over-rewrite
- Single API at `api/app.py`, Single CLI at `nfl_cli.py`
- All new functions/classes must be used
- No dead ends, no parallel v2 files

## KEEP (as-is or with minimal changes)

- `nfl_cli.py` — canonical CLI. Commands present: `fetch`, `features`, `train`, `project`, `sim`, `backtest`, `run-api`, `status`.
- `api/app.py` — canonical and unified FastAPI app (health, players, predictions, betting, sim, models, web, websocket, rate limits, caching hooks).
- `core/` (canonical business logic)
  - `core/database_models.py`
  - `core/prediction_bounds.py`
  - `core/models/feature_engineering.py`
  - `core/models/prediction_models.py`
  - `core/models/streamlined_models.py`
  - `core/data/data_foundation.py`
  - `core/data/enhanced_data_collector.py` (as primary input for Phase 2 adapters)
  - `core/data/ingestion_adapters.py` (as the adapter home for Phase 2)
  - `core/data/data_validator.py` and `core/data/data_validation_pipeline.py`
- Tests under `tests/` — will be expanded with schema tests in Phase 2.
- Config under `config/`

## REFACTOR (merge into core/* and/or wire into CLI/API)

- Advanced/overlapping feature/model code
  - `advanced_feature_engineering.py` → fold reusable functions into `core/models/feature_engineering.py`.
  - `feature_engineering.py` (legacy) → migrate unique logic into `core/models/feature_engineering.py`; remove duplicate constructs.
  - `advanced_ml_models.py` and `ml_models.py` → unify patterns and metadata saving under `core/models/prediction_models.py` and `core/models/streamlined_models.py`.
  - `model_evaluation.py` → keep evaluation utilities; expose via `nfl_cli.py backtest` and Phase 4 artifacts.
- System orchestrators (reduce to CLI/API calls)
  - `prediction_pipeline.py` → provide callable utilities that `nfl_cli.py` can invoke; remove `__main__`.
  - `validation_backtesting_framework.py` → extract reusable routines under `core/models/` and call from `nfl_cli.py backtest`.
- Analytical/duplicative modules
  - `comprehensive_matchup_analyzer.py`, `comprehensive_stats_engine.py`, `opponent_defensive_analysis.py`, `team_matchup_analyzer.py` → migrate unique, actively-used feature creation to `core/models/feature_engineering.py`; remove overlapping reporting code.
  - `player_comparison_optimizer.py` → extract any reusable math/optimizer helpers to `core/models/prediction_models.py` (or utils); drop UI/demo scaffolding.
- Real-time integrations
  - `real_time_data_integration.py`, `real_time_market_data.py`, `injury_data_integration.py` → convert data acquisition into Phase 2 adapters under `core/data/ingestion_adapters.py`; wire summaries to CLI/API.
- Logging
  - `enhanced_logging.py` → integrate structured logging where relevant; remove unused hooks.

## REMOVE / ARCHIVE (after verification; no loss of unique logic)

**ALREADY REMOVED BY CLAUDE:**
- ✅ `backup_*/` directories — REMOVED
- ✅ `models_backup/*` — REMOVED  
- ✅ `api/prediction_api.py` — REMOVED
- ✅ `api/enhanced_prediction_api.py` — REMOVED
- ✅ `api/web_app.py` — REMOVED
- ✅ `nfl_browser.py` — REMOVED
- ✅ `web/web_server.py` — REMOVED
- ✅ `start_nfl_system.py` — REMOVED
- ✅ `scripts/*` directory — REMOVED

**STILL PRESENT (need cleanup):**
- Root level duplicates:
  - `data_collector.py` (duplicate of core/data/data_collector.py)
  - `data_foundation.py` (duplicate of core/data/data_foundation.py) 
  - `data_validator.py` (duplicate of core/data/data_validator.py)
  - `database_models.py` (duplicate of core/database_models.py)
  - `feature_engineering.py` (duplicate of core/models/feature_engineering.py)

## Duplicate Groups — Canonical Targets

- `advanced_ml_models.py` vs `player_comparison_optimizer.py` initializers → canonical in `core/models/prediction_models.py`.
- `nfl_browser.py` vs `start_nfl_system.py:find_available_port` → remove both implementations; use `uvicorn` via `nfl_cli.py run-api`.
- `verify_production_readiness.py` vs backup copy → keep root, drop backup.
- `get_db` variations across API files → canonical in `api/app.py:get_db`.

All 16 duplicate groups listed in `audit_report.json` will be resolved by the above.

## __main__ Elimination Plan

- Remove all non-test `__main__` blocks and route functionality via `nfl_cli.py`.
- Confirm `nfl_cli.py --help` exposes all canonical commands.
- Confirm `uvicorn api.app:app` boots and `/health` returns 200; `/docs` loads.

## Data & Models

- Keep `models/` active artifacts. Remove `models_backup/*`.
- Centralize snapshots to `data/snapshots/YYYY-MM-DD/` in Phase 2 with schema tests.

## Acceptance Gates for Phase 0

- Inventory reflects ~80 modules and ~56 data files (active) — confirmed by audit scan and listings.
- Duplicate groups enumerated with canonical targets — mapped above.
- Concrete plan to eliminate extra `__main__` scripts — above, via CLI/API canonical paths.
