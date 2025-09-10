# KEEP / REFACTOR / REMOVE-ARCHIVE Plan (Phase A Baseline)

This plan adheres to the HARD RULES:
- Reuse-only; no parallel `_v2/_new` files
- Single API at `api/app.py`, single CLI at `nfl_cli.py`
- Business logic canonical under `core/*`
- New code must be utilized by CLI/API/tests; remove unused code

Audit evidence (audit_report.md): Python modules = 50, Data files = 42, Duplicate bodies = 2 groups, `__main__` files = 1.

## KEEP (as-is or with minimal changes)

- `nfl_cli.py` — canonical CLI
- `api/app.py` — canonical FastAPI app
- `core/` business logic modules:
  - `core/database_models.py`, `core/prediction_bounds.py`
  - `core/models/feature_engineering.py`, `core/models/prediction_models.py`, `core/models/streamlined_models.py`
  - `core/data/data_foundation.py`, `core/data/data_collector.py`, `core/data/enhanced_data_collector.py`
  - `core/data/ingestion_adapters.py`, `core/data/odds_snapshot.py`, `core/data/market_mapping.py`
  - `core/data/data_validator.py`, `core/data/data_processing_pipeline.py`, `core/data/data_validation_pipeline.py`
- Tests under `tests/`
- Config under `config/`

## REFACTOR (fold into `core/*` and wire via CLI/API; no feature changes)

- Root duplicates (status):
  - `feature_engineering.py` → unique helpers folded into `core/models/feature_engineering.py` (AdvancedFeatureEngineer, FeatureConfig ported); imports switched to `core.models.feature_engineering`; root file removed (2025-09-09) after green tests.
  - `database_models.py` → canonical `core/database_models.py` in use; root shim removed (2025-09-09).
  - `data_collector.py` → canonical `core/data/data_collector.py` in use; root shim removed (2025-09-09).
  - `data_foundation.py` → canonical `core/data/data_foundation.py` in use; root shim removed (2025-09-09).

- Modeling orchestration (retain, but ensure canonical imports):
  - `ml_models.py`, `model_evaluation.py`, `prediction_pipeline.py`, `validation_backtesting_framework.py` → keep for now; ensure they import from `core/*`. Defer removal until their unique logic is folded or replaced by CLI flows.

## REMOVE / ARCHIVE (only after refactor imports swapped and tests green)

- Root duplicates listed above (four files) — REMOVED (2025-09-09); verified imports point to `core/*`; tests green (86 passed, 11 skipped).
- Empty model subdirectories: `models/advanced/`, `models/comprehensive/`, `models/enhanced/`, `models/final/` — keep or remove at maintainer discretion; they contain no code and are not used by gating.
- No `advanced_*` Python modules are present in the repo root; N/A.

## Duplicate Groups — Canonical Targets

- Test helpers duplicated across schema tests (see audit groups) — acceptable; test-local duplication, no action.
- All imports that previously referenced root duplicates must be updated to `core/*`.

## __main__ Elimination Plan

- Ensure no non-test `__main__` remain; route via `nfl_cli.py`.
- Verify `uvicorn api.app:app` and `python nfl_cli.py --help`.

## Data & Models

- Gating path: `models/streamlined/*.pkl` (required by API/CLI)
- Snapshots under `data/snapshots/YYYY-MM-DD/`

## Acceptance Gates for Phase A

- Docs reflect current repo (counts from audit, duplicates enumerated)
- Exact KEEP/REFACTOR/REMOVE for root duplicates recorded
- Advanced modules section resolved (N/A); model subdirs noted
