# Test Playbook

This playbook describes how to run snapshot verification, coverage generation, API contract tests, and backtests locally and in CI.

## Prerequisites

- Environment: `conda activate nfl`
- Install deps: `pip install -r requirements.txt`

## Quick Verification

- Run the full verification pipeline (prepares artifacts, runs tests, lint, mypy):

```bash
make verify
```

Artifacts produced:
- `data/snapshots/YYYY-MM-DD/odds.csv` (mock odds)
- `reports/coverage/*.csv` and `coverage_validation.json`
- `reports/backtests/player_receiving_yards/metrics.json` and `calibration.png`
- `reports/junit.xml`

## Snapshot Verification

- Validate the latest (or a specific) snapshot and auto-repair headers if needed:

```bash
python nfl_cli.py odds-snapshot --provider mock --max-offers 200
python nfl_cli.py snapshot-verify --repair
```

See canonical headers in `docs/SNAPSHOT_SCHEMAS.md`. The minimal schemas are also mirrored in `core/data/ingestion_adapters.py:SNAPSHOT_MIN_COLUMNS`.

## Coverage Matrices

- Generate coverage matrices and validation report:

```bash
python generate_coverage_matrices.py
```

Outputs under `reports/coverage/`:
- `stats_feature_matrix.csv`
- `model_market_matrix.csv`
- `coverage_validation.json`

## Models

- Check if streamlined models are present:

```bash
python nfl_cli.py models-status
```

- Train models (if needed):

```bash
python nfl_cli.py train --target fantasy_points_ppr
```

## Backtests (Real Metrics)

- Create mock odds snapshot (for lines):

```bash
python nfl_cli.py odds-snapshot --provider mock --max-offers 200
```

- Run backtest using predictions vs. closing lines when available:

```bash
python nfl_cli.py backtest \
  --target receiving_yards \
  --market player_receiving_yards \
  --start 2024-10-01 \
  --end 2025-01-31 \
  --limit 300
```

Outputs under `reports/backtests/player_receiving_yards/`:
- `metrics.json`
- `calibration.png`

## API Contract Tests

- Run pytest (junit report saved to `reports/junit.xml`):

```bash
pytest -q --junitxml reports/junit.xml
```

## CI

- GitHub Actions uses `make verify` to prepare artifacts and run tests. Artifacts to publish:
  - `reports/coverage/*`
  - `reports/junit.xml`
  - `reports/backtests/**/metrics.json`
  - `reports/backtests/**/calibration.png`
