# NFL Betting Analyzer - User Guide

Welcome to the NFL Betting Analyzer. This guide helps you set up the environment, fetch data, train models, and generate predictions.

## Quick Start

1. Install Python 3.9+ and required packages from `requirements.txt`.
2. Run the CLI to fetch data and verify snapshots:
   - `python nfl_cli.py fetch --season 2025 --week 1`
   - `python nfl_cli.py snapshot-verify`
   - `python nfl_cli.py snapshot-schedules --season 2024 --date 2024-01-01`
   - `python nfl_cli.py snapshot-depthcharts --season 2024 --week 1 --date 2024-01-01`
   - `python nfl_cli.py weather-snapshot --days-ahead 14 --date 2024-01-01` (creates `weather.csv` using Open-Meteo)
3. Train models:
   - `python nfl_cli.py train --target fantasy_points_ppr`
4. Generate projections:
   - `python nfl_cli.py project --week 1`

## Components

- Core database models define `Player`, `Game`, and `PlayerGameStats` tables.
- Feature engineering produces rolling averages, matchup context, situational (weather/injury) and advanced metrics.
- Prediction models include LightGBM, XGBoost and Random Forest, with ensemble support.

## Data Snapshots

Snapshots are stored under `data/snapshots/YYYY-MM-DD/` and include `rosters.csv`, `schedules.csv`, `weekly_stats.csv`, `snaps.csv`, `weather.csv`, `depth_charts.csv`, `injuries.csv`, `pbp.csv`, and `odds.csv`.

### Snapshot and Ingestion Commands

- Write schedules snapshot:
  - `python nfl_cli.py snapshot-schedules --season 2024 --date 2024-01-01`

- Write depth charts snapshot:
  - `python nfl_cli.py snapshot-depthcharts --season 2024 --week 1 --date 2024-01-01`

- Ingest schedules from a snapshot into the DB:
  - `python nfl_cli.py schedule-ingest --date 2024-01-01`

- Ingest depth charts from a snapshot into the DB:
  - `python nfl_cli.py depthchart-ingest --date 2024-01-01 --season 2024 --week 1`

- Ingest both schedules and depth charts for a snapshot date:
  - `python nfl_cli.py ingest-snapshot --date 2024-01-01 --season 2024 --week 1`

- Ingest extended weekly components (games metadata and box stats):
  - `python nfl_cli.py extended-ingest --date 2024-01-01`

### Browsing Team Schedule Options

The schedule API endpoints accept additional parameters:

- `include_past` (bool): Include past games as well as future (default: `false`).
- `timezone` (string): IANA timezone name to compute the "today" cutoff (default: `America/Chicago`).

Example:

`GET /api/browse/team/NE/schedule?season=2024&include_past=true&timezone=America/New_York`

## Troubleshooting

- Ensure snapshot headers exist (the system writes headers even when no rows are present).
- Verify database tables are created automatically when first opening a session.
- Use `python nfl_cli.py models-status` to inspect saved models and performance.
