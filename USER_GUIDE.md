# NFL Betting Analyzer - User Guide

Welcome to the NFL Betting Analyzer. This guide helps you set up the environment, fetch data, train models, and generate predictions.

## Quick Start

1. Install Python 3.9+ and required packages from `requirements.txt`.
2. Run the CLI to fetch data and verify snapshots:
   - `python nfl_cli.py fetch --season 2025 --week 1`
   - `python nfl_cli.py snapshot-verify`
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

## Troubleshooting

- Ensure snapshot headers exist (the system writes headers even when no rows are present).
- Verify database tables are created automatically when first opening a session.
- Use `python nfl_cli.py models-status` to inspect saved models and performance.
