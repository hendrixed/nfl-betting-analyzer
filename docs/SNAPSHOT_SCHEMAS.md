# Snapshot CSV Schemas

This document defines the minimal, canonical headers for snapshot CSVs written to `data/snapshots/YYYY-MM-DD/`.

These headers are enforced by:
- CLI command: `nfl_cli.py snapshot-verify` (uses `--repair` to autofix headers)
- Central mapping: `core/data/ingestion_adapters.py` → `SNAPSHOT_MIN_COLUMNS`

Keep this document and the code mapping in sync. Tests assert these exact headers for some files.

## Foundation

- schedules.csv
```
["game_id","season","week","season_type","home_team","away_team","kickoff_dt_utc","kickoff_dt_local","network","spread_close","total_close","officials_crew","stadium","roof_state"]
```

- rosters.csv
```
["player_id","name","position","team","jersey_number","status","depth_chart_rank","snap_percentage","last_updated"]
```

## Weekly Snapshots

- weekly_stats.csv
```
["player_id","game_id","week","season","team","opponent","position","passing_attempts","passing_completions","passing_yards","passing_touchdowns","interceptions","rushing_attempts","rushing_yards","rushing_touchdowns","targets","receptions","receiving_yards","receiving_touchdowns","offensive_snaps","snap_percentage"]
```

- snaps.csv
```
["player_id","game_id","team","position","offense_snaps","defense_snaps","st_snaps","offense_pct","defense_pct","st_pct"]
```

- pbp.csv
```
["play_id","game_id","offense","defense","play_type","epa","success","air_yards","yac","pressure","blitz","personnel","formation"]
```

- weather.csv
```
["game_id","stadium","temperature","humidity","wind_speed","wind_direction","precipitation","conditions","timestamp"]
```

- depth_charts.csv
```
["team","player_id","player_name","position","slot","role","package","depth_chart_rank","last_updated"]
```

- injuries.csv
```
["player_id","name","team","position","practice_status","game_status","designation","report_date","return_date"]
```

## Odds

- odds.csv
```
["timestamp","book","market","player_id","team_id","line","over_odds","under_odds"]
```

- odds_history.csv
```
["ts_utc","book","market","selection_id","line","price","event_id","is_closing"]
```

## Reference (for tests and tooling)

- teams.csv
```
["team_id","abbr","conference","division","coach","home_stadium_id"]
```

- stadiums.csv
```
["stadium_id","name","city","state","lat","lon","surface","roof","elevation"]
```

- players.csv
```
["player_id","name","birthdate","age","position","team","height_inches","weight_lbs","dominant_hand","draft_year","draft_round","draft_pick"]
```

## Notes

- Files may be header-only (zero data rows). Tests accept header-only files.
- Additional snapshot files may exist; only files listed here are subject to header enforcement in tests/CLI.
- `snapshot-verify --repair` will create or rewrite files with header-only content using these exact headers.
