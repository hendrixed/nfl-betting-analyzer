import csv
import glob
from pathlib import Path
from datetime import datetime


def latest_snapshot_dir() -> Path:
    candidates = [Path(p) for p in glob.glob("data/snapshots/*") if Path(p).is_dir()]
    assert candidates, "No snapshot directory found under data/snapshots/"
    return sorted(candidates)[-1]


def read_header(csv_path: Path):
    assert csv_path.exists(), f"Missing snapshot file: {csv_path}"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []
    return header


def test_schedule_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "schedules.csv")
    expected = [
        "game_id","season","week","season_type","home_team","away_team","kickoff_dt_utc","kickoff_dt_local","network","spread_close","total_close","officials_crew","stadium","roof_state"
    ]
    assert header == expected


def test_snaps_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "snaps.csv")
    expected = [
        "player_id","game_id","team","position","offense_snaps","defense_snaps","st_snaps","offense_pct","defense_pct","st_pct"
    ]
    assert header == expected


def test_pbp_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "pbp.csv")
    expected = [
        "play_id","game_id","offense","defense","play_type","epa","success","air_yards","yac","pressure","blitz","personnel","formation"
    ]
    assert header == expected


def test_weather_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "weather.csv")
    expected = [
        "game_id","stadium","temperature","humidity","wind_speed","wind_direction","precipitation","conditions","timestamp"
    ]
    assert header == expected


def test_odds_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "odds.csv")
    expected = [
        "timestamp","book","market","player_id","team_id","line","over_odds","under_odds"
    ]
    assert header == expected
