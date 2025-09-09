import os
import csv
import glob
from pathlib import Path


def latest_snapshot_dir() -> Path:
    candidates = [Path(p) for p in glob.glob("data/snapshots/*") if Path(p).is_dir()]
    assert candidates, "No snapshot directory found under data/snapshots/"
    # Sort by name desc works for YYYY-MM-DD
    return sorted(candidates)[-1]


def read_header(csv_path: Path):
    assert csv_path.exists(), f"Missing snapshot file: {csv_path}"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            # Allow empty file; header-less files should fail schema
            header = []
    return header


def test_reference_teams_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "teams.csv")
    expected = [
        "team_id","abbr","conference","division","coach","home_stadium_id"
    ]
    assert header == expected


def test_reference_stadiums_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "stadiums.csv")
    expected = [
        "stadium_id","name","city","state","lat","lon","surface","roof","elevation"
    ]
    assert header == expected


def test_reference_players_schema():
    snap = latest_snapshot_dir()
    header = read_header(snap / "players.csv")
    expected = [
        "player_id","name","birthdate","age","position","team",
        "height_inches","weight_lbs","dominant_hand","draft_year","draft_round","draft_pick"
    ]
    assert header == expected
