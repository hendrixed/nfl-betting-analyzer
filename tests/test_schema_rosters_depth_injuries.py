import csv
import glob
from pathlib import Path
from datetime import datetime


def latest_snapshot_dir() -> Path:
    candidates = [Path(p) for p in glob.glob("data/snapshots/*") if Path(p).is_dir()]
    assert candidates, "No snapshot directory found under data/snapshots/"
    return sorted(candidates)[-1]


def read_rows(csv_path: Path):
    assert csv_path.exists(), f"Missing snapshot file: {csv_path}"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        return header, list(reader)


def test_rosters_schema_and_pk():
    snap = latest_snapshot_dir()
    header, rows = read_rows(snap / "rosters.csv")
    expected = [
        "player_id","name","position","team","jersey_number","status","depth_chart_rank","snap_percentage","last_updated"
    ]
    assert header == expected
    # Primary key: player_id unique if any rows
    if rows:
        player_ids = [r[0] for r in rows]
        assert len(player_ids) == len(set(player_ids))
        # Basic dtype checks (allow blanks)
        for r in rows:
            if r[4]:
                assert r[4].isdigit()
            if r[6]:
                assert r[6].lstrip("-").isdigit()
            if r[7]:
                try:
                    float(r[7])
                except ValueError:
                    assert False, f"snap_percentage not float-like: {r[7]}"
            if r[8]:
                # datetime parse
                try:
                    datetime.fromisoformat(r[8])
                except ValueError:
                    assert False, f"last_updated not ISO datetime: {r[8]}"


def test_depth_charts_schema():
    snap = latest_snapshot_dir()
    header, rows = read_rows(snap / "depth_charts.csv")
    expected = [
        "team","player_id","player_name","position","slot","role","package","depth_chart_rank","last_updated"
    ]
    assert header == expected
    # Composite uniqueness (team, player_id) if present
    if rows:
        keys = [(r[0], r[1]) for r in rows]
        assert len(keys) == len(set(keys))


def test_injuries_schema():
    snap = latest_snapshot_dir()
    header, rows = read_rows(snap / "injuries.csv")
    expected = [
        "player_id","name","team","position","practice_status","game_status","designation","report_date","return_date"
    ]
    assert header == expected
    if rows:
        # report_date should be parseable ISO date if present
        for r in rows:
            if r[7]:
                try:
                    datetime.fromisoformat(r[7])
                except ValueError:
                    assert False, f"report_date not ISO: {r[7]}"
