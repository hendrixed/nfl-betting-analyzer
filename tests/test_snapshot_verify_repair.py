#!/usr/bin/env python3
from pathlib import Path
from typer.testing import CliRunner
from nfl_cli import app


def test_snapshot_verify_repair_creates_missing_headers(tmp_path):
    runner = CliRunner()
    # Create an empty snapshot directory
    date_str = "2024-01-04"
    snap_dir = Path("data/snapshots") / date_str
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Run snapshot-verify with --date and --repair to create header-only CSVs
    result = runner.invoke(app, ["snapshot-verify", "--date", date_str, "--repair"])
    assert result.exit_code == 0

    # Spot-check some critical files exist after repair
    must_exist = [
        "schedules.csv",
        "rosters.csv",
        "depth_charts.csv",
        "weekly_stats.csv",
        "routes.csv",
        "usage_shares.csv",
    ]
    for fname in must_exist:
        assert (snap_dir / fname).exists()
