#!/usr/bin/env python3
from pathlib import Path
from typer.testing import CliRunner
from nfl_cli import app


def test_cli_routes_and_usage_ingest_header_only(tmp_path):
    runner = CliRunner()
    date_str = "2024-01-02"
    snap_dir = Path("data/snapshots") / date_str
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Write header-only files per SNAPSHOT_MIN_COLUMNS
    from core.data.ingestion_adapters import SNAPSHOT_MIN_COLUMNS
    import pandas as pd

    routes_cols = SNAPSHOT_MIN_COLUMNS.get("routes.csv", [])
    usage_cols = SNAPSHOT_MIN_COLUMNS.get("usage_shares.csv", [])

    pd.DataFrame(columns=routes_cols).to_csv(snap_dir / "routes.csv", index=False)
    pd.DataFrame(columns=usage_cols).to_csv(snap_dir / "usage_shares.csv", index=False)

    # Run CLI ingests; these should not error even when no rows
    res_routes = runner.invoke(app, ["routes-ingest", "--date", date_str])
    assert res_routes.exit_code == 0

    res_usage = runner.invoke(app, ["usage-ingest", "--date", date_str])
    assert res_usage.exit_code == 0
