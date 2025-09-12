#!/usr/bin/env python3
import os
from pathlib import Path
import csv
from datetime import datetime

from core.database_models import get_db_session
from core.data.ingestion_adapters import UnifiedDataIngestion
from core.services import browse_service as bs


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def test_depth_chart_and_schedule_ingest_roundtrip(tmp_path):
    # Use a fixed snapshot date for determinism
    date_str = "2024-01-01"
    snap_dir = Path("data/snapshots") / date_str
    _ensure_dir(snap_dir)

    # Write minimal schedules.csv consistent with SNAPSHOT_MIN_COLUMNS
    sched_file = snap_dir / "schedules.csv"
    with open(sched_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_id","season","week","season_type","home_team","away_team",
                "kickoff_dt_utc","kickoff_dt_local","network","spread_close","total_close","officials_crew","stadium","roof_state"
            ],
        )
        writer.writeheader()
        writer.writerow({
            "game_id": "TEST-NE-BUF-2024-W1",
            "season": 2024,
            "week": 1,
            "season_type": "REG",
            "home_team": "NE",
            "away_team": "BUF",
            "kickoff_dt_utc": "2024-09-08T17:00:00",
            "kickoff_dt_local": "2024-09-08 13:00:00",
            "network": "",
            "spread_close": "",
            "total_close": "",
            "officials_crew": "",
            "stadium": "Gillette Stadium",
            "roof_state": "outdoors",
        })

    # Write minimal depth_charts.csv consistent with SNAPSHOT_MIN_COLUMNS
    depth_file = snap_dir / "depth_charts.csv"
    with open(depth_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "team","player_id","player_name","position","slot","role","package","depth_chart_rank","last_updated"
            ],
        )
        writer.writeheader()
        writer.writerow({
            "team": "NE",
            "player_id": "test-p1",
            "player_name": "Test Player One",
            "position": "WR",
            "slot": "",
            "role": "",
            "package": "",
            "depth_chart_rank": 1,
            "last_updated": "2024-09-01T00:00:00",
        })
        writer.writerow({
            "team": "NE",
            "player_id": "test-p2",
            "player_name": "Test Player Two",
            "position": "WR",
            "slot": "",
            "role": "",
            "package": "",
            "depth_chart_rank": 2,
            "last_updated": "2024-09-01T00:00:00",
        })

    # Ingest
    session = get_db_session()
    try:
        ingestion = UnifiedDataIngestion(session)
        dc_res = ingestion.ingest_depth_chart_snapshot(date_str=date_str, season=2024, week=1)
        sch_res = ingestion.ingest_schedule_snapshot(date_str=date_str)

        assert isinstance(dc_res, dict)
        assert isinstance(sch_res, dict)

        # Verify depth chart browse uses DB-backed entries
        depth = bs.get_team_depth_chart(session, team_id="NE", week=1)
        assert isinstance(depth, dict)
        assert "WR" in depth
        wr_list = depth.get("WR") or []
        # At least one entry should exist for WR
        assert len(wr_list) >= 1
        # Ensure at least one of our inserted players is present
        assert any(e.get("player_id") in {"test-p1", "test-p2"} for e in wr_list)
        # Ensure ranks present and limited to <= 4
        for e in wr_list:
            assert "player_id" in e and e["player_id"] in {"test-p1", "test-p2"}
            assert "rank" in e and int(e["rank"]) <= 4
            assert "name" in e and e["name"]

        # Verify schedule browse sees our game (since_date early enough)
        sched = bs.get_team_schedule(session, team_id="NE", season=2024, since_date=datetime(2000, 1, 1))
        assert isinstance(sched, list)
        assert any(g.get("game_id") == "TEST-NE-BUF-2024-W1" for g in sched)
    finally:
        session.close()
