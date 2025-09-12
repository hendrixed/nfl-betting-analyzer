from fastapi.testclient import TestClient
from pathlib import Path
import csv
from datetime import datetime

from api.app import app


def write_snapshot(date_dir: Path):
    date_dir.mkdir(parents=True, exist_ok=True)
    odds_path = date_dir / "odds.csv"
    with open(odds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "book",
                "market",
                "player_id",
                "team_id",
                "line",
                "over_odds",
                "under_odds",
            ],
        )
        writer.writeheader()
        now = datetime.utcnow().isoformat()
        # Team market rows (h2h) with numeric line
        writer.writerow({
            "timestamp": now,
            "book": "draftkings",
            "market": "h2h",
            "player_id": "",
            "team_id": "KC",
            "line": 0,
            "over_odds": -120,
            "under_odds": "",
        })
        writer.writerow({
            "timestamp": now,
            "book": "draftkings",
            "market": "h2h",
            "player_id": "",
            "team_id": "BUF",
            "line": 0,
            "over_odds": 105,
            "under_odds": "",
        })
    return odds_path


def test_props_team_filter_h2h(tmp_path: Path, monkeypatch):
    # Create an isolated snapshot directory under tmp_path
    snap_dir = tmp_path / "snapshot"
    write_snapshot(snap_dir)

    # Monkeypatch latest_snapshot_dir used by the API to point to our test directory
    import api.app as api_app
    monkeypatch.setattr(api_app, "latest_snapshot_dir", lambda base="data/snapshots": snap_dir)

    client = TestClient(app)

    # Filter by book + market + team (Chiefs)
    resp = client.get("/betting/props", params={
        "book": "DraftKings",
        "market": "h2h",
        "team": "Chiefs",
    })
    assert resp.status_code == 200
    data = resp.json()
    offers = data.get("offers", [])
    assert isinstance(offers, list)
    # Only KC should match
    assert all((o.get("team_id") or "").upper() == "KC" for o in offers)
    assert any((o.get("team_id") or "").upper() == "KC" for o in offers)

    # Bills should match only BUF
    resp2 = client.get("/betting/props", params={
        "book": "DraftKings",
        "market": "h2h",
        "team": "Bills",
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    offers2 = data2.get("offers", [])
    assert all((o.get("team_id") or "").upper() == "BUF" for o in offers2)
