from fastapi.testclient import TestClient
from api.app import app
from pathlib import Path

from core.data.ingestion_adapters import UnifiedDataIngestion, SNAPSHOT_MIN_COLUMNS
from core.database_models import get_db_session


def test_admin_snapshot_and_props_offers():
    client = TestClient(app)

    # Create a mock odds snapshot
    resp = client.post("/admin/odds/snapshot", params={"max_offers": 15})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["rows"] == 15

    # Query props with canonicalization
    resp2 = client.get("/betting/props", params={"book": "DK", "market": "Passing Yards"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["book"] == "draftkings"
    assert data2["market"] == "player_pass_yds"
    # Offers should be non-empty
    assert isinstance(data2.get("offers"), list)
    assert len(data2["offers"]) > 0


def test_mock_provider_schema_matches_snapshot_mapping():
    """Ensure odds.csv created via mock provider matches SNAPSHOT_MIN_COLUMNS mapping."""
    # Use ingestion layer to snapshot odds via mock provider
    session = get_db_session("sqlite:///nfl_predictions.db")
    ingestion = UnifiedDataIngestion(session)
    result = ingestion.snapshot_odds(provider="mock", max_offers=10)
    odds_path = Path(result["snapshot_path"]) if result.get("snapshot_path") else None
    assert odds_path and odds_path.exists(), "odds.csv should be created by mock provider"

    # Read header and compare with mapping
    expected = SNAPSHOT_MIN_COLUMNS["odds.csv"]
    with open(odds_path, newline="", encoding="utf-8") as f:
        import csv
        reader = csv.reader(f)
        header = next(reader, [])
    assert header == expected, f"odds.csv header mismatch. Got {header}, expected {expected}"
