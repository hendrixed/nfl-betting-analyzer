from fastapi.testclient import TestClient
from api.app import app


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
