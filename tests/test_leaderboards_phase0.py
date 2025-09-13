#!/usr/bin/env python3
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_api_leaderboard_excludes_invalid_and_nameless_by_default():
    resp = client.get("/api/browse/leaderboard")
    assert resp.status_code == 200
    data = resp.json()
    rows = data.get("rows", [])
    assert isinstance(rows, list)
    for r in rows:
        pid = (r.get("player_id") or "").lower()
        name = (r.get("name") or "").strip()
        assert "_ol" not in pid
        assert name != ""
