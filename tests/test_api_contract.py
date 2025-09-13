#!/usr/bin/env python3
"""
API Contract Tests for core endpoints:
- /health
- /models
- /predictions/players/fantasy
- /betting/props

These tests validate response codes and basic schema expectations.
"""

from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


def test_health_endpoint_contract():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "timestamp" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], bool)


def test_models_endpoint_contract():
    resp = client.get("/models")
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, list)
        # If models are present, confirm schema
        if data:
            row = data[0]
            for k in ("model_name", "position", "accuracy", "mae", "rmse", "last_updated"):
                assert k in row
    else:
        assert resp.status_code in (400, 403)


def test_fantasy_predictions_contract():
    resp = client.get("/predictions/players/fantasy", params={"limit": 1})
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, list)
        if data:
            row = data[0]
            for k in ("player_id", "name", "position", "team", "fantasy_points_ppr", "confidence", "model", "last_updated"):
                assert k in row
    else:
        # When models are not present, endpoint should be gated with 503
        assert resp.status_code in (400, 403, 503)


def test_betting_props_contract():
    # Ensure a snapshot exists
    snap = client.post("/admin/odds/snapshot", params={"max_offers": 8})
    assert snap.status_code == 200

    # Query props; market/book optional - endpoint should still return structure
    resp = client.get("/betting/props", params={"market": "Passing Yards", "book": "DK"})
    assert resp.status_code == 200
    data = resp.json()
    for key in ("book", "market", "offers", "timestamp", "opportunities"):
        assert key in data
    assert isinstance(data["offers"], list)
