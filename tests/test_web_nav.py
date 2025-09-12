#!/usr/bin/env python3
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root_redirects_to_games():
    resp = client.get("/")
    assert resp.status_code in (200, 302, 307)
    if resp.status_code in (302, 307):
        assert resp.headers.get("location", "").startswith("/games")
    else:
        assert "Games" in resp.text


def test_web_redirects_to_games():
    resp = client.get("/web")
    assert resp.status_code in (200, 302, 307)
    if resp.status_code in (302, 307):
        assert resp.headers.get("location", "").startswith("/games")
    else:
        assert "Games" in resp.text


def test_navbar_contains_tabs():
    resp = client.get("/games")
    assert resp.status_code == 200
    html = resp.text
    for tab in ["Players", "Teams", "Games", "Leaderboards", "Odds", "Backtests", "Insights"]:
        assert tab in html
