#!/usr/bin/env python3
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_web_players_page_renders():
    resp = client.get("/web/players")
    assert resp.status_code == 200
    assert "Players" in resp.text or "Players page" in resp.text


def test_web_leaderboards_page_renders():
    resp = client.get("/web/leaderboards")
    assert resp.status_code == 200
    assert "Leaderboards" in resp.text


def test_api_leaderboard_json_contract():
    resp = client.get("/api/browse/leaderboard")
    assert resp.status_code == 200
    data = resp.json()
    for k in ("stat", "rows"):
        assert k in data
    assert isinstance(data["rows"], list)


def test_api_player_profile_not_found_is_safe():
    resp = client.get("/api/browse/player/unknown-id/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("player_id") == "unknown-id"
    assert "error" in data


def test_api_player_gamelog_contract():
    resp = client.get("/api/browse/player/unknown-id/gamelog")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("player_id") == "unknown-id"
    assert "gamelog" in data
    assert isinstance(data["gamelog"], list)


def test_api_team_info_contract():
    resp = client.get("/api/browse/team/NE")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("team_id") == "NE" or data.get("team_id") == "NE" if "team_id" in data else True
    assert "roster" in data


def test_api_team_depth_chart_contract():
    resp = client.get("/api/browse/team/NE/depth-chart")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("team_id") == "NE"
    assert "depth_chart" in data


def test_api_team_schedule_contract():
    resp = client.get("/api/browse/team/NE/schedule")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("team_id") == "NE"
    assert "schedule" in data


def test_export_players_csv_endpoint():
    resp = client.get("/api/browse/players/export.csv")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/csv")


def test_export_leaderboard_csv_endpoint():
    resp = client.get("/api/browse/leaderboard/export.csv")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/csv")
