#!/usr/bin/env python3
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app, follow_redirects=True)


def test_web_team_detail_accepts_schedule_params():
    resp = client.get("/team/NE", params={"include_past": "true", "timezone": "America/New_York"})
    assert resp.status_code == 200
