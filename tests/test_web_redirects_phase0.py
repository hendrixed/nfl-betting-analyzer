#!/usr/bin/env python3
from fastapi.testclient import TestClient
from api.app import app

# Ensure redirects are followed so that final status is 200
client = TestClient(app, follow_redirects=True)


def test_root_redirects_and_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200


def test_web_redirects_and_returns_200():
    resp = client.get("/web")
    assert resp.status_code == 200
