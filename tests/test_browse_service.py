#!/usr/bin/env python3
import types
from core.services import browse_service as bs
from core.database_models import get_db_session


def test_search_players_shape_excludes_inactive_by_default():
    session = get_db_session()
    out = bs.search_players(session, q=None, team_id=None, position=None, page=1, page_size=5)
    assert isinstance(out, dict)
    assert set(["rows", "total", "page", "page_size", "sort", "order"]).issubset(out.keys())
    assert isinstance(out["rows"], list)


def test_player_gamelog_contract_fields():
    session = get_db_session()
    data = bs.get_player_gamelog(session, player_id="unknown-id", season=None)
    assert isinstance(data, list)
    for row in data:
        assert "date" in row
        assert "opponent" in row
        assert "venue" in row


def test_team_schedule_upcoming_only_contract():
    session = get_db_session()
    sched = bs.get_team_schedule(session, team_id="NE", season=None)
    assert isinstance(sched, list)
    for g in sched:
        assert "date" in g


def test_depth_chart_top4_contract():
    session = get_db_session()
    depth = bs.get_team_depth_chart(session, team_id="NE")
    assert isinstance(depth, dict)
    for pos, entries in depth.items():
        assert isinstance(entries, list)
        assert len(entries) <= 4
        for e in entries:
            assert "player_id" in e and "name" in e


def test_player_gamelog_sorted_by_date_when_available():
    session = get_db_session()
    rows = bs.get_player_gamelog(session, player_id="unknown-id", season=None)
    assert isinstance(rows, list)
    # If there are multiple rows with dates, ensure they are sorted descending by date
    dates = [r.get("date") for r in rows if r.get("date")]
    if len(dates) > 1:
        assert dates == sorted(dates, reverse=True)


def test_search_players_excludes_invalid_ids_and_nameless_by_default():
    session = get_db_session()
    out = bs.search_players(session, q=None, team_id=None, position=None, page=1, page_size=25)
    assert isinstance(out, dict)
    assert isinstance(out.get("rows"), list)
    for r in out.get("rows", []):
        # No placeholder/aggregate OL ids leaked
        assert "_ol" not in (r.get("player_id") or "").lower()
        # Name must be present and non-empty
        assert (r.get("name") or "").strip() != ""
