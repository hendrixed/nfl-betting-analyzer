#!/usr/bin/env python3
"""
Browse Service Layer
Provides player/team browsing helpers for API and web routes.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_, desc, asc

from core.database_models import get_db_session, Player, PlayerGameStats, Game, Team


def _resolve_opponent_and_venue(team: str, game: Game) -> Tuple[str, str]:
    try:
        if (team or "").upper() == (game.home_team or "").upper():
            return game.away_team, "HOME"
        elif (team or "").upper() == (game.away_team or "").upper():
            return game.home_team, "AWAY"
        else:
            # Fallback when team code mismatch
            return (game.away_team or game.home_team or ""), "NEUTRAL"
    except Exception:
        return (game.away_team or game.home_team or ""), "NEUTRAL"


def get_player_profile(session: Session, player_id: str) -> Dict[str, Any]:
    """Return player profile plus inferred current-season summary.
    Current season is defined as the max(Game.season) for games the player has stats.
    """
    p = session.query(Player).filter(Player.player_id == player_id).first()
    if not p:
        return {"error": "not_found", "player_id": player_id}

    # Determine latest season for this player's stats
    latest_season = (
        session.query(func.max(Game.season))
        .join(PlayerGameStats, PlayerGameStats.game_id == Game.game_id)
        .filter(PlayerGameStats.player_id == player_id)
        .scalar()
    )

    season_summary: Dict[str, Any] = {}
    if latest_season is not None:
        agg = (
            session.query(
                func.coalesce(func.sum(PlayerGameStats.passing_yards), 0),
                func.coalesce(func.sum(PlayerGameStats.passing_touchdowns), 0),
                func.coalesce(func.sum(PlayerGameStats.passing_interceptions), 0),
                func.coalesce(func.sum(PlayerGameStats.passing_completions), 0),
                func.coalesce(func.sum(PlayerGameStats.rushing_attempts), 0),
                func.coalesce(func.sum(PlayerGameStats.rushing_yards), 0),
                func.coalesce(func.sum(PlayerGameStats.rushing_touchdowns), 0),
                func.coalesce(func.sum(PlayerGameStats.receptions), 0),
                func.coalesce(func.sum(PlayerGameStats.receiving_yards), 0),
                func.coalesce(func.sum(PlayerGameStats.receiving_touchdowns), 0),
            )
            .join(Game, PlayerGameStats.game_id == Game.game_id)
            .filter(PlayerGameStats.player_id == player_id, Game.season == latest_season)
            .first()
        )
        season_summary = {
            "season": int(latest_season),
            "passing_yards": int(agg[0] or 0),
            "passing_tds": int(agg[1] or 0),
            "interceptions": int(agg[2] or 0),
            "completions": int(agg[3] or 0),
            "rushing_att": int(agg[4] or 0),
            "rushing_yards": int(agg[5] or 0),
            "rushing_tds": int(agg[6] or 0),
            "receptions": int(agg[7] or 0),
            "receiving_yards": int(agg[8] or 0),
            "receiving_tds": int(agg[9] or 0),
        }

    return {
        "player_id": p.player_id,
        "name": p.name,
        "position": p.position,
        "team": p.current_team,
        "status": "active" if getattr(p, "is_active", False) else "inactive",
        "depth_chart_rank": getattr(p, "depth_chart_rank", None),
        "current_season": season_summary,
    }


def get_player_gamelog(session: Session, player_id: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return game-by-game stats with date, opponent, and home/away for given season (or latest)."""
    q = (
        session.query(PlayerGameStats, Game)
        .join(Game, PlayerGameStats.game_id == Game.game_id)
        .filter(PlayerGameStats.player_id == player_id)
    )
    if season is not None:
        q = q.filter(Game.season == season)
    else:
        latest = session.query(func.max(Game.season)).join(PlayerGameStats, PlayerGameStats.game_id == Game.game_id).filter(PlayerGameStats.player_id == player_id).scalar()
        if latest is not None:
            q = q.filter(Game.season == latest)
    rows = q.order_by(desc(Game.game_date)).all()

    out: List[Dict[str, Any]] = []
    # Get player's team for opponent resolution
    pl = session.query(Player).filter(Player.player_id == player_id).first()
    team = (pl.current_team if pl else "") or ""
    for s, g in rows:
        opp, venue = _resolve_opponent_and_venue(team, g)
        out.append({
            "game_id": g.game_id,
            "date": g.game_date.isoformat() if g.game_date else None,
            "week": g.week,
            "season": g.season,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "opponent": opp,
            "venue": venue,
            "stats": {
                "passing_yards": s.passing_yards,
                "passing_tds": s.passing_touchdowns,
                "interceptions": s.passing_interceptions,
                "completions": s.passing_completions,
                "rushing_att": s.rushing_attempts,
                "rushing_yards": s.rushing_yards,
                "rushing_tds": s.rushing_touchdowns,
                "receptions": s.receptions,
                "receiving_yards": s.receiving_yards,
                "receiving_tds": s.receiving_touchdowns,
                "fantasy_ppr": s.fantasy_points_ppr,
            }
        })
    return out


def get_player_career_totals(session: Session, player_id: str) -> Dict[str, Any]:
    """Return career total counters across available PlayerGameStats."""
    agg = (
        session.query(
            func.coalesce(func.sum(PlayerGameStats.passing_yards), 0),
            func.coalesce(func.sum(PlayerGameStats.passing_touchdowns), 0),
            func.coalesce(func.sum(PlayerGameStats.passing_interceptions), 0),
            func.coalesce(func.sum(PlayerGameStats.passing_completions), 0),
            func.coalesce(func.sum(PlayerGameStats.rushing_attempts), 0),
            func.coalesce(func.sum(PlayerGameStats.rushing_yards), 0),
            func.coalesce(func.sum(PlayerGameStats.rushing_touchdowns), 0),
            func.coalesce(func.sum(PlayerGameStats.receptions), 0),
            func.coalesce(func.sum(PlayerGameStats.receiving_yards), 0),
            func.coalesce(func.sum(PlayerGameStats.receiving_touchdowns), 0),
        )
        .filter(PlayerGameStats.player_id == player_id)
        .first()
    )
    return {
        "passing_yards": int(agg[0] or 0),
        "passing_tds": int(agg[1] or 0),
        "interceptions": int(agg[2] or 0),
        "completions": int(agg[3] or 0),
        "rushing_att": int(agg[4] or 0),
        "rushing_yards": int(agg[5] or 0),
        "rushing_tds": int(agg[6] or 0),
        "receptions": int(agg[7] or 0),
        "receiving_yards": int(agg[8] or 0),
        "receiving_tds": int(agg[9] or 0),
    }


def search_players(
    session: Session,
    q: Optional[str] = None,
    team_id: Optional[str] = None,
    position: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    sort: str = "name",
    order: str = "asc",
) -> Dict[str, Any]:
    """Search/browse players with pagination and sorting.
    Returns dict with rows, total, page, page_size, sort, order.
    """
    page = max(1, int(page or 1))
    page_size = max(1, min(200, int(page_size or 50)))
    base = session.query(Player)
    if q:
        qq = f"%{q.strip()}%"
        base = base.filter(or_(Player.name.ilike(qq), Player.player_id.ilike(qq)))
    if team_id:
        base = base.filter((Player.current_team or "").astext == team_id.upper()) if hasattr(Player.current_team, 'astext') else base.filter(Player.current_team == team_id.upper())
    if position:
        base = base.filter(Player.position == position.upper())

    # Total before pagination
    try:
        total = base.count()
    except Exception:
        total = 0

    # Sorting
    sort_key = (sort or "name").lower()
    sort_map = {
        "name": Player.name,
        "position": Player.position,
        "team": Player.current_team,
        "rank": getattr(Player, "depth_chart_rank", None),
    }
    col = sort_map.get(sort_key) or Player.name
    direction = desc if (order or "asc").lower() == "desc" else asc

    query = base.order_by(direction(col)).offset((page - 1) * page_size).limit(page_size)
    rows: List[Dict[str, Any]] = []
    for p in query.all():
        rows.append({
            "player_id": p.player_id,
            "name": p.name,
            "position": p.position,
            "team": p.current_team,
            "status": "active" if getattr(p, "is_active", False) else "inactive",
            "depth_chart_rank": getattr(p, "depth_chart_rank", None),
        })
    return {
        "rows": rows,
        "total": int(total or 0),
        "page": page,
        "page_size": page_size,
        "sort": sort_key,
        "order": (order or "asc").lower(),
    }


def get_team_info(session: Session, team_id: str) -> Dict[str, Any]:
    t = session.query(Team).filter(Team.team_id == team_id.upper()).first()
    roster = session.query(Player).filter(Player.current_team == team_id.upper()).order_by(Player.position.asc(), Player.name.asc()).all()
    return {
        "team_id": team_id.upper(),
        "team_name": getattr(t, "team_name", team_id.upper()),
        "conference": getattr(t, "conference", None),
        "division": getattr(t, "division", None),
        "head_coach": getattr(t, "coach", None) if hasattr(t, "coach") else None,
        "stadium": getattr(t, "stadium_name", None),
        "roster": [
            {
                "player_id": p.player_id,
                "name": p.name,
                "position": p.position,
                "depth_chart_rank": getattr(p, "depth_chart_rank", None),
                "status": "active" if getattr(p, "is_active", False) else "inactive",
            }
            for p in roster
        ],
    }


def get_team_depth_chart(session: Session, team_id: str, week: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Approximate depth chart from Player.depth_chart_rank per position."""
    players = session.query(Player).filter(Player.current_team == team_id.upper()).all()
    by_pos: Dict[str, List[Dict[str, Any]]] = {}
    for p in players:
        by_pos.setdefault(p.position or "UNK", []).append({
            "player_id": p.player_id,
            "name": p.name,
            "rank": getattr(p, "depth_chart_rank", None),
            "status": "active" if getattr(p, "is_active", False) else "inactive",
        })
    # sort by rank then name
    for k in list(by_pos.keys()):
        by_pos[k] = sorted(by_pos[k], key=lambda x: (x.get("rank") if x.get("rank") is not None else 9999, x.get("name") or ""))
    return by_pos


def get_team_schedule(session: Session, team_id: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
    q = session.query(Game).filter(or_(Game.home_team == team_id.upper(), Game.away_team == team_id.upper()))
    if season is not None:
        q = q.filter(Game.season == season)
    games = q.order_by(Game.game_date.asc()).all()
    out: List[Dict[str, Any]] = []
    for g in games:
        out.append({
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "date": g.game_date.isoformat() if g.game_date else None,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_score": g.home_score,
            "away_score": g.away_score,
        })
    return out


_ALLOWED_STATS = {
    # map external stat keys to PlayerGameStats attribute names
    "passing_yards": PlayerGameStats.passing_yards,
    "passing_touchdowns": PlayerGameStats.passing_touchdowns,
    "interceptions": PlayerGameStats.passing_interceptions,
    "completions": PlayerGameStats.passing_completions,
    "rushing_attempts": PlayerGameStats.rushing_attempts,
    "rushing_yards": PlayerGameStats.rushing_yards,
    "rushing_touchdowns": PlayerGameStats.rushing_touchdowns,
    "receptions": PlayerGameStats.receptions,
    "receiving_yards": PlayerGameStats.receiving_yards,
    "receiving_touchdowns": PlayerGameStats.receiving_touchdowns,
    "fantasy_points_ppr": PlayerGameStats.fantasy_points_ppr,
}


def get_leaderboard(
    session: Session,
    stat: str,
    season: Optional[int] = None,
    position: Optional[str] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    col = _ALLOWED_STATS.get(stat)
    if col is None:
        # default to fantasy points
        col = PlayerGameStats.fantasy_points_ppr
    q = session.query(
        Player.player_id,
        Player.name,
        Player.position,
        Player.current_team,
        func.coalesce(func.sum(col), 0).label("value")
    ).join(PlayerGameStats, Player.player_id == PlayerGameStats.player_id)
    if season is not None:
        q = q.join(Game, PlayerGameStats.game_id == Game.game_id).filter(Game.season == season)
    if position:
        q = q.filter(Player.position == position.upper())
    q = q.group_by(Player.player_id, Player.name, Player.position, Player.current_team).order_by(desc("value")).limit(limit)

    out: List[Dict[str, Any]] = []
    for pid, name, pos, team, val in q:
        out.append({
            "player_id": pid,
            "name": name,
            "position": pos,
            "team": team,
            "value": float(val or 0.0),
        })
    return out


def get_leaderboard_paginated(
    session: Session,
    stat: str,
    season: Optional[int] = None,
    position: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
    sort: str = "value",
    order: str = "desc",
) -> Dict[str, Any]:
    """Leaderboard with pagination and sorting.
    sort in {value, name, team, position}; order in {asc, desc}.
    """
    page = max(1, int(page or 1))
    page_size = max(1, min(200, int(page_size or 25)))

    col_map = {
        "value": None,  # filled after expression defined
        "name": Player.name,
        "team": Player.current_team,
        "position": Player.position,
    }

    stat_col = _ALLOWED_STATS.get(stat) or PlayerGameStats.fantasy_points_ppr

    base = session.query(
        Player.player_id.label("pid"),
        func.coalesce(func.sum(stat_col), 0).label("value"),
    ).join(PlayerGameStats, Player.player_id == PlayerGameStats.player_id)
    if season is not None:
        base = base.join(Game, PlayerGameStats.game_id == Game.game_id).filter(Game.season == season)
    if position:
        base = base.filter(Player.position == position.upper())
    base = base.group_by(Player.player_id)

    # Total distinct grouped players
    try:
        total = session.query(func.count(func.distinct(Player.player_id))).join(PlayerGameStats, Player.player_id == PlayerGameStats.player_id)
        if season is not None:
            total = total.join(Game, PlayerGameStats.game_id == Game.game_id).filter(Game.season == season)
        if position:
            total = total.filter(Player.position == position.upper())
        total = total.scalar() or 0
    except Exception:
        total = 0

    # Sorting
    sort_key = (sort or "value").lower()
    direction = desc if (order or "desc").lower() == "desc" else asc

    # Build full query with Player columns for name/team/position
    full = session.query(
        Player.player_id,
        Player.name,
        Player.position,
        Player.current_team,
        func.coalesce(func.sum(stat_col), 0).label("value"),
    ).join(PlayerGameStats, Player.player_id == PlayerGameStats.player_id)
    if season is not None:
        full = full.join(Game, PlayerGameStats.game_id == Game.game_id).filter(Game.season == season)
    if position:
        full = full.filter(Player.position == position.upper())
    full = full.group_by(Player.player_id, Player.name, Player.position, Player.current_team)

    if sort_key == "value":
        full = full.order_by(direction(func.coalesce(func.sum(stat_col), 0)))
    else:
        full = full.order_by(direction(col_map.get(sort_key) or Player.name))

    full = full.offset((page - 1) * page_size).limit(page_size)

    rows: List[Dict[str, Any]] = []
    for pid, name, pos, team, val in full.all():
        rows.append({
            "player_id": pid,
            "name": name,
            "position": pos,
            "team": team,
            "value": float(val or 0.0),
        })

    return {
        "rows": rows,
        "total": int(total or 0),
        "page": page,
        "page_size": page_size,
        "sort": sort_key,
        "order": (order or "desc").lower(),
    }
