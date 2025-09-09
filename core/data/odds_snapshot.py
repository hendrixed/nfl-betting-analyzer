"""
Odds snapshot writer
- Generates a lightweight odds.csv under data/snapshots/YYYY-MM-DD/
- Uses only standard library and market mapping for canonical names
- Intended as a placeholder until real odds integrations are enabled
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Sequence
from datetime import datetime
import csv
import random

from core.data.market_mapping import to_internal


def ensure_snapshot_dir(date_str: Optional[str] = None) -> Path:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    base = Path("data/snapshots") / date_str
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_mock_odds_snapshot(
    session,
    date_str: Optional[str] = None,
    books: Optional[Sequence[str]] = None,
    markets: Optional[Sequence[str]] = None,
    max_offers: int = 100,
) -> Path:
    """Write a mock odds.csv based on current players in DB or generic placeholders.
    Returns the path to the written odds.csv
    """
    books = books or ["DraftKings", "FanDuel", "BetMGM"]
    # A small set of common markets; these will be canonicalized
    markets = markets or [
        "Passing Yards",
        "Rushing Yards",
        "Receptions",
        "Receiving Yards",
        "Interceptions",
    ]

    snapshot_dir = ensure_snapshot_dir(date_str)
    odds_path = snapshot_dir / "odds.csv"

    # Choose some players from DB if available
    try:
        from core.database_models import Player
        players = (
            session.query(Player)
            .filter(getattr(Player, "is_active", True) == True)  # backward compatible
            .limit(50)
            .all()
        )
    except Exception:
        players = []

    # Prepare iterable of player entries
    player_entries = []
    if players:
        for p in players:
            team_id = getattr(p, "current_team", None) or getattr(p, "team", None) or "UNK"
            player_entries.append((p.player_id, team_id))
    else:
        # Fallback generic entries
        teams = ["KC", "BUF", "SF", "DAL", "PHI", "BAL", "CIN", "NYJ", "MIA", "DET"]
        for i in range(30):
            player_entries.append((f"mock_{i}", random.choice(teams)))

    rows_written = 0
    with open(odds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "book",
                "market",
                "player_id",
                "team_id",
                "line",
                "over_odds",
                "under_odds",
            ],
        )
        writer.writeheader()

        # Shuffle to vary
        random.shuffle(player_entries)

        for player_id, team_id in player_entries:
            for book in books:
                for market in markets:
                    # Canonicalize
                    try:
                        cb, cm = to_internal(book, market)
                    except Exception:
                        continue

                    # Simple line generation: center around role-based guesses
                    base_line = 0.0
                    if cm in ("player_pass_yds",):
                        base_line = random.uniform(220.0, 295.0)
                    elif cm in ("player_rush_yds",):
                        base_line = random.uniform(35.0, 85.0)
                    elif cm in ("player_rec",):
                        base_line = random.uniform(3.0, 7.5)
                    elif cm in ("player_rec_yds",):
                        base_line = random.uniform(38.0, 82.0)
                    elif cm in ("player_ints",):
                        base_line = random.uniform(0.5, 1.5)
                    else:
                        base_line = random.uniform(10.0, 50.0)

                    over_odds = random.choice([-120, -115, -110, -105, 100, 105])
                    under_odds = -over_odds if over_odds > 0 else -110

                    writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "book": cb,
                            "market": cm,
                            "player_id": player_id,
                            "team_id": team_id,
                            "line": round(base_line, 1),
                            "over_odds": over_odds,
                            "under_odds": under_odds,
                        }
                    )
                    rows_written += 1
                    if rows_written >= max_offers:
                        break
                if rows_written >= max_offers:
                    break
            if rows_written >= max_offers:
                break

    return odds_path
