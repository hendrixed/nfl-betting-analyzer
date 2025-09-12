"""
Market mapping utilities to normalize sportsbook market and book names
into internal canonical identifiers used by this repo.

Canonical market names (subset, extend as needed):
- QB: player_pass_att, player_pass_cmp, player_pass_yds, player_pass_tds, player_ints, player_rush_yds (QB), player_pass_long
- RB: player_rush_att, player_rush_yds, player_rush_tds, player_rec, player_rec_yds, player_rec_tds, player_rush_rec_yds
- WR/TE: player_tgts, player_rec, player_rec_yds, player_rec_tds, player_long_rec
- K: player_fg_made, player_fg_made_50+, player_xp_made
- Team: team_total, team_total_1H, team_total_2H
"""
from typing import Dict, Tuple

# Canonical lowercase sportsbook keys
BOOK_ALIASES: Dict[str, str] = {
    # DraftKings variants
    "dk": "draftkings",
    "draft kings": "draftkings",
    "draftkings": "draftkings",
    # FanDuel variants
    "fd": "fanduel",
    "fan duel": "fanduel",
    "fanduel": "fanduel",
    # BetMGM
    "betmgm": "betmgm",
    # Caesars
    "caesars": "caesars",
    # Others
    "pointsbet": "pointsbet",
    "barstool": "barstool",
    "unibet": "unibet",
    "bet365": "bet365",
    "espnbet": "espnbet",
}

# Raw market string -> canonical internal name (normalized to lower)
# Keep this deliberately redundant to catch real-world label variance
MARKET_ALIASES: Dict[str, str] = {
    # QB
    "passing yards": "player_pass_yds",
    "pass yds": "player_pass_yds",
    "pass_yds": "player_pass_yds",
    "py": "player_pass_yds",
    "pass yards": "player_pass_yds",
    "passing tds": "player_pass_tds",
    "pass tds": "player_pass_tds",
    "interceptions": "player_ints",
    "ints": "player_ints",
    "pass attempts": "player_pass_att",
    "passing attempts": "player_pass_att",
    "pass completions": "player_pass_cmp",
    "passing completions": "player_pass_cmp",
    "longest completion": "player_pass_long",
    "longest pass": "player_pass_long",
    # RB
    "rushing yards": "player_rush_yds",
    "rush yds": "player_rush_yds",
    "rushing attempts": "player_rush_att",
    "rush attempts": "player_rush_att",
    "rushing tds": "player_rush_tds",
    "receptions": "player_rec",
    "receiving yards": "player_rec_yds",
    "rec yards": "player_rec_yds",
    "receiving tds": "player_rec_tds",
    "rush+rec yards": "player_rush_rec_yds",
    "rushing + receiving yards": "player_rush_rec_yds",
    # WR/TE
    "targets": "player_tgts",
    "receptions alt": "player_rec",  # some books use this label set
    "longest reception": "player_long_rec",
    # K
    "field goals made": "player_fg_made",
    "fg made": "player_fg_made",
    "fg 50+ made": "player_fg_made_50+",
    "extra points made": "player_xp_made",
    # Team
    "team total": "team_total",
    "team total 1h": "team_total_1H",
    "team total 2h": "team_total_2H",
    # Team markets (book-level)
    "h2h": "h2h",
    "moneyline": "h2h",
    "ml": "h2h",
    "spreads": "spreads",
    "spread": "spreads",
    "point spread": "spreads",
    "totals": "totals",
    "total": "totals",
    "over/under": "totals",
}


def normalize_book(book: str) -> str:
    key = (book or "").strip().lower()
    return BOOK_ALIASES.get(key, key)


def normalize_market(raw_market: str) -> str:
    key = (raw_market or "").strip().lower()
    # quick exact match
    if key in MARKET_ALIASES:
        return MARKET_ALIASES[key]
    # fallback: remove punctuation and extra spaces
    import re
    norm = re.sub(r"[^a-z0-9]+", " ", key).strip()
    mapped = MARKET_ALIASES.get(norm)
    if mapped:
        return mapped
    # last-resort heuristics can be added here; default to unknown
    return ""


def to_internal(book: str, raw_market: str) -> Tuple[str, str]:
    """Return canonical (book, market). Raises ValueError if unknown market.
    """
    b = normalize_book(book)
    m = normalize_market(raw_market)
    if not m:
        raise ValueError(f"Unknown market pattern: {raw_market}")
    return b, m


# Team normalization
# Map common team names/aliases to standard NFL abbreviations
TEAM_ALIASES: Dict[str, str] = {
    # AFC East
    "buffalo bills": "BUF", "bills": "BUF", "buf": "BUF",
    "miami dolphins": "MIA", "dolphins": "MIA", "mia": "MIA",
    "new england patriots": "NE", "patriots": "NE", "ne": "NE", "new england": "NE",
    "new york jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    # AFC North
    "baltimore ravens": "BAL", "ravens": "BAL", "bal": "BAL",
    "cincinnati bengals": "CIN", "bengals": "CIN", "cin": "CIN",
    "cleveland browns": "CLE", "browns": "CLE", "cle": "CLE",
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pit": "PIT",
    # AFC South
    "houston texans": "HOU", "texans": "HOU", "hou": "HOU",
    "indianapolis colts": "IND", "colts": "IND", "ind": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jags": "JAX", "jax": "JAX",
    "tennessee titans": "TEN", "titans": "TEN", "ten": "TEN",
    # AFC West
    "denver broncos": "DEN", "broncos": "DEN", "den": "DEN",
    "kansas city chiefs": "KC", "kansas city": "KC", "chiefs": "KC", "kc": "KC",
    "las vegas raiders": "LV", "raiders": "LV", "lv": "LV", "oakland raiders": "LV", "lvr": "LV",
    "los angeles chargers": "LAC", "chargers": "LAC", "lac": "LAC", "la chargers": "LAC",
    # NFC East
    "dallas cowboys": "DAL", "cowboys": "DAL", "dal": "DAL",
    "new york giants": "NYG", "giants": "NYG", "nyg": "NYG",
    "philadelphia eagles": "PHI", "eagles": "PHI", "phi": "PHI",
    "washington commanders": "WAS", "washington football team": "WAS", "commanders": "WAS", "was": "WAS", "washington": "WAS",
    # NFC North
    "chicago bears": "CHI", "bears": "CHI", "chi": "CHI",
    "detroit lions": "DET", "lions": "DET", "det": "DET",
    "green bay packers": "GB", "packers": "GB", "gb": "GB", "green bay": "GB",
    "minnesota vikings": "MIN", "vikings": "MIN", "min": "MIN",
    # NFC South
    "atlanta falcons": "ATL", "falcons": "ATL", "atl": "ATL",
    "carolina panthers": "CAR", "panthers": "CAR", "car": "CAR",
    "new orleans saints": "NO", "saints": "NO", "no": "NO", "nola": "NO",
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "bucs": "TB", "tb": "TB",
    # NFC West
    "arizona cardinals": "ARI", "cardinals": "ARI", "cards": "ARI", "ari": "ARI",
    "los angeles rams": "LAR", "rams": "LAR", "lar": "LAR", "la rams": "LAR",
    "san francisco 49ers": "SF", "49ers": "SF", "niners": "SF", "sf": "SF",
    "seattle seahawks": "SEA", "seahawks": "SEA", "sea": "SEA",
}


def normalize_team_name(raw: str) -> str:
    """Normalize a provider team string to a standard NFL team abbreviation.

    - Accepts full team names (e.g., "Kansas City Chiefs"), common nicknames ("Chiefs"),
      or abbreviations ("KC", "LAC"). Returns the standard 2-3 letter abbreviation.
    - Fallback returns an uppercased trimmed string (or empty) when unknown.
    """
    key = (raw or "").strip().lower()
    if key.startswith("the "):
        key = key[4:]
    mapped = TEAM_ALIASES.get(key)
    if mapped:
        return mapped
    # If already looks like abbreviation (2-3 chars), return uppercased
    if 1 < len(key) <= 3 and key.isalpha():
        return key.upper()
    return (raw or "").strip().upper()
