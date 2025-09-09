"""
Compatibility real-time system facade used by legacy/integration tests.
Provides a thin wrapper around enhanced collectors and validators under core/.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from core.database_models import get_db_session
from enhanced_data_collector import EnhancedNFLDataCollector, RoleBasedStatsCollector
from data_validator import ComprehensiveValidator


@dataclass
class GameInfo:
    game_id: str
    home_team: str
    away_team: str
    game_date: date
    week: int
    season: int


class RealTimeNFLSystem:
    def __init__(self, db_path: str = "nfl_predictions.db", current_season: int = 2025):
        # Session
        self.session = get_db_session(f"sqlite:///{db_path}")
        self.current_season = current_season

        # Components
        self.enhanced_collector = EnhancedNFLDataCollector(self.session, current_season=current_season)
        self.stats_collector = RoleBasedStatsCollector(self.enhanced_collector)
        self.validator = ComprehensiveValidator()

        # Simple cache for roster snapshots
        self.roster_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None

    async def get_upcoming_games(self, days_ahead: int = 14) -> List[GameInfo]:
        """Return upcoming games. This shim returns an empty list; tests handle this path by creating a mock game."""
        return []

    async def get_game_players(self, game: GameInfo) -> List[Any]:
        """Return list of players for a game using cached roster snapshots if available."""
        # Ensure we have a cache for the given week
        await self._ensure_roster_cache(game.week)
        snapshot = self.roster_cache.get(game.home_team)
        if not snapshot:
            return []
        # Return active players for the home team snapshot
        try:
            return snapshot.get_active_players()
        except Exception:
            return []

    async def _ensure_roster_cache(self, week: int) -> None:
        """Populate roster cache for the given week using the enhanced collector."""
        if not self.roster_cache or not self.cache_timestamp or (datetime.now() - self.cache_timestamp) > timedelta(hours=6):
            try:
                self.roster_cache = await self.enhanced_collector.collect_weekly_foundation_data(week)
            except Exception:
                self.roster_cache = {}
            self.cache_timestamp = datetime.now()

    async def _get_fallback_players(self, game: GameInfo) -> List[Any]:
        """Fallback logic if enhanced collector is unavailable. Returns empty list in shim."""
        return []

    async def predict_player_performance(self, player: Any, game: GameInfo) -> Optional[Dict[str, Any]]:
        """Shim prediction method. Returns None to indicate prediction not available in tests."""
        return None
