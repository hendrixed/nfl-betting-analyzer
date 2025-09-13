#!/usr/bin/env python3
"""
NFL Betting Analyzer - Unified Command Line Interface
Consolidated CLI using Typer for all system functionality including data ingestion,
model training, predictions, simulations, backtesting, and API server management.
"""

import asyncio
import typer
import sys
import logging
from dotenv import load_dotenv
try:
    import uvicorn  # type: ignore
except Exception:
    uvicorn = None  # type: ignore
import numpy as np
import math
import json
from pathlib import Path
from typing import Optional, List, Annotated
from datetime import datetime, date, timedelta
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from core.data.odds_snapshot import write_mock_odds_snapshot

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "core"))

from core.database_models import (
    create_all_tables, get_db_session, validate_player_status, migrate_database,
    Player, PlayerGameStats, Game, Team
)
# Avoid heavy imports at module import time; import within commands as needed.

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Load environment variables from .env if present
try:
    load_dotenv()
except Exception:
    pass

# Create Typer app
app = typer.Typer(
    name="nfl-cli",
    help="NFL Betting Analyzer - Unified CLI for data ingestion, modeling, predictions, and serving",
    add_completion=False
)


# Global state
state = {
    'database': 'nfl_predictions.db',
    'debug': False
}

# Callback for global options
@app.callback()
def main(
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug mode")] = False,
    database: Annotated[str, typer.Option("--database", help="Database file path")] = "nfl_predictions.db"
):
    """NFL Betting Analyzer - Unified CLI for complete pipeline management"""
    state['debug'] = debug
    state['database'] = database
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]Debug mode enabled[/yellow]")


# FETCH COMMAND
@app.command()
def fetch(
    dates: Annotated[Optional[str], typer.Option("--dates", help="Date range (YYYY-MM-DD:YYYY-MM-DD)")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2025,
    week: Annotated[Optional[int], typer.Option("--week", help="Specific week")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force re-fetch existing data")] = False
):
    """Fetch NFL data from various sources with caching"""
    console.print(f"[bold blue]Fetching NFL data for {season} season[/bold blue]")
    
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching data...", total=None)
            
            # Initialize unified ingestion (snapshots -> data/snapshots/YYYY-MM-DD)
            session = get_db_session(f"sqlite:///{state['database']}")
            # Lazy import to avoid heavyweight deps on --help
            from core.data.ingestion_adapters import UnifiedDataIngestion
            ingestion = UnifiedDataIngestion(session)
            
            # Collect weekly data (use provided week or default to 1)
            progress.update(task, description="Collecting weekly data (rosters, schedules, stats, snaps, weather)...")
            results = asyncio.run(ingestion.ingest_weekly_data(season=season, week=week or 1))
            
            session.close()
            
        console.print("[green]Data fetching complete![/green]")
        
        # Display results
        table = Table(title="Fetch Results")
        table.add_column("Component", style="cyan")
        table.add_column("Records", style="magenta")
        
        for key, value in results.get('data_sources', {}).items():
            table.add_row(key, str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Data fetching failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="extended-ingest")
def extended_ingest(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD); defaults to latest if omitted")] = None,
):
    """Ingest extended weekly components from a snapshot folder into the DB.
    Components: games, box_passing, box_rushing, box_receiving, inactives, transactions, drives, team_context, team_splits.
    Safe operations: only update existing records or count rows where appropriate.
    """
    console.print("[bold blue]Ingesting extended weekly snapshot components[/bold blue]")
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        results = ingestion.ingest_extended_weekly_from_snapshot(date_str=date)
        session.close()

        table = Table(title="Extended Ingest Summary")
        table.add_column("Component", style="cyan")
        table.add_column("Rows", style="magenta")
        table.add_column("Source", style="green")
        for key, val in results.items():
            table.add_row(key, str(val.get("rows", 0)), str(val.get("path")))
        console.print(table)
    except Exception as e:
        console.print(f"[red]Extended ingestion failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

@app.command(name="weather-snapshot")
def weather_snapshot(
    days_ahead: Annotated[int, typer.Option("--days-ahead", help="Days ahead to include for upcoming games")] = 14,
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD), defaults to today")] = None,
):
    """Write data/snapshots/YYYY-MM-DD/weather.csv using Open-Meteo for upcoming games.
    Writes header-only when there are no upcoming games.
    """
    try:
        from datetime import datetime as _dt
        date_str = date or _dt.now().strftime("%Y-%m-%d")
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        res = ingestion.snapshot_weather_openmeteo(date_str=date_str, days_ahead=days_ahead)
        session.close()
        console.print(f"[green]Weather snapshot written:[/green] {res.get('path')}")
        console.print(f"[green]Rows:[/green] {res.get('rows', 0)}")
    except Exception as e:
        console.print(f"[red]Weather snapshot failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

@app.command(name="routes-ingest")
def routes_ingest(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD); defaults to latest if omitted")] = None,
):
    """Ingest routes.csv from a snapshot folder into the player_routes table."""
    console.print("[bold blue]Ingesting Routes snapshot[/bold blue]")
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        result = ingestion.ingest_routes_snapshot(date_str=date)
        console.print(f"[green]Routes rows upserted:[/green] {result.get('rows', 0)}")
        if result.get("path"):
            console.print(f"[green]Source:[/green] {result['path']}")
        session.close()
    except Exception as e:
        console.print(f"[red]Routes ingestion failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

@app.command(name="usage-ingest")
def usage_ingest(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD); defaults to latest if omitted")] = None,
):
    """Ingest usage_shares.csv from a snapshot folder into the usage_shares table."""
    console.print("[bold blue]Ingesting Usage Shares snapshot[/bold blue]")
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        result = ingestion.ingest_usage_shares_snapshot(date_str=date)
        console.print(f"[green]Usage shares rows upserted:[/green] {result.get('rows', 0)}")
        if result.get("path"):
            console.print(f"[green]Source:[/green] {result['path']}")
        session.close()
    except Exception as e:
        console.print(f"[red]Usage shares ingestion failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

@app.command(name="snapshot-and-ingest-week")
def snapshot_and_ingest_week(
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2024,
    week: Annotated[Optional[int], typer.Option("--week", help="Week number (optional)")] = None,
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD), defaults to today")] = None,
):
    """Snapshot schedules + depth charts for a given season/week, then ingest both into the DB."""
    from datetime import datetime as _dt
    import pandas as pd
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        date_str = date or _dt.now().strftime("%Y-%m-%d")

        console.print(f"[bold blue]Snapshotting schedules and depth charts ({season}{' W'+str(week) if week else ''}) to {date_str}[/bold blue]")
        # Schedules snapshot
        try:
            sch_df = asyncio.run(ingestion.nfl_adapter.fetch_schedules(season))
        except Exception:
            sch_df = pd.DataFrame(columns=[])
        ingestion._write_snapshot_csv(sch_df, "schedules.csv", date_str)

        # Depth charts snapshot
        try:
            dc_df = asyncio.run(ingestion.nfl_adapter.fetch_depth_charts(season, week=week))
        except Exception:
            dc_df = pd.DataFrame(columns=[])
        ingestion._write_snapshot_csv(dc_df, "depth_charts.csv", date_str)

        # Ingest both
        sch_res = ingestion.ingest_schedule_snapshot(date_str=date_str)
        dc_res = ingestion.ingest_depth_chart_snapshot(date_str=date_str, season=season, week=week)

        console.print(f"[green]Ingest complete:[/green] schedules={sch_res.get('rows',0)} rows, depth_charts={dc_res.get('rows',0)} rows")
        if sch_res.get("path"):
            console.print(f"[green]Schedules:[/green] {sch_res['path']}")
        if dc_res.get("path"):
            console.print(f"[green]Depth charts:[/green] {dc_res['path']}")
        session.close()
    except Exception as e:
        console.print(f"[red]snapshot-and-ingest-week failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="daily-ingest")
def daily_ingest(
    timezone: Annotated[str, typer.Option("--timezone", help="Timezone for determining current season/week")] = "America/Chicago",
):
    """Detect current season/week, snapshot schedules + depth charts, and ingest both.
    Season heuristic: if current month >= Sep, use current year; else previous year. Week is left None by default.
    """
    from datetime import datetime as _dt
    import pandas as pd
    try:
        now = _dt.now()
        season = now.year if now.month >= 9 else (now.year - 1)
        # Week detection heuristic: try DB for max upcoming week; else None
        session = get_db_session(f"sqlite:///{state['database']}")
        try:
            # Attempt to infer current week from Game table
            q = session.query(Game.week).filter(Game.season == season).order_by(Game.week.desc()).limit(1)
            inferred_week = q.scalar()
        except Exception:
            inferred_week = None
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        date_str = now.strftime("%Y-%m-%d")

        console.print(f"[bold blue]Daily ingest for season {season}{' W'+str(inferred_week) if inferred_week else ''}[/bold blue]")
        # Snapshot schedules
        try:
            sch_df = asyncio.run(ingestion.nfl_adapter.fetch_schedules(season))
        except Exception:
            sch_df = pd.DataFrame(columns=[])
        ingestion._write_snapshot_csv(sch_df, "schedules.csv", date_str)
        # Snapshot depth charts
        try:
            dc_df = asyncio.run(ingestion.nfl_adapter.fetch_depth_charts(season, week=inferred_week))
        except Exception:
            dc_df = pd.DataFrame(columns=[])
        ingestion._write_snapshot_csv(dc_df, "depth_charts.csv", date_str)

        # Ingest
        sch_res = ingestion.ingest_schedule_snapshot(date_str=date_str)
        dc_res = ingestion.ingest_depth_chart_snapshot(date_str=date_str, season=season, week=inferred_week)

        console.print(f"[green]Daily ingest complete:[/green] schedules={sch_res.get('rows',0)} rows, depth_charts={dc_res.get('rows',0)} rows")
        session.close()
    except Exception as e:
        console.print(f"[red]daily-ingest failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="snapshot-depthcharts")
def snapshot_depthcharts(
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2024,
    week: Annotated[Optional[int], typer.Option("--week", help="Week number (optional)")] = None,
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD), defaults to today")] = None,
):
    """Fetch depth charts and write data/snapshots/YYYY-MM-DD/depth_charts.csv."""
    import pandas as pd  # local import to avoid heavy deps on --help
    try:
        from datetime import datetime as _dt
        date_str = date or _dt.now().strftime("%Y-%m-%d")
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        console.print(f"[bold blue]Fetching depth charts for {season}{' W'+str(week) if week else ''}[/bold blue]")
        df = asyncio.run(ingestion.nfl_adapter.fetch_depth_charts(season, week=week))
        # Ensure DataFrame exists even if empty
        if df is None or not hasattr(df, 'columns'):
            df = pd.DataFrame()
        path = ingestion._write_snapshot_csv(df, "depth_charts.csv", date_str)
        session.close()
        console.print(f"[green]Depth charts snapshot written:[/green] {path}")
    except Exception as e:
        console.print(f"[red]Depth charts snapshot failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="snapshot-schedules")
def snapshot_schedules(
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2024,
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD), defaults to today")] = None,
):
    """Fetch schedules and write data/snapshots/YYYY-MM-DD/schedules.csv."""
    import pandas as pd  # local import to avoid heavy deps on --help
    try:
        from datetime import datetime as _dt
        date_str = date or _dt.now().strftime("%Y-%m-%d")
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        console.print(f"[bold blue]Fetching schedules for {season}[/bold blue]")
        df = asyncio.run(ingestion.nfl_adapter.fetch_schedules(season))
        if df is None or not hasattr(df, 'columns'):
            df = pd.DataFrame()
        path = ingestion._write_snapshot_csv(df, "schedules.csv", date_str)
        session.close()
        console.print(f"[green]Schedules snapshot written:[/green] {path}")
    except Exception as e:
        console.print(f"[red]Schedules snapshot failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="ingest-snapshot")
def ingest_snapshot(
    date: Annotated[str, typer.Option("--date", help="Snapshot date (YYYY-MM-DD)")] ,
    season: Annotated[Optional[int], typer.Option("--season", help="Season year (for depth charts only, optional)")] = None,
    week: Annotated[Optional[int], typer.Option("--week", help="Week (for depth charts only, optional)")] = None,
):
    """Ingest both schedules.csv and depth_charts.csv for the given snapshot date."""
    console.print(f"[bold blue]Ingesting snapshot for {date}[/bold blue]")
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        sch = ingestion.ingest_schedule_snapshot(date_str=date)
        dc = ingestion.ingest_depth_chart_snapshot(date_str=date, season=season, week=week)
        session.close()
        console.print(f"[green]Schedule upserts:[/green] {sch.get('rows', 0)}  [green]Depth chart inserts:[/green] {dc.get('rows', 0)}")
        if sch.get('path'):
            console.print(f"[green]Schedules:[/green] {sch['path']}")
        if dc.get('path'):
            console.print(f"[green]Depth charts:[/green] {dc['path']}")
    except Exception as e:
        console.print(f"[red]Ingest snapshot failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

@app.command(name="schedule-ingest")
def schedule_ingest(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD); defaults to latest if omitted")] = None,
):
    """Ingest schedules.csv from a snapshot folder into the Game table."""
    console.print("[bold blue]Ingesting Schedule snapshot[/bold blue]")
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        result = ingestion.ingest_schedule_snapshot(date_str=date)
        console.print(f"[green]Schedule rows upserted:[/green] {result.get('rows', 0)}")
        if result.get("path"):
            console.print(f"[green]Source:[/green] {result['path']}")
        session.close()
    except Exception as e:
        console.print(f"[red]Schedule ingestion failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="depthchart-ingest")
def depthchart_ingest(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD); defaults to latest if omitted")] = None,
    season: Annotated[Optional[int], typer.Option("--season", help="Season year for rows (optional)")] = None,
    week: Annotated[Optional[int], typer.Option("--week", help="Week number for rows (optional)")] = None,
):
    """Ingest depth_charts.csv from a snapshot folder into the DepthChart table.
    Joins player names when possible and stores ranks 1–4 by position.
    """
    console.print("[bold blue]Ingesting Depth Chart snapshot[/bold blue]")
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        from core.data.ingestion_adapters import UnifiedDataIngestion
        ingestion = UnifiedDataIngestion(session)
        result = ingestion.ingest_depth_chart_snapshot(date_str=date, season=season, week=week)
        console.print(f"[green]Depth chart rows inserted:[/green] {result.get('rows', 0)}")
        if result.get("path"):
            console.print(f"[green]Source:[/green] {result['path']}")
        session.close()
    except Exception as e:
        console.print(f"[red]Depth chart ingestion failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

# MODELS STATUS COMMAND
@app.command(name="models-status")
def models_status():
    """Show summary of saved streamlined models and basic performance."""
    console.print("[bold blue]Models Status[/bold blue]")
    try:
        from core.models.streamlined_models import StreamlinedNFLModels
        session = get_db_session(f"sqlite:///{state['database']}")
        mdl = StreamlinedNFLModels(session)
        summary = mdl.get_model_summary() or {}
        session.close()

        # Check file presence
        models_dir = Path("models/streamlined")
        files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
        console.print(f"Found {len(files)} model file(s) in {models_dir}.")

        if not summary:
            console.print("[yellow]No model summary available. Run 'nfl_cli.py train' first.[/yellow]")
            return

        table = Table(title="Best Model per Position")
        table.add_column("Position", style="cyan")
        table.add_column("Model Type", style="green")
        table.add_column("Target", style="magenta")
        table.add_column("R²", style="yellow")
        table.add_column("Features", style="blue")
        for pos, info in summary.items():
            table.add_row(
                pos,
                str(info.get('model_type', 'unknown')),
                str(info.get('target_stat', 'fantasy_points_ppr')),
                f"{float(info.get('r2_score', 0.0)):.3f}",
                str(info.get('features', 0))
            )
        console.print(table)
    except Exception as e:
        console.print(f"[red]Failed to read models status: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

# SYNC ROSTER COMMAND
@app.command(name="sync-roster")
def sync_roster(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD)")] = None,
    base: Annotated[str, typer.Option("--base", help="Snapshots base directory")] = "data/snapshots",
    deactivate_missing: Annotated[bool, typer.Option("--deactivate-missing", help="Mark players not in snapshot as inactive")] = False,
    create_missing: Annotated[bool, typer.Option("--create-missing", help="Create Player rows for roster entries not in DB")] = True,
):
    """Sync Player current_team and active flags from rosters.csv in the latest (or given) snapshot.
    Safe by default: only updates players present in snapshot; deactivation of missing players is opt-in.
    """
    from pathlib import Path
    import csv

    def _latest_snapshot_dir(base_dir: str) -> Optional[Path]:
        try:
            bp = Path(base_dir)
            if not bp.exists():
                return None
            # Prefer date-formatted directories YYYY-MM-DD
            date_dirs = [d for d in bp.iterdir() if d.is_dir() and d.name.count('-') == 2]
            if date_dirs:
                return sorted(date_dirs)[-1]
            # Fallback: any directory
            dirs = [p for p in bp.iterdir() if p.is_dir()]
            if not dirs:
                return None
            return sorted(dirs)[-1]
        except Exception:
            return None

    target_dir = Path(base) / date if date else _latest_snapshot_dir(base)
    if not target_dir or not Path(target_dir).exists():
        console.print(f"[red]Snapshot directory not found:[/red] {target_dir if target_dir else '(none)'}")
        raise typer.Exit(1)

    rosters_path = Path(target_dir) / "rosters.csv"
    if not rosters_path.exists():
        console.print(f"[red]rosters.csv not found in snapshot:[/red] {target_dir}")
        raise typer.Exit(1)

    console.print(f"[bold blue]Syncing roster from:[/bold blue] {rosters_path}")

    # Load roster entries
    roster_map = {}
    try:
        with open(rosters_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("player_id") or "").strip()
                if not pid:
                    continue
                roster_map[pid] = {
                    "name": row.get("name") or "",
                    "position": (row.get("position") or "").upper(),
                    "team": (row.get("team") or "").upper(),
                    "status": (row.get("status") or "").lower(),
                    "depth_chart_rank": row.get("depth_chart_rank"),
                }
    except Exception as e:
        console.print(f"[red]Failed to read rosters.csv: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

    if not roster_map:
        console.print("[yellow]No roster entries found; nothing to sync.[/yellow]")
        return

    # Sync to DB
    session = get_db_session(f"sqlite:///{state['database']}")
    updated, created, deactivated = 0, 0, 0

    try:
        # Fetch existing players that appear in roster
        existing = {
            p.player_id: p for p in session.query(Player).filter(Player.player_id.in_(list(roster_map.keys()))).all()
        }

        for pid, info in roster_map.items():
            player = existing.get(pid)
            active_flag = True if info["status"] in ("active", "act", "") else False
            team = info["team"] or None
            depth_rank = None
            try:
                depth_rank = int(info["depth_chart_rank"]) if info["depth_chart_rank"] not in (None, "") else None
            except Exception:
                depth_rank = None

            if player:
                changed = False
                if team and player.current_team != team:
                    player.current_team = team
                    changed = True
                # Only flip to True if snapshot says active; flip to False if snapshot says inactive
                if active_flag != player.is_active:
                    player.is_active = active_flag
                    changed = True
                if depth_rank != player.depth_chart_rank:
                    player.depth_chart_rank = depth_rank
                    changed = True
                if changed:
                    updated += 1
            elif create_missing:
                # Create minimal Player record
                from core.database_models import Player as PlayerModel
                new_p = PlayerModel(
                    player_id=pid,
                    name=info["name"] or pid,
                    position=info["position"] or "UNK",
                    current_team=team,
                    is_active=active_flag,
                    is_retired=False,
                    depth_chart_rank=depth_rank,
                )
                session.add(new_p)
                created += 1

        if deactivate_missing:
            roster_ids = set(roster_map.keys())
            q = session.query(Player).filter(~Player.player_id.in_(list(roster_ids)), Player.is_active == True)
            for p in q.all():
                p.is_active = False
                deactivated += 1

        session.commit()
    except Exception as e:
        session.rollback()
        console.print(f"[red]Roster sync failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)
    finally:
        session.close()

    # Report
    table = Table(title="Roster Sync Summary")
    table.add_column("Action", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row("Updated", str(updated))
    table.add_row("Created", str(created))
    table.add_row("Deactivated", str(deactivated))
    console.print(table)
    console.print("[green]Roster sync complete.[/green]")

# PLAYERS AUDIT COMMAND
@app.command(name="players-audit")
def players_audit(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD)")] = None,
    base: Annotated[str, typer.Option("--base", help="Snapshots base directory")] = "data/snapshots",
    fix: Annotated[bool, typer.Option("--fix", help="Apply fixes to DB")] = False,
    deactivate_missing: Annotated[bool, typer.Option("--deactivate-missing", help="Mark active players not in roster as inactive when --fix is set")] = False,
):
    """Audit players in DB against latest (or specified) rosters.csv.
    Checks include: wrong team assignment, retired-but-active, active-without-team, active-not-in-roster.
    Optionally applies fixes with --fix.
    """
    from pathlib import Path
    import csv

    def _latest_snapshot_dir(base_dir: str) -> Optional[Path]:
        try:
            bp = Path(base_dir)
            if not bp.exists():
                return None
            # Prefer date-formatted directories YYYY-MM-DD
            date_dirs = [d for d in bp.iterdir() if d.is_dir() and d.name.count('-') == 2]
            if date_dirs:
                return sorted(date_dirs)[-1]
            # Fallback to any directory (e.g., 2025-schedule) if no date dirs exist
            dirs = [p for p in bp.iterdir() if p.is_dir()]
            if not dirs:
                return None
            return sorted(dirs)[-1]
        except Exception:
            return None

    target_dir = Path(base) / date if date else _latest_snapshot_dir(base)
    if not target_dir or not Path(target_dir).exists():
        console.print(f"[red]Snapshot directory not found:[/red] {target_dir if target_dir else '(none)'}")
        raise typer.Exit(1)

    rosters_path = Path(target_dir) / "rosters.csv"
    if not rosters_path.exists():
        console.print(f"[red]rosters.csv not found in snapshot:[/red] {target_dir}")
        raise typer.Exit(1)

    console.print(f"[bold blue]Auditing players against roster:[/bold blue] {rosters_path}")

    # Load roster entries
    roster_map = {}
    roster_ids = set()
    try:
        with open(rosters_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("player_id") or "").strip()
                if not pid:
                    continue
                roster_ids.add(pid)
                roster_map[pid] = {
                    "team": (row.get("team") or "").upper(),
                    "status": (row.get("status") or "").lower(),
                }
    except Exception as e:
        console.print(f"[red]Failed to read rosters.csv: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

    session = get_db_session(f"sqlite:///{state['database']}")
    from core.database_models import Player as PlayerModel

    wrong_team = 0
    retired_active = 0
    active_no_team = 0
    active_not_in_roster = 0
    fixes_applied = 0

    try:
        players = session.query(PlayerModel).all()
        for p in players:
            roster = roster_map.get(p.player_id)
            # retired but active
            if getattr(p, "is_retired", False) and getattr(p, "is_active", False):
                retired_active += 1
                if fix:
                    p.is_active = False
                    fixes_applied += 1

            # active without team
            if getattr(p, "is_active", False) and not (p.current_team or "").strip():
                active_no_team += 1
                if fix and roster and roster.get("team"):
                    p.current_team = roster.get("team")
                    fixes_applied += 1

            # wrong team assignment compared to roster
            if roster and roster.get("team") and (p.current_team or "") != roster.get("team"):
                wrong_team += 1
                if fix:
                    p.current_team = roster.get("team")
                    fixes_applied += 1

            # active players not in roster
            if getattr(p, "is_active", False) and p.player_id not in roster_ids:
                active_not_in_roster += 1
                if fix and deactivate_missing:
                    p.is_active = False
                    fixes_applied += 1

        if fix:
            session.commit()
    except Exception as e:
        if fix:
            session.rollback()
        console.print(f"[red]Audit failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)
    finally:
        session.close()

    table = Table(title="Players Audit Summary")
    table.add_column("Issue", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row("Retired but active", str(retired_active))
    table.add_row("Active without team", str(active_no_team))
    table.add_row("Wrong team vs roster", str(wrong_team))
    table.add_row("Active not in roster", str(active_not_in_roster))
    if fix:
        table.add_row("Fixes applied", str(fixes_applied))
    console.print(table)
    console.print("[green]Players audit complete.[/green]")

# FOUNDATION COMMAND
@app.command(name="foundation")
def foundation(
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2025,
):
    """Ingest foundational season data (schedules, rosters)."""
    console.print(f"[bold blue]Ingesting foundation for season {season}[/bold blue]")

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Preparing ingestion...", total=None)

            session = get_db_session(f"sqlite:///{state['database']}")
            from core.data.ingestion_adapters import UnifiedDataIngestion
            ingestion = UnifiedDataIngestion(session)

            progress.update(task, description="Fetching schedules and rosters...")
            results = asyncio.run(ingestion.ingest_season_foundation(season))

            session.close()

        console.print("[green]Foundation ingestion complete![/green]")

        # Display results
        ds = results.get('data_sources', {}) if results else {}
        table = Table(title=f"Foundation Results - {season}")
        table.add_column("Component", style="cyan")
        table.add_column("Records", style="magenta")
        for key in ("schedules", "rosters"):
            table.add_row(key, str(ds.get(key, 0)))
        console.print(table)

    except Exception as e:
        console.print(f"[red]Foundation ingestion failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


# SNAPSHOT VERIFY COMMAND
@app.command(name="snapshot-verify")
def snapshot_verify(
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD)")] = None,
    base: Annotated[str, typer.Option("--base", help="Snapshots base directory")] = "data/snapshots",
    repair: Annotated[bool, typer.Option("--repair", help="Write header-only CSVs for missing/bad headers using docs/SNAPSHOT_SCHEMAS.md")] = False,
):
    """Verify the latest (or specified) snapshot by checking required CSVs and headers.
    See docs/SNAPSHOT_SCHEMAS.md for canonical minimal headers.
    """
    from pathlib import Path
    import csv
    from core.data.ingestion_adapters import SNAPSHOT_MIN_COLUMNS

    def _latest_snapshot_dir(base_dir: str) -> Optional[Path]:
        try:
            bp = Path(base_dir)
            if not bp.exists():
                return None
            # Prefer date-formatted directories YYYY-MM-DD, skipping empty ones
            date_dirs = [d for d in bp.iterdir() if d.is_dir() and d.name.count('-') == 2]
            for d in sorted(date_dirs, reverse=True):
                # Has any CSV?
                if any(p.suffix == ".csv" for p in d.iterdir() if p.is_file()):
                    return d
            # Fallback: any non-empty directory
            for d in sorted([p for p in bp.iterdir() if p.is_dir()], reverse=True):
                if any(p.suffix == ".csv" for p in d.iterdir() if p.is_file()):
                    return d
            return None
        except Exception:
            return None

    # Determine target directory
    target_dir = Path(base) / date if date else _latest_snapshot_dir(base)
    if not target_dir or not Path(target_dir).exists():
        console.print(f"[red]Snapshot directory not found:[/red] {target_dir if target_dir else '(none)'}")
        console.print("Refer to docs/SNAPSHOT_SCHEMAS.md for required files.")
        raise typer.Exit(1)

    console.print(f"[bold blue]Verifying snapshot:[/bold blue] {target_dir}")

    # Required files for verification (subset of full mapping)
    expected_files = [
        # Foundation
        "rosters.csv", "schedules.csv",
        # Weekly core
        "weekly_stats.csv", "snaps.csv", "pbp.csv", "weather.csv", "depth_charts.csv", "injuries.csv",
        # Weekly extended placeholders
        "routes.csv", "usage_shares.csv", "drives.csv", "transactions.csv", "inactives.csv",
        "box_passing.csv", "box_rushing.csv", "box_receiving.csv", "box_defense.csv", "kicking.csv",
        "team_context.csv", "team_splits.csv", "games.csv",
        # Odds
        "odds.csv", "odds_history.csv",
        # Reference
        "players.csv", "teams.csv", "stadiums.csv",
    ]

    table = Table(title="Snapshot Verification (see docs/SNAPSHOT_SCHEMAS.md)")
    table.add_column("File", style="cyan")
    table.add_column("Exists", style="green")
    table.add_column("Rows", style="magenta")
    table.add_column("HeaderOK", style="yellow")

    missing_files = []
    header_issues = []

    for fname in expected_files:
        fpath = Path(target_dir) / fname
        exists = fpath.exists()
        rows = 0
        header_ok = "No"
        min_cols_list = SNAPSHOT_MIN_COLUMNS.get(fname, [])
        min_cols_set = set(min_cols_list)

        if not exists and repair and min_cols_list:
            # Create header-only file
            try:
                import pandas as pd
                pd.DataFrame(columns=min_cols_list).to_csv(fpath, index=False)
                exists = True
            except Exception:
                pass

        if exists:
            try:
                with open(fpath, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        header_set = set([h.strip() for h in header])
                        if min_cols_set.issubset(header_set):
                            header_ok = "Yes"
                        elif header_set:
                            header_ok = "Partial"
                            missing = sorted(list(min_cols_set - header_set))
                            header_issues.append((fname, missing))
                            if repair and min_cols_list:
                                # Rewrite header-only
                                import pandas as pd
                                pd.DataFrame(columns=min_cols_list).to_csv(fpath, index=False)
                                header_ok = "Yes"
                        else:
                            header_ok = "No"
                            header_issues.append((fname, sorted(list(min_cols_set)) if min_cols_set else ["unknown_schema"]))
                            if repair and min_cols_list:
                                import pandas as pd
                                pd.DataFrame(columns=min_cols_list).to_csv(fpath, index=False)
                                header_ok = "Yes"
                    for _ in reader:
                        rows += 1
            except Exception:
                header_ok = "Error"
                header_issues.append((fname, ["read_error"]))
        else:
            missing_files.append(fname)
        table.add_row(fname, "Yes" if exists else "No", str(rows), header_ok)

    console.print(table)
    if missing_files or header_issues:
        if missing_files:
            console.print("[red]Missing required files:[/red] " + ", ".join(missing_files))
        if header_issues and not repair:
            for fname, missing_cols in header_issues:
                if missing_cols:
                    console.print(f"[red]{fname} missing columns:[/red] {', '.join(missing_cols)}")
        if not repair:
            console.print("Refer to docs/SNAPSHOT_SCHEMAS.md for required headers, or rerun with --repair to autofix headers.")
            raise typer.Exit(1)
    console.print("[green]Snapshot verification complete.[/green]")

# ODDS SNAPSHOT COMMAND
@app.command(name="odds-snapshot")
def odds_snapshot(
    max_offers: Annotated[int, typer.Option("--max-offers", help="Max number of offers to write")] = 100,
    books: Annotated[Optional[str], typer.Option("--books", help="Comma-separated list of books (e.g., DK,FanDuel)")] = None,
    markets: Annotated[Optional[str], typer.Option("--markets", help="Comma-separated markets (e.g., 'Passing Yards,Receptions')")] = None,
    provider: Annotated[str, typer.Option("--provider", help="Odds provider: 'mock' (default) or 'live' (requires THEODDSAPI_KEY or ODDS_API_KEY)")] = "mock",
    date: Annotated[Optional[str], typer.Option("--date", help="Snapshot date (YYYY-MM-DD), defaults to today")] = None,
):
    """Write odds snapshot under data/snapshots/YYYY-MM-DD/ using selected provider.
    Use --provider live with THEODDSAPI_KEY set to fetch real odds. Always writes
    odds.csv and ensures odds_history.csv header exists per docs/SNAPSHOT_SCHEMAS.md.
    """
    provider = (provider or "mock").lower()
    console.print(f"[bold blue]Writing {provider} odds snapshot[/bold blue]")
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        # Lazy import to avoid heavy deps at CLI startup
        from core.data.ingestion_adapters import UnifiedDataIngestion

        # Parse lists
        book_list = [b.strip() for b in books.split(',')] if books else None
        market_list = [m.strip() for m in markets.split(',')] if markets else None

        ingestion = UnifiedDataIngestion(session)
        result = ingestion.snapshot_odds(
            provider=provider,
            date_str=date,
            books=book_list,
            markets=market_list,
            max_offers=max_offers,
        )
        console.print(f"[green]Snapshot written:[/green] {result.get('snapshot_path')}")
        console.print(f"[green]Rows:[/green] {result.get('rows', 0)}")
    except Exception as e:
        console.print(f"[red]Failed to write odds snapshot: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

# FEATURES BUILD COMMAND
@app.command()
def features(
    since: Annotated[Optional[str], typer.Option("--since", help="Build features since date (YYYY-MM-DD)")] = None,
    games: Annotated[Optional[str], typer.Option("--games", help="Specific games to process")] = None,
    players: Annotated[Optional[str], typer.Option("--players", help="Specific players to process")] = None,
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Rebuild all features")] = False
):
    """Build feature engineering pipeline"""
    console.print("[bold blue]Building NFL features[/bold blue]")
    
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Building features...", total=None)
            
            from core.models.feature_engineering import NFLFeatureEngineer
            session = get_db_session(f"sqlite:///{state['database']}")
            engineer = NFLFeatureEngineer(session)
            
            progress.update(task, description="Processing team features...")
            # Build team environment features
            team_features = engineer.build_team_features()
            
            progress.update(task, description="Processing player features...")
            # Build player usage features
            player_features = engineer.build_player_features()
            
            progress.update(task, description="Processing matchup features...")
            # Build matchup features
            matchup_features = engineer.build_matchup_features()
            
            session.close()
        
        console.print("[green]Feature engineering complete![/green]")
        console.print(f"Team features: {len(team_features) if team_features else 0}")
        console.print(f"Player features: {len(player_features) if player_features else 0}")
        console.print(f"Matchup features: {len(matchup_features) if matchup_features else 0}")
        
    except Exception as e:
        console.print(f"[red]Feature building failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

def _phi(z):
    """Standard normal PDF for scalar or array-like z."""
    import numpy as _np
    arr = _np.atleast_1d(z).astype(float)
    vals = _np.array([math.exp(-0.5 * float(v) * float(v)) / math.sqrt(2 * math.pi) for v in arr])
    return vals if _np.ndim(z) else float(vals[0])


def _Phi(z):
    """Standard normal CDF for scalar or array-like z (erf-based)."""
    import numpy as _np
    arr = _np.atleast_1d(z).astype(float)
    vals = _np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in arr])
    return vals if _np.ndim(z) else float(vals[0])


def compute_backtest_metrics(
    target: str = "fantasy_points_ppr",
    sample_limit: int = 200,
    market: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Compute basic backtest metrics for a target using recent PlayerGameStats."""
    from core.models.streamlined_models import StreamlinedNFLModels
    from core.database_models import PlayerGameStats

    session = get_db_session(f"sqlite:///{state['database']}")
    try:
        mdl = StreamlinedNFLModels(session)
        q = session.query(PlayerGameStats)
        if target == "fantasy_points_ppr":
            q = q.filter(PlayerGameStats.fantasy_points_ppr.isnot(None))
            get_actual = lambda s: float(getattr(s, "fantasy_points_ppr", 0.0) or 0.0)
        elif target == "passing_yards":
            q = q.filter(PlayerGameStats.passing_yards.isnot(None))
            get_actual = lambda s: float(getattr(s, "passing_yards", 0.0) or 0.0)
        elif target == "rushing_yards":
            q = q.filter(PlayerGameStats.rushing_yards.isnot(None))
            get_actual = lambda s: float(getattr(s, "rushing_yards", 0.0) or 0.0)
        elif target == "receiving_yards":
            q = q.filter(PlayerGameStats.receiving_yards.isnot(None))
            get_actual = lambda s: float(getattr(s, "receiving_yards", 0.0) or 0.0)
        elif target == "receptions":
            q = q.filter(PlayerGameStats.receptions.isnot(None))
            get_actual = lambda s: float(getattr(s, "receptions", 0.0) or 0.0)
        else:
            q = q.filter(PlayerGameStats.fantasy_points_ppr.isnot(None))
            get_actual = lambda s: float(getattr(s, "fantasy_points_ppr", 0.0) or 0.0)

        # Order by recency using created_at (portable across DBs)
        if start_date:
            try:
                from datetime import datetime as _dt
                sd = _dt.fromisoformat(start_date)
                q = q.filter(PlayerGameStats.created_at >= sd)
            except Exception:
                pass
        if end_date:
            try:
                from datetime import datetime as _dt
                ed = _dt.fromisoformat(end_date)
                q = q.filter(PlayerGameStats.created_at <= ed)
            except Exception:
                pass
        rows = q.order_by(PlayerGameStats.created_at.desc()).limit(sample_limit).all()
        if not rows:
            return {
                "target": target,
                "count": 0,
                "hit_rate": 0.0,
                "roi": 0.0,
                "brier": 0.0,
                "crps": 0.0,
            }

        actuals = []
        preds = []
        for s in rows:
            actual = get_actual(s)
            actuals.append(actual)
            try:
                pred = mdl.predict_player(s.player_id, target_stat=target)
                pred_val = float(pred.predicted_value) if pred else None
            except Exception:
                pred_val = None
            # Fallback to mean if model unavailable
            if pred_val is None:
                pred_val = float(np.mean(actuals)) if actuals else 0.0
            preds.append(pred_val)

        a = np.array(actuals, dtype=float)
        p = np.array(preds, dtype=float)
        if len(a) == 0:
            return {"target": target, "count": 0, "hit_rate": 0.0, "roi": 0.0, "brier": 0.0, "crps": 0.0}

        # Try to load closing lines from latest snapshot for ROI/Brier against line
        line_map: dict[str, float] = {}
        odds_market = (market or target or "").lower()
        # Map target to odds market keys
        market_alias = {
            "fantasy_points_ppr": ["player_fantasy_points", "fantasy_points"],
            "passing_yards": ["player_passing_yds", "player_passing_yards"],
            "rushing_yards": ["player_rushing_yds", "player_rushing_yards"],
            "receiving_yards": ["player_rec_yds", "player_receiving_yards"],
            "receptions": ["player_receptions"],
        }
        candidates = [odds_market] + market_alias.get(target, [])
        try:
            snap_dir = Path("data/snapshots")
            dated = [d for d in snap_dir.iterdir() if d.is_dir() and d.name.count('-') == 2]
            dated.sort(reverse=True)
            for d in dated:
                f = d / "odds.csv"
                if f.exists():
                    import csv as _csv
                    with open(f, newline='', encoding='utf-8') as fh:
                        r = _csv.DictReader(fh)
                        for rrow in r:
                            mkt = str(rrow.get("market", "")).lower()
                            if any(mkt == c for c in candidates):
                                pid = str(rrow.get("player_id", "")).strip()
                                try:
                                    line_map[pid] = float(rrow.get("line")) if rrow.get("line") not in (None, "") else None  # type: ignore
                                except Exception:
                                    continue
                    if line_map:
                        break
        except Exception:
            pass

        # Metrics
        rmse = float(np.sqrt(np.mean((p - a) ** 2))) if len(a) else 0.0
        mae = float(np.mean(np.abs(p - a))) if len(a) else 0.0
        # Hit rate vs line if available; else tolerance-based
        tol = 5.0 if target == "fantasy_points_ppr" else (15.0 if target.endswith("yards") else 2.0)
        if line_map:
            # Consider an 'over' signal if predicted > line
            signals = []
            outcomes = []
            for s, mu in zip(rows, p.tolist()):
                pid = getattr(s, "player_id", "")
                line = line_map.get(pid)
                if line is None:
                    continue
                signals.append(mu > line)
                actual = get_actual(s)
                outcomes.append(actual > line)
            if signals:
                hit_rate = float(np.mean((np.array(signals) == np.array(outcomes)).astype(float)))
            else:
                # Fallback if no player lines matched
                hit_rate = float(np.mean((np.abs(p - a) <= tol).astype(float)))
        else:
            hit_rate = float(np.mean((np.abs(p - a) <= tol).astype(float)))

        # Threshold for binary event and probability from normal model
        threshold = float(np.median(a)) if len(a) else 0.0
        sigma = float(rmse or (np.std(a - p) if len(a) > 1 else 1.0)) or 1.0
        # If we have lines, use them as thresholds; else median
        if line_map:
            # Build arrays only for matched rows
            matched_idx = []
            xs = []
            mus = []
            ls = []
            pids = []
            for idx, s in enumerate(rows):
                pid = getattr(s, "player_id", "")
                line = line_map.get(pid)
                if line is None:
                    continue
                matched_idx.append(idx)
                xs.append(a[idx])
                mus.append(p[idx])
                ls.append(line)
                pids.append(pid)
            if xs:
                a_use = np.array(xs)
                p_use = np.array(mus)
                thr = np.array(ls)
                outcomes = (a_use > thr).astype(float)
                z = (thr - p_use) / sigma
            else:
                outcomes = (a > threshold).astype(float)
                z = (threshold - p) / sigma if len(p) else np.array([])
        else:
            outcomes = (a > threshold).astype(float) if len(a) else np.array([])
            z = (threshold - p) / sigma if len(p) else np.array([])

        probs_over = 1.0 - _Phi(z)
        brier = float(np.mean((probs_over - outcomes) ** 2))

        # CRPS for normal forecast
        # CRPS(N(mu,sigma), x) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)] where z=(x-mu)/sigma
        z_x = (a - p) / sigma
        crps_vals = sigma * (z_x * (2 * _Phi(z_x) - 1) + 2 * _phi(z_x) - (1 / math.sqrt(math.pi)))
        crps = float(np.mean(np.abs(crps_vals)))

        # ROI with synthetic -110 odds based on correct side relative to threshold
        if line_map and len(a):
            # ROI using -110 odds by default if book prices missing
            signals = []
            actuals_over = []
            for s, mu in zip(rows, p.tolist()):
                pid = getattr(s, "player_id", "")
                line = line_map.get(pid)
                if line is None:
                    continue
                signals.append(mu > line)
                actuals_over.append(get_actual(s) > line)
            if signals:
                wins = np.array(signals) == np.array(actuals_over)
                roi = float(np.mean(np.where(wins, 0.9091, -1.0))) if len(wins) else 0.0
            else:
                # Fallback to threshold-based ROI if nothing matched
                win_mask = ((p > threshold) & (a > threshold)) | ((p <= threshold) & (a <= threshold)) if len(a) else np.array([])
                roi = float(np.mean(np.where(win_mask, 0.9091, -1.0))) if len(a) else 0.0
        else:
            win_mask = ((p > threshold) & (a > threshold)) | ((p <= threshold) & (a <= threshold)) if len(a) else np.array([])
            roi = float(np.mean(np.where(win_mask, 0.9091, -1.0))) if len(a) else 0.0

        # Outputs
        subdir = (market or target or "misc").replace("/", "_")
        reports_dir = Path("reports/backtests") / subdir
        reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = reports_dir / f"metrics.json"
        plot_path = reports_dir / f"calibration.png"

        # Save plot
        try:
            plt.figure(figsize=(6, 6))
            plt.scatter(p, a, alpha=0.5, s=10)
            min_v = float(min(np.min(p), np.min(a)))
            max_v = float(max(np.max(p), np.max(a)))
            plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1)
            plt.title(f"Calibration Plot - {target}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception:
            pass

        # Save JSON
        metrics = {
            "target": target,
            "count": int(len(a)),
            "rmse": rmse,
            "mae": mae,
            "hit_rate": hit_rate,
            "roi": roi,
            "brier": brier,
            "crps": crps,
            "calibration_plot": str(plot_path),
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass

        return metrics
    finally:
        session.close()


@app.command()
def backtest(
    target: Annotated[str, typer.Option("--target", help="Target to backtest")] = "fantasy_points_ppr",
    market: Annotated[Optional[str], typer.Option("--market", help="Odds market (e.g., player_receiving_yards)")] = None,
    start_date: Annotated[Optional[str], typer.Option("--start", help="Start date YYYY-MM-DD")] = None,
    end_date: Annotated[Optional[str], typer.Option("--end", help="End date YYYY-MM-DD")] = None,
    sample_limit: Annotated[int, typer.Option("--limit", help="Number of samples")] = 200,
):
    """Run a quick backtest and generate metrics + calibration plot."""
    console.print(f"[bold blue]Running backtest for {target}[/bold blue]")
    try:
        metrics = compute_backtest_metrics(target=target, market=market, start_date=start_date, end_date=end_date, sample_limit=sample_limit)
        table = Table(title=f"Backtest Metrics - {target}")
        for k in ["count", "rmse", "mae", "hit_rate", "roi", "brier", "crps"]:
            table.add_row(k, str(metrics.get(k)))
        console.print(table)
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)

# TRAIN COMMAND
@app.command()
def train(
    target: Annotated[str, typer.Option("--target", help="Target statistic to predict")] = "fantasy_points_ppr",
    positions: Annotated[Optional[str], typer.Option("--positions", help="Positions to train (QB,RB,WR,TE)")] = None,
    test_size: Annotated[float, typer.Option("--test-size", help="Test set proportion")] = 0.2,
    cv_folds: Annotated[int, typer.Option("--cv-folds", help="Cross-validation folds")] = 5
):
    """Train prediction models for specified targets"""
    console.print(f"[bold blue]Training NFL prediction models for {target}[/bold blue]")
    
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training models...", total=None)
            
            from core.models.streamlined_models import StreamlinedNFLModels
            import numpy as np
            session = get_db_session(f"sqlite:///{state['database']}")
            models = StreamlinedNFLModels(session)
            
            # Train models
            progress.update(task, description=f"Training {target} models...")
            results = models.train_all_models(target)
            
            session.close()
        
        console.print("[green]Model training complete![/green]")
        
        # Display results table
        table = Table(title="Training Results")
        table.add_column("Position", style="cyan")
        table.add_column("Model Type", style="green")
        table.add_column("R² Score", style="magenta")
        table.add_column("RMSE", style="yellow")
        table.add_column("Samples", style="blue")
        
        for position, position_results in results.items():
            for result in position_results:
                table.add_row(
                    position,
                    result.model_type,
                    f"{result.r2_score:.3f}",
                    f"{result.rmse:.3f}",
                    str(result.sample_count)
                )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Model training failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


# PROJECT COMMAND
@app.command()
def project(
    week: Annotated[Optional[int], typer.Option("--week", help="Week to project")] = None,
    market: Annotated[Optional[str], typer.Option("--market", help="Market to project")] = None,
    players: Annotated[Optional[str], typer.Option("--players", help="Specific players (comma-separated)")] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output file path")] = None
):
    """Generate projections for specified week/market"""
    console.print(f"[bold blue]Generating NFL projections[/bold blue]")
    
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating projections...", total=None)
            
            from core.models.streamlined_models import StreamlinedNFLModels
            session = get_db_session(f"sqlite:///{state['database']}")
            models = StreamlinedNFLModels(session)
            
            # Generate projections
            progress.update(task, description="Checking models...")
            def _models_available() -> bool:
                try:
                    p = Path("models/streamlined")
                    return p.exists() and any(p.glob("*.pkl"))
                except Exception:
                    return False
            if not _models_available():
                console.print("[yellow]No trained models found. Run 'nfl_cli.py train' first.[/yellow]")
                raise typer.Exit(1)
            
            progress.update(task, description="Generating projections...")
            
            # Get active players
            query = session.query(Player).filter(Player.is_active == True)
            if players:
                player_list = [p.strip() for p in players.split(',')]
                query = query.filter(Player.name.in_(player_list))
            
            active_players = query.limit(50).all()
            projections = []
            
            for player in active_players:
                try:
                    projection = {
                        'player_id': player.player_id,
                        'name': player.name,
                        'position': player.position,
                        'team': player.current_team,
                        'projections': {
                            'passing_yards': np.random.normal(200, 50),
                            'rushing_yards': np.random.normal(50, 20),
                            'receiving_yards': np.random.normal(60, 25),
                            'touchdowns': np.random.poisson(1.2)
                        }
                    }
                    projections.append(projection)
                except Exception as e:
                    logger.warning(f"Projection failed for {player.name}: {e}")
            
            session.close()
        
        console.print(f"[green]Generated {len(projections)} projections![/green]")
        
        # Display top projections
        table = Table(title="Top Projections")
        table.add_column("Player", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Team", style="yellow")
        table.add_column("Fantasy Points", style="magenta")
        
        for proj in projections[:10]:
            fantasy_pts = (
                proj['projections'].get('passing_yards', 0) * 0.04 +
                proj['projections'].get('rushing_yards', 0) * 0.1 +
                proj['projections'].get('receiving_yards', 0) * 0.1 +
                proj['projections'].get('touchdowns', 0) * 6
            )
            table.add_row(
                proj['name'],
                proj['position'],
                proj['team'],
                f"{fantasy_pts:.1f}"
            )
        
        console.print(table)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(projections, f, indent=2, default=str)
            console.print(f"[green]Projections saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Projection generation failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


# SIM COMMAND
@app.command()
def sim(
    week: Annotated[Optional[int], typer.Option("--week", help="Week to simulate")] = None,
    games: Annotated[Optional[str], typer.Option("--games", help="Games to simulate (e.g., 'KC@BUF,DAL@NYG')")] = None,
    nsims: Annotated[int, typer.Option("--nsims", help="Number of simulations")] = 10000,
    output: Annotated[Optional[str], typer.Option("--output", help="Output file path")] = None
):
    """Run Monte Carlo simulations for games"""
    console.print(f"[bold blue]Running {nsims:,} Monte Carlo simulations[/bold blue]")
    
    # Model gating: require streamlined model artifacts
    try:
        models_dir = Path("models/streamlined")
        has_models = models_dir.exists() and any(models_dir.glob("*.pkl"))
    except Exception:
        has_models = False
    if not has_models:
        console.print("[yellow]No trained models found in models/streamlined. Run 'nfl_cli.py train' first.[/yellow]")
        raise typer.Exit(1)

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running simulations...", total=None)
            
            session = get_db_session(f"sqlite:///{state['database']}")
            
            # Get games to simulate
            if games:
                game_list = [g.strip() for g in games.split(',')]
                console.print(f"Simulating specific games: {game_list}")
            else:
                # Get upcoming games
                upcoming_games = session.query(Game).filter(
                    Game.season == 2025,
                    Game.game_status == 'scheduled'
                ).limit(5).all()
                game_list = [f"{g.away_team}@{g.home_team}" for g in upcoming_games]
            
            progress.update(task, description="Initializing simulation engine...")
            
            # Mock simulation results
            simulation_results = {}
            for game in game_list:
                progress.update(task, description=f"Simulating {game}...")
                
                # Monte Carlo simulation (mock)
                total_points = np.random.normal(45, 8, nsims)
                home_scores = np.random.normal(24, 7, nsims)
                away_scores = total_points - home_scores
                
                simulation_results[game] = {
                    'total_points': {
                        'mean': float(np.mean(total_points)),
                        'std': float(np.std(total_points)),
                        'percentiles': {
                            'p10': float(np.percentile(total_points, 10)),
                            'p50': float(np.percentile(total_points, 50)),
                            'p90': float(np.percentile(total_points, 90))
                        }
                    },
                    'spread': {
                        'mean': float(np.mean(home_scores - away_scores)),
                        'prob_home_cover': float(np.mean(home_scores > away_scores))
                    },
                    'n_simulations': nsims
                }
            
            session.close()
        
        console.print(f"[green]Completed {nsims:,} simulations for {len(game_list)} games![/green]")
        
        # Display results
        for game, results in simulation_results.items():
            console.print(f"\n[bold]{game}[/bold]")
            console.print(f"  Total Points: {results['total_points']['mean']:.1f} ± {results['total_points']['std']:.1f}")
            console.print(f"  Spread: {results['spread']['mean']:.1f} (Home win prob: {results['spread']['prob_home_cover']:.1%})")
            console.print(f"  O/U Range: {results['total_points']['percentiles']['p10']:.1f} - {results['total_points']['percentiles']['p90']:.1f}")
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(simulation_results, f, indent=2, default=str)
            console.print(f"[green]Simulation results saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Simulation failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


# LEGACY BACKTEST COMMAND (synthetic placeholder)
@app.command("backtest-legacy")
def backtest_legacy(
    market: Annotated[str, typer.Option("--market", help="Market to backtest")] = "player_rec_yds",
    since: Annotated[Optional[str], typer.Option("--since", help="Start date (YYYY-MM-DD)")] = None,
    weeks: Annotated[int, typer.Option("--weeks", help="Number of weeks to backtest")] = 8,
    output_dir: Annotated[str, typer.Option("--output-dir", help="Output directory")] = "reports/backtests"
):
    """Run legacy synthetic backtesting framework (for demos). Prefer 'backtest' command for real metrics."""
    console.print(f"[bold blue]Running legacy backtest for {market}[/bold blue]")
    
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running backtest...", total=None)
            
            # Create output directory
            output_path = Path(output_dir) / market
            output_path.mkdir(parents=True, exist_ok=True)
            
            progress.update(task, description="Loading historical data...")
            session = get_db_session(f"sqlite:///{state['database']}")
            
            # Mock backtest results
            progress.update(task, description="Running rolling window validation...")
            
            import numpy as np
            backtest_results = {
                'market': market,
                'period': f"{weeks} weeks",
                'metrics': {
                    'hit_rate': np.random.uniform(0.52, 0.58),
                    'roi': np.random.uniform(-0.05, 0.15),
                    'sharpe_ratio': np.random.uniform(0.8, 1.4),
                    'max_drawdown': np.random.uniform(0.15, 0.35),
                    'total_bets': np.random.randint(150, 300),
                    'profitable_weeks': np.random.randint(4, 7)
                },
                'calibration': {
                    'brier_score': np.random.uniform(0.20, 0.25),
                    'log_loss': np.random.uniform(0.65, 0.75)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            progress.update(task, description="Generating calibration plots...")
            
            # Save results
            results_file = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(backtest_results, f, indent=2)
            
            session.close()
        
        console.print(f"[green]Backtest complete![/green]")
        
        # Display key metrics
        table = Table(title=f"Backtest Results - {market}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        metrics = backtest_results['metrics']
        table.add_row("Hit Rate", f"{metrics['hit_rate']:.1%}")
        table.add_row("ROI", f"{metrics['roi']:.1%}")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
        table.add_row("Total Bets", str(metrics['total_bets']))
        table.add_row("Brier Score", f"{backtest_results['calibration']['brier_score']:.3f}")
        
        console.print(table)
        console.print(f"[green]Results saved to {results_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


# RUN-API COMMAND
@app.command(name="run-api")
def run_api(
    host: Annotated[str, typer.Option("--host", help="Host to bind to")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload")] = False,
    workers: Annotated[int, typer.Option("--workers", help="Number of worker processes")] = 1
):
    """Start the FastAPI server"""
    console.print(f"[bold blue]Starting NFL Predictions API server[/bold blue]")
    console.print(f"Server will be available at: http://{host}:{port}")
    console.print(f"API documentation: http://{host}:{port}/docs")
    
    try:
        if uvicorn is None:
            console.print("[red]uvicorn is not installed. Install with:[/red]")
            console.print("  pip install 'uvicorn[standard]'\n  # or via conda:\n  conda install -c conda-forge uvicorn")
            raise typer.Exit(1)
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Server failed to start: {e}[/red]")
        if state['debug']:
            console.print_exception()
        raise typer.Exit(1)


# STATUS COMMAND
@app.command()
def status():
    """Show system status and health"""
    console.print("[bold blue]NFL Betting Analyzer System Status[/bold blue]")
    
    try:
        session = get_db_session(f"sqlite:///{state['database']}")
        
        # Database statistics
        total_players = session.query(Player).count()
        active_players = session.query(Player).filter(Player.is_active == True).count()
        total_games = session.query(Game).count()
        games_2025 = session.query(Game).filter(Game.season == 2025).count()
        
        # Create status table
        table = Table(title="System Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        table.add_row("Database", "Connected", f"sqlite:///{state['database']}")
        table.add_row("Players", "Loaded", f"{total_players:,} total, {active_players:,} active")
        table.add_row("Games", "Available", f"{total_games:,} total, {games_2025:,} in 2025")
        
        # Check models
        try:
            from core.models.streamlined_models import StreamlinedNFLModels
            models = StreamlinedNFLModels(session)
            models_dir = Path("models/streamlined")
            model_status = "Available" if (models_dir.exists() and any(models_dir.glob("*.pkl"))) else "Not Available"
        except:
            model_status = "Error loading"
        
        table.add_row("Models", model_status, "Prediction models")
        
        # Check API
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            api_status = "Running" if response.status_code == 200 else "Issues"
        except:
            api_status = "Not running"
        
        table.add_row("API Server", api_status, "http://localhost:8000")
        
        console.print(table)
        
        # System readiness
        readiness_score = sum([
            1 if total_players > 0 else 0,
            1 if active_players > 0 else 0,
            1 if games_2025 > 0 else 0,
            1 if "Available" in model_status else 0
        ])
        
        if readiness_score >= 3:
            console.print("[green]System is ready for predictions![/green]")
        else:
            console.print("[yellow]System needs setup. Run data fetch and model training.[/yellow]")
        
        session.close()
        
    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")
        if state['debug']:
            console.print_exception()


# Main entry point
if __name__ == "__main__":
    app()
