#!/usr/bin/env python3
"""
Comprehensive schema tests for NFL data categories A-H
Tests all required columns, data types, and constraints for each data category.
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def latest_snapshot_dir() -> Path:
    """Get the latest snapshot directory"""
    snapshots_dir = Path("data/snapshots")
    if not snapshots_dir.exists():
        pytest.skip("No snapshots directory found")
    
    # Get latest date directory
    date_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir() and d.name.count('-') == 2]
    if not date_dirs:
        pytest.skip("No date directories found in snapshots")
    
    return max(date_dirs)

def read_csv_safe(file_path: Path) -> pd.DataFrame:
    """Read CSV file safely with error handling"""
    if not file_path.exists():
        pytest.skip(f"File not found: {file_path}")
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(f"Failed to read {file_path}: {e}")

def validate_columns(df: pd.DataFrame, required_cols: List[str], file_name: str):
    """Validate that all required columns are present"""
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        pytest.fail(f"{file_name}: Missing required columns: {missing_cols}")

def validate_no_nulls(df: pd.DataFrame, non_null_cols: List[str], file_name: str):
    """Validate that specified columns have no null values"""
    for col in non_null_cols:
        if col in df.columns and df[col].isnull().any():
            pytest.fail(f"{file_name}: Column '{col}' contains null values")

# A. REFERENCE DATA TESTS

def test_reference_teams_schema():
    """Test reference teams schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "reference_teams.csv")
    
    required_cols = ['team_id', 'abbr', 'name', 'conference', 'division', 'head_coach', 'home_stadium_id']
    validate_columns(df, required_cols, "reference_teams.csv")
    validate_no_nulls(df, required_cols, "reference_teams.csv")
    
    # Validate data constraints
    assert df['conference'].isin(['AFC', 'NFC']).all(), "Invalid conference values"
    assert df['division'].isin(['North', 'South', 'East', 'West']).all(), "Invalid division values"
    assert len(df) == 32, f"Expected 32 teams, got {len(df)}"
    assert df['team_id'].is_unique, "Team IDs must be unique"

def test_reference_stadiums_schema():
    """Test reference stadiums schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "reference_stadiums.csv")
    
    required_cols = ['stadium_id', 'name', 'city', 'state', 'lat', 'lon', 'surface', 'roof', 'elevation']
    validate_columns(df, required_cols, "reference_stadiums.csv")
    validate_no_nulls(df, required_cols, "reference_stadiums.csv")
    
    # Validate data constraints
    assert df['lat'].between(-90, 90).all(), "Invalid latitude values"
    assert df['lon'].between(-180, 180).all(), "Invalid longitude values"
    assert df['roof'].isin(['open', 'dome', 'retractable']).all(), "Invalid roof types"
    assert df['stadium_id'].is_unique, "Stadium IDs must be unique"

def test_reference_players_schema():
    """Test reference players schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "reference_players.csv")
    
    required_cols = ['player_id', 'name', 'position', 'team', 'jersey_number', 'height', 'weight', 'college']
    validate_columns(df, required_cols, "reference_players.csv")
    validate_no_nulls(df, required_cols, "reference_players.csv")
    
    # Validate data constraints
    valid_positions = ['QB', 'RB', 'FB', 'WR', 'TE', 'OL', 'C', 'G', 'T', 'DL', 'DE', 'DT', 'NT', 'LB', 'ILB', 'OLB', 'DB', 'CB', 'S', 'FS', 'SS', 'K', 'P', 'LS']
    assert df['position'].isin(valid_positions).all(), "Invalid position values"
    assert df['jersey_number'].between(0, 99).all(), "Invalid jersey numbers"
    assert df['player_id'].is_unique, "Player IDs must be unique"

# B. ROSTER/DEPTH/STATUS TESTS

def test_rosters_schema():
    """Test weekly rosters schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "rosters.csv")
    
    required_cols = ['player_id', 'name', 'position', 'team', 'jersey_number', 'status', 'week', 'season']
    validate_columns(df, required_cols, "rosters.csv")
    validate_no_nulls(df, required_cols, "rosters.csv")
    
    # Validate data constraints
    assert df['status'].isin(['active', 'inactive', 'injured_reserve', 'pup', 'suspended']).all(), "Invalid status values"
    assert df['week'].between(1, 18).all(), "Invalid week values"
    assert df['season'].between(2020, 2030).all(), "Invalid season values"

def test_depth_charts_schema():
    """Test depth charts schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "depth_charts.csv")
    
    required_cols = ['team', 'position', 'player_id', 'depth_rank', 'role', 'week', 'season']
    validate_columns(df, required_cols, "depth_charts.csv")
    validate_no_nulls(df, required_cols, "depth_charts.csv")
    
    # Validate data constraints
    assert df['depth_rank'].between(1, 10).all(), "Invalid depth rank values"
    assert df['role'].isin(['starter', 'backup', 'special_teams', 'inactive']).all(), "Invalid role values"

def test_injuries_schema():
    """Test injuries schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "injuries.csv")
    
    required_cols = ['player_id', 'name', 'team', 'position', 'injury_status', 'injury_description', 'week', 'season', 'report_date']
    validate_columns(df, required_cols, "injuries.csv")
    validate_no_nulls(df, required_cols, "injuries.csv")
    
    # Validate data constraints
    valid_statuses = ['out', 'doubtful', 'questionable', 'probable', 'limited', 'full', 'dnp']
    assert df['injury_status'].isin(valid_statuses).all(), "Invalid injury status values"

# C. SCHEDULES & GAMES TESTS

def test_schedule_schema():
    """Test game schedule schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "schedule.csv")
    
    required_cols = ['game_id', 'season', 'week', 'season_type', 'game_date', 'home_team', 'away_team', 'stadium_id']
    validate_columns(df, required_cols, "schedule.csv")
    validate_no_nulls(df, required_cols, "schedule.csv")
    
    # Validate data constraints
    assert df['season_type'].isin(['PRE', 'REG', 'POST']).all(), "Invalid season type values"
    assert df['week'].between(1, 22).all(), "Invalid week values"
    assert df['game_id'].is_unique, "Game IDs must be unique"
    assert (df['home_team'] != df['away_team']).all(), "Home and away teams cannot be the same"

# D. USAGE/SNAP COUNTS TESTS

def test_snaps_schema():
    """Test snap counts schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "snaps.csv")
    
    required_cols = ['player_id', 'game_id', 'team', 'position', 'offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'snap_percentage']
    validate_columns(df, required_cols, "snaps.csv")
    validate_no_nulls(df, required_cols, "snaps.csv")
    
    # Validate data constraints
    assert df['offensive_snaps'].ge(0).all(), "Offensive snaps cannot be negative"
    assert df['defensive_snaps'].ge(0).all(), "Defensive snaps cannot be negative"
    assert df['special_teams_snaps'].ge(0).all(), "Special teams snaps cannot be negative"
    assert df['snap_percentage'].between(0, 100).all(), "Snap percentage must be between 0-100"

# E. PLAY-BY-PLAY TESTS

def test_pbp_schema():
    """Test play-by-play schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "pbp.csv")
    
    required_cols = ['game_id', 'play_id', 'quarter', 'down', 'yards_to_go', 'yard_line', 'offense', 'defense', 'play_type']
    validate_columns(df, required_cols, "pbp.csv")
    validate_no_nulls(df, required_cols, "pbp.csv")
    
    # Validate data constraints
    assert df['quarter'].between(1, 5).all(), "Invalid quarter values"
    assert df['down'].between(1, 4).all(), "Invalid down values"
    assert df['yards_to_go'].between(1, 99).all(), "Invalid yards to go values"
    valid_play_types = ['pass', 'rush', 'punt', 'field_goal', 'extra_point', 'kickoff', 'penalty']
    assert df['play_type'].isin(valid_play_types).all(), "Invalid play type values"

# F. WEATHER TESTS

def test_weather_schema():
    """Test weather data schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "weather.csv")
    
    required_cols = ['game_id', 'temperature', 'humidity', 'wind_speed', 'conditions', 'roof_state']
    validate_columns(df, required_cols, "weather.csv")
    validate_no_nulls(df, required_cols, "weather.csv")
    
    # Validate data constraints
    assert df['temperature'].between(-20, 120).all(), "Invalid temperature values"
    assert df['humidity'].between(0, 100).all(), "Invalid humidity values"
    assert df['wind_speed'].ge(0).all(), "Wind speed cannot be negative"
    assert df['roof_state'].isin(['open', 'closed', 'retractable_open', 'retractable_closed']).all(), "Invalid roof state values"

# G. ODDS & PROPS TESTS

def test_odds_schema():
    """Test odds and props schema and data quality"""
    snapshot_dir = latest_snapshot_dir()
    df = read_csv_safe(snapshot_dir / "odds.csv")
    
    required_cols = ['book', 'market', 'line', 'over_odds', 'under_odds', 'timestamp']
    validate_columns(df, required_cols, "odds.csv")
    validate_no_nulls(df, required_cols, "odds.csv")
    
    # Validate canonical market names
    valid_markets = [
        'player_pass_att', 'player_pass_cmp', 'player_pass_yds', 'player_pass_tds', 'player_ints', 'player_rush_yds', 'player_pass_long',
        'player_rush_att', 'player_rush_tds', 'player_rec', 'player_rec_yds', 'player_rec_tds', 'player_rush_rec_yds',
        'player_tgts', 'player_long_rec', 'player_fg_made', 'player_fg_made_50+', 'player_xp_made',
        'team_total', 'team_total_1H', 'team_total_2H'
    ]
    unknown_markets = set(df['market'].unique()) - set(valid_markets)
    assert not unknown_markets, f"Unknown market types found: {unknown_markets}"
    
    # Validate canonical book names
    valid_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet', 'barstool', 'unibet', 'bet365', 'espnbet']
    unknown_books = set(df['book'].unique()) - set(valid_books)
    assert not unknown_books, f"Unknown book names found: {unknown_books}"

if __name__ == "__main__":
    pytest.main([__file__])
