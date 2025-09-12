import pytest

from pathlib import Path
import pandas as pd
from generate_coverage_matrices import (
    generate_stats_feature_matrix,
    generate_model_market_matrix,
)


def test_stats_feature_matrix_includes_key_stats_and_features():
    df = generate_stats_feature_matrix()

    # Required columns
    assert "stat" in df.columns

    required_features = [
        "rolling_averages_3game",
        "rolling_averages_5game",
        "season_to_date",
        "opponent_adjusted",
        "situational_splits",
        "red_zone_efficiency",
        "goal_line_efficiency",
        "air_yard_distribution",
        "yac_efficiency",
        "pressure_splits",
        "coverage_splits",
        "rest_days_adjustment",
        "travel_distance_adjustment",
        "weather_adjustment",
        "home_away_splits",
        "division_opponent_adjustment",
        "pace_adjustment",
    ]
    for col in required_features:
        assert col in df.columns, f"Missing feature column: {col}"

    # Required stats present
    required_stats = [
        "passing_yards",
        "passing_touchdowns",
        "interceptions",
        "passing_completions",
        "rushing_yards",
        "rushing_touchdowns",
        "receiving_yards",
        "receiving_touchdowns",
        "receptions",
        "rushing_attempts",
        "longest_pass",
        "longest_rush",
        "longest_reception",
        "tackles",
        "sacks",
        "targets",
        "offensive_snaps",
        "snap_percentage",
    ]
    present_stats = set(df["stat"].tolist())
    for stat in required_stats:
        assert stat in present_stats, f"Missing required stat: {stat}"


def test_model_market_matrix_includes_key_markets_and_models():
    df = generate_model_market_matrix()

    # Required columns
    assert "market" in df.columns

    required_models = [
        "XGBoost",
        "LightGBM",
        "RandomForest",
        "LinearRegression",
        "NeuralNetwork",
        "EnsembleModel",
    ]
    for col in required_models:
        assert col in df.columns, f"Missing model column: {col}"

    # Required markets present
    required_markets = [
        "player_passing_yds",
        "player_passing_tds",
        "player_interceptions",
        "player_completions",
        "player_rushing_yds",
        "player_rushing_tds",
        "player_rushing_attempts",
        "player_rec_yds",
        "player_receptions",
        "player_rec_tds",
        "player_fantasy_points",
        "player_longest_pass",
        "player_longest_rush",
        "player_longest_reception",
        "player_tackles",
        "player_sacks",
        "kicker_field_goals_made",
        "kicker_field_goals_attempted",
        "kicker_longest_field_goal",
        "team_total_points",
    ]
    present_markets = set(df["market"].tolist())
    for m in required_markets:
        assert m in present_markets, f"Missing required market: {m}"


def test_generated_csvs_parse_cleanly_and_include_expected_columns(tmp_path):
    # Generate matrices and save them to reports/coverage
    reports_dir = Path("reports/coverage")
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats_df = generate_stats_feature_matrix()
    model_df = generate_model_market_matrix()

    stats_csv = reports_dir / "stats_feature_matrix.csv"
    model_csv = reports_dir / "model_market_matrix.csv"

    stats_df.to_csv(stats_csv, index=False)
    model_df.to_csv(model_csv, index=False)

    # Read back the CSVs to ensure clean parsing and one-line headers
    stats_read = pd.read_csv(stats_csv)
    model_read = pd.read_csv(model_csv)

    # Ensure first row is data, not header duplication
    assert "stat" in stats_read.columns
    assert "market" in model_read.columns

    # Spot-check some expected stats/markets exist in the files
    stats_present = set(stats_read["stat"].tolist())
    markets_present = set(model_read["market"].tolist())

    assert "passing_completions" in stats_present
    assert "rushing_attempts" in stats_present
    assert "longest_reception" in stats_present
    assert "tackles" in stats_present
    assert "sacks" in stats_present

    assert "player_completions" in markets_present
    assert "player_rushing_attempts" in markets_present
    assert "player_longest_pass" in markets_present
    assert "player_tackles" in markets_present
    assert "kicker_field_goals_attempted" in markets_present
