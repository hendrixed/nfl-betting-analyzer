import pytest

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
        "rushing_yards",
        "rushing_touchdowns",
        "receiving_yards",
        "receiving_touchdowns",
        "receptions",
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
        "player_rushing_yds",
        "player_rushing_tds",
        "player_rec_yds",
        "player_receptions",
        "player_fantasy_points",
    ]
    present_markets = set(df["market"].tolist())
    for m in required_markets:
        assert m in present_markets, f"Missing required market: {m}"
