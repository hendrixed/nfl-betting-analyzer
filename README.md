# NFL Prediction System

A comprehensive NFL betting analysis system with machine learning predictions, real-time data integration, and advanced analytics.

## Features

- **Unified CLI & API**: Single command-line interface (`nfl_cli.py`) and FastAPI server (`api/app.py`)
- **Machine Learning Models**: Advanced prediction models for player performance and game outcomes
- **Comprehensive Data Coverage**: Categories A-H with schema validation and snapshot caching
- **Real-time Analytics**: Market analysis and betting opportunity identification
- **Monte Carlo Simulation**: Advanced game simulation with correlation modeling
- **Backtesting Framework**: Model validation with calibration plots and performance metrics

## Quick Start (Linux/Conda)

1. **Environment Setup**:
```bash
conda activate nfl
pip install -r requirements.txt
```

2. **Initialize System**:
```bash
python nfl_cli.py foundation  # Ingest foundational data
python nfl_cli.py fetch       # Fetch current season data
# WEATHER_API_KEY=your_weather_api_key
```

### 3. System Setup

```bash
# Initialize the system (foundation data)
python nfl_cli.py foundation --season 2025

# Download initial data
python nfl_cli.py fetch --season 2025 --week 1
```

### 4. Run Demo

```bash
# Run comprehensive demo
python demo.py
```

## Usage

### Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
# Show all available commands
python nfl_cli.py --help

# Data ingestion (populates data/snapshots/YYYY-MM-DD/*.csv)
python nfl_cli.py fetch --season 2025 --week 1

# Ingest foundational season data (schedules, rosters)
python nfl_cli.py foundation --season 2025

# Ingest schedules.csv from a snapshot directory into the DB
python nfl_cli.py schedule-ingest --date 2024-01-01

# Generate mock odds snapshot (for /betting/props)
python nfl_cli.py odds-snapshot --max-offers 200

# Verify latest snapshot files & headers
python nfl_cli.py snapshot-verify

# Sync roster into DB from latest snapshot (updates team/active flags)
python nfl_cli.py sync-roster --deactivate-missing False

# Show saved models and performance summary
python nfl_cli.py models-status

# Run a real backtest (uses odds lines when available)
python nfl_cli.py backtest --target receiving_yards --market player_receiving_yards --limit 300

# Legacy synthetic backtest (for demos; prefer 'backtest')
python nfl_cli.py backtest-legacy --market player_rec_yds --weeks 8

# Start the unified FastAPI server
python nfl_cli.py run-api --host 0.0.0.0 --port 8000

# Train models (optional)
python nfl_cli.py train --target fantasy_points_ppr

# System status
python nfl_cli.py status
```

### API Usage

Start the API server:

```bash
python nfl_cli.py run-api
```

Example API calls:

```bash
# Health
curl "http://localhost:8000/health"

# Models summary (empty list until trained)
curl "http://localhost:8000/models"

# Props from latest snapshot with canonical filters
curl "http://localhost:8000/betting/props?book=DK&market=Passing%20Yards"

# Players (optionally filter)
curl "http://localhost:8000/players?team=KC&position=QB"

# Fantasy predictions (requires trained models)
curl "http://localhost:8000/predictions/players/fantasy?team=KC&position=QB"
```

### Browsing API and Web

The app includes a browsing layer to explore players, teams, games, and leaderboards. These endpoints return JSON suitable for programmatic use, and there are matching web pages for interactive browsing.

Web pages:

- Players: http://localhost:8000/web/players
- Leaderboards: http://localhost:8000/web/leaderboards
- Teams: http://localhost:8000/teams
- Team detail: http://localhost:8000/team/{TEAM_ID}
- Games: http://localhost:8000/games
- Odds: http://localhost:8000/web/odds
- Insights: http://localhost:8000/web/insights
- Backtests: http://localhost:8000/web/backtests

Players JSON API:

```bash
# Paginated players with search/filters/sorting
curl "http://localhost:8000/api/browse/players?q=mahomes&team=KC&position=QB&page=1&page_size=25&sort=name&order=asc"

# CSV export (current page)
curl -L -o players.csv "http://localhost:8000/api/browse/players/export.csv?q=mahomes&team=KC&position=QB&page=1&page_size=25&sort=name&order=asc"

# CSV export (all matches)
curl -L -o players_all.csv "http://localhost:8000/api/browse/players/export.csv?q=mahomes&team=KC&position=QB&all=true"
```

Leaderboards JSON API:

```bash
# Paginated leaderboards with filtering and sorting
curl "http://localhost:8000/api/browse/leaderboard?stat=fantasy_points_ppr&season=2024&position=WR&page=1&page_size=25&sort=value&order=desc"

# CSV export (current page)
curl -L -o leaderboard.csv "http://localhost:8000/api/browse/leaderboard/export.csv?stat=receiving_yards&season=2024&position=WR&page=1&page_size=25&sort=value&order=desc"

# CSV export (all matches)
curl -L -o leaderboard_all.csv "http://localhost:8000/api/browse/leaderboard/export.csv?stat=receiving_yards&season=2024&position=WR&all=true"
```

Player-centric JSON endpoints:

- Profile: `/api/browse/player/{player_id}/profile`
- Gamelog: `/api/browse/player/{player_id}/gamelog?season=YYYY`
- Career totals: `/api/browse/player/{player_id}/career`

Team-centric JSON endpoints:

- Team info + roster: `/api/browse/team/{team_id}`
- Depth chart: `/api/browse/team/{team_id}/depth-chart`
- Schedule: `/api/browse/team/{team_id}/schedule?season=YYYY`

Pagination and sorting:

- Players: `page`, `page_size` (1–200), `sort` in `name|position|team|rank`, `order` in `asc|desc`
- Leaderboards: `page`, `page_size` (1–200), `sort` in `value|name|team|position`, `order` in `asc|desc`

CSV export notes:

- Use the `/export.csv` variants to download current page by default.
- Append `&all=true` to export all matching rows (capped internally for safety).

Caching:

- Simple in-memory TTL caches are used for frequently requested browse endpoints:
  - Players: 60s TTL keyed by filters/pagination/sort params.
  - Leaderboards: 120s TTL keyed by filters/pagination/sort params.
- For production, consider replacing with Redis-backed cache.

### Python Integration

```python
import asyncio
from prediction_pipeline import NFLPredictionPipeline, PipelineConfig
from ml_models import NFLPredictor, ModelConfig

# Initialize pipeline
config = PipelineConfig(database_url="sqlite:///nfl.db")
pipeline = NFLPredictionPipeline(config)

# Run predictions
async def main():
    await pipeline.initialize()
    await pipeline.run_daily_pipeline()

asyncio.run(main())
```

## Architecture

### Core Components

1. **Database Models** (`database_models.py`)
   - SQLAlchemy ORM models for players, games, stats, predictions
   - Comprehensive relationships and indexing

2. **Data Collector** (`data_collector.py`)
   - Async data collection from nfl_data_py and external APIs
   - Automated data validation and cleaning

3. **Feature Engineering** (`feature_engineering.py`)
   - Advanced statistical features (800+ lines)
   - Rolling averages, opponent adjustments, situational features
   - Feature store for caching and versioning

4. **ML Models** (`ml_models.py`)
   - Ensemble modeling with multiple algorithms
   - Hyperparameter tuning and cross-validation
   - Model persistence and versioning

5. **Prediction Pipeline** (`prediction_pipeline.py`)
   - Orchestrates end-to-end prediction workflow
   - Concurrent processing and error handling
   - Performance monitoring and alerting

6. **Configuration Manager** (`config_manager.py`)
   - Centralized configuration with validation
   - Environment-specific settings
   - YAML-based configuration files

### Data Flow

```
Raw Data -> Data Collector -> Database -> Feature Engineering -> ML Models -> Predictions -> API
    v              v            v              v              v           v
External APIs   Validation   Storage    Feature Store   Model Store   Results
```

## Project Structure

```
nfl-betting-analyzer/
├── config/
│   ├── config.yaml          # Main configuration
│   └── logging.yaml         # Logging configuration
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data
│   ├── features/            # Feature store
│   └── nfl_predictions.db   # SQLite database
├── models/
│   ├── trained/             # Trained model files
│   └── performance/         # Model performance metrics
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── logs/                    # Log files
├── database_models.py       # Database ORM models
├── data_collector.py        # Data collection system
├── feature_engineering.py   # Feature engineering pipeline
├── ml_models.py            # Machine learning models
├── prediction_pipeline.py   # Main orchestration pipeline
├── prediction_api.py        # FastAPI REST API
├── config_manager.py        # Configuration management
├── run_nfl_system.py       # Main CLI entry point
├── demo.py                 # Comprehensive demo
└── requirements.txt        # Python dependencies
```

## Configuration

The system uses a hierarchical configuration system with the following sections:

### Database Configuration
```yaml
database:
  type: postgresql  # or sqlite
  url: postgresql://user:pass@localhost:5432/nfl_predictions
  pool_size: 10
```

### Data Collection
```yaml
data:
  seasons: [2022, 2023, 2024, 2025]
  current_season: 2025
  enable_live_data: true
  update_frequency: daily
```

### Feature Engineering
```yaml
features:
  version: v2.0
  lookback_windows: [3, 5, 10]
  enable_opponent_adjustments: true
  enable_weather_features: true
```

### Model Configuration
```yaml
models:
  model_types: [xgboost, lightgbm, random_forest]
  ensemble_method: weighted_average
  hyperparameter_tuning: true
  save_models: true
```

## Testing

Run the test suite:

```bash
# Full verification (prepares artifacts, runs tests, lint, mypy)
make verify

# Run all tests
python run_nfl_system.py test

# Run with verbose output
python run_nfl_system.py test --verbose

# Run specific test file
pytest tests/test_system.py -v
```

## Performance

The system is designed for production use with:

- **Async Operations**: Non-blocking data collection and processing
- **Concurrent Processing**: Parallel feature engineering and predictions
- **Caching**: Feature store and model caching for performance
- **Monitoring**: Built-in performance tracking and alerting
- **Scalability**: Configurable worker pools and batch processing

### Benchmarks

- **Data Collection**: ~1000 players/minute
- **Feature Engineering**: ~500 features/second
- **Model Training**: 2-5 minutes per position (depending on data size)
- **Prediction Generation**: ~100 predictions/second
- **API Response Time**: <100ms for single predictions

## Security

- Environment variable management for API keys
- SQL injection prevention with SQLAlchemy ORM
- Input validation with Pydantic models
- Rate limiting on API endpoints
- Secure database connections

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Documentation**: Full API documentation at `/docs`
- **CLI Help**: `python nfl_cli.py --help` for command reference

## Changelog

### v2.0.0 (Current)
- Complete codebase cleanup and consolidation
- Centralized configuration management
- Enhanced CLI with comprehensive commands
- Improved error handling and logging
- Production-ready architecture
- Comprehensive demo system

### v1.0.0
- Initial implementation with basic prediction capabilities
- Multiple demo files (now consolidated)
- Basic ML models and data collection

## Roadmap

- Real-time prediction updates
- Advanced ensemble techniques (stacking, blending)
- Web dashboard for predictions and monitoring
- Mobile app integration
- Advanced betting strategy optimization
- Integration with more data sources
- Kubernetes deployment configurations
- [ ] Real-time prediction updates
- [ ] Advanced ensemble techniques (stacking, blending)
- [ ] Web dashboard for predictions and monitoring
- [ ] Mobile app integration
- [ ] Advanced betting strategy optimization
- [ ] Integration with more data sources
- [ ] Kubernetes deployment configurations
