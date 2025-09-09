# NFL Prediction System

A comprehensive NFL player performance prediction system using machine learning, featuring data collection, feature engineering, ensemble modeling, and automated prediction pipelines.

## 🏈 Features

- **Advanced Data Collection**: Automated NFL data gathering from multiple sources
- **Feature Engineering**: 800+ lines of sophisticated feature extraction and processing
- **Ensemble ML Models**: XGBoost, LightGBM, Random Forest, Gradient Boosting, and Neural Networks
- **Prediction Pipeline**: Automated daily predictions with performance tracking
- **REST API**: FastAPI-based prediction serving
- **Configuration Management**: Centralized YAML-based configuration
- **Production Ready**: Async operations, logging, error handling, and monitoring

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd nfl-betting-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Conda environment (recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate nfl

# Or update an existing env named "nfl"
conda env update -n nfl -f environment.yml
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your API keys (optional for basic usage)
# DATABASE_URL=postgresql://user:password@localhost:5432/nfl_predictions
# ODDS_API_KEY=your_odds_api_key
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

## 📊 Usage

### Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
# Show all available commands
python nfl_cli.py --help

# Data ingestion (populates data/snapshots/YYYY-MM-DD/*.csv)
python nfl_cli.py fetch --season 2025 --week 1

# Ingest foundational season data (schedules, rosters)
python nfl_cli.py foundation --season 2025

# Generate mock odds snapshot (for /betting/props)
python nfl_cli.py odds-snapshot --max-offers 200

# Verify latest snapshot files & headers
python nfl_cli.py snapshot-verify

# Sync roster into DB from latest snapshot (updates team/active flags)
python nfl_cli.py sync-roster --deactivate-missing False

# Show saved models and performance summary
python nfl_cli.py models-status

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

## 🏗️ Architecture

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
Raw Data → Data Collector → Database → Feature Engineering → ML Models → Predictions → API
    ↓              ↓            ↓              ↓              ↓           ↓
External APIs   Validation   Storage    Feature Store   Model Store   Results
```

## 📁 Project Structure

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

## 🔧 Configuration

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
  seasons: [2022, 2023, 2024]
  current_season: 2024
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

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python run_nfl_system.py test

# Run with verbose output
python run_nfl_system.py test --verbose

# Run specific test file
pytest tests/test_system.py -v
```

## 📈 Performance

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

## 🔐 Security

- Environment variable management for API keys
- SQL injection prevention with SQLAlchemy ORM
- Input validation with Pydantic models
- Rate limiting on API endpoints
- Secure database connections

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

## 🤝 Contributing

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Documentation**: Full API documentation at `/docs`
- **CLI Help**: `python nfl_cli.py --help` for command reference

---

**Built for the 2025 NFL Season** 🏆

## 🆘 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs and feature requests via GitHub issues

## 🔄 Changelog

### v2.0.0 (Current)
- ✅ Complete codebase cleanup and consolidation
- ✅ Centralized configuration management
- ✅ Enhanced CLI with comprehensive commands
- ✅ Improved error handling and logging
- ✅ Production-ready architecture
- ✅ Comprehensive demo system

### v1.0.0
- Initial implementation with basic prediction capabilities
- Multiple demo files (now consolidated)
- Basic ML models and data collection

## 🎯 Roadmap

- [ ] Real-time prediction updates
- [ ] Advanced ensemble techniques (stacking, blending)
- [ ] Web dashboard for predictions and monitoring
- [ ] Mobile app integration
- [ ] Advanced betting strategy optimization
- [ ] Integration with more data sources
- [ ] Kubernetes deployment configurations
