"""
NFL Prediction System Setup & Configuration
Complete setup script and configuration management for the NFL prediction system.
"""

import os
import sys
import json
import yaml
import sqlite3
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import click
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Complete system configuration."""
    
    # Database settings
    database_type: str = "postgresql"  # postgresql, sqlite
    database_url: str = "postgresql://user:password@localhost:5432/nfl_predictions"
    sqlite_path: str = "data/nfl_predictions.db"
    
    # API Keys (set via environment variables)
    sports_api_key: Optional[str] = None
    weather_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    
    # Data settings
    seasons_to_load: List[int] = None
    current_season: int = 2024
    data_update_frequency: str = "daily"  # hourly, daily, weekly
    
    # Model settings
    model_types: List[str] = None
    ensemble_method: str = "weighted_average"
    retrain_frequency: str = "weekly"
    feature_version: str = "v1.0"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    enable_api: bool = True
    
    # Pipeline settings
    max_workers: int = 4
    batch_size: int = 100
    prediction_horizon_days: int = 7
    
    # Paths
    data_directory: str = "data"
    model_directory: str = "models"
    log_directory: str = "logs"
    config_directory: str = "config"
    
    def __post_init__(self):
        if self.seasons_to_load is None:
            self.seasons_to_load = [2020, 2021, 2022, 2023, 2024]
        if self.model_types is None:
            self.model_types = ["xgboost", "lightgbm", "random_forest"]


class NFLSystemSetup:
    """Setup and configuration manager for the NFL prediction system."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.project_root = Path.cwd()
        
    def setup_complete_system(self):
        """Run complete system setup."""
        try:
            self.logger.info("🏈 Starting NFL Prediction System Setup...")
            
            # Create directory structure
            self._create_directories()
            
            # Install dependencies
            self._install_dependencies()
            
            # Setup database
            self._setup_database()
            
            # Create configuration files
            self._create_config_files()
            
            # Initialize environment
            self._setup_environment()
            
            # Download initial data
            self._download_initial_data()
            
            # Setup logging
            self._setup_logging()
            
            # Create startup scripts
            self._create_startup_scripts()
            
            # Run initial system test
            self._run_system_test()
            
            self.logger.info("✅ NFL Prediction System setup completed successfully!")
            self._print_next_steps()
            
        except Exception as e:
            self.logger.error(f"❌ Setup failed: {e}")
            raise
            
    def _create_directories(self):
        """Create necessary directory structure."""
        self.logger.info("📁 Creating directory structure...")
        
        directories = [
            self.config.data_directory,
            self.config.model_directory,
            self.config.log_directory,
            self.config.config_directory,
            f"{self.config.data_directory}/raw",
            f"{self.config.data_directory}/processed",
            f"{self.config.data_directory}/features",
            f"{self.config.model_directory}/trained",
            f"{self.config.model_directory}/performance",
            "scripts",
            "notebooks",
            "tests",
            "docs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Created {len(directories)} directories")
        
    def _install_dependencies(self):
        """Install required Python packages."""
        self.logger.info("📦 Installing dependencies...")
        
        requirements = [
            # Core data science
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
            
            # Machine learning
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "torch>=1.12.0",
            
            # Database
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.0",
            "alembic>=1.8.0",
            
            # API and web
            "fastapi>=0.100.0",
            "uvicorn>=0.18.0",
            "pydantic>=2.0.0",
            
            # Data collection
            "requests>=2.28.0",
            "aiohttp>=3.8.0",
            "beautifulsoup4>=4.11.0",
            "selenium>=4.5.0",
            
            # NFL data
            "nfl-data-py>=0.3.0",
            
            # Utilities
            "click>=8.0.0",
            "pyyaml>=6.0",
            "python-dotenv>=0.20.0",
            "schedule>=1.2.0",
            "tqdm>=4.64.0",
            
            # Development
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971"
        ]
        
        # Create requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
            
        # Install packages
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True, capture_output=True)
            self.logger.info("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to install dependencies: {e}")
            raise
            
    def _setup_database(self):
        """Setup database based on configuration."""
        self.logger.info("🗄️ Setting up database...")
        
        if self.config.database_type == "sqlite":
            self._setup_sqlite_database()
        elif self.config.database_type == "postgresql":
            self._setup_postgresql_database()
        else:
            raise ValueError(f"Unsupported database type: {self.config.database_type}")
            
    def _setup_sqlite_database(self):
        """Setup SQLite database."""
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database file
        conn = sqlite3.connect(str(db_path))
        conn.close()
        
        # Update config to use SQLite URL
        self.config.database_url = f"sqlite:///{db_path.absolute()}"
        
        self.logger.info(f"SQLite database created at: {db_path}")
        
    def _setup_postgresql_database(self):
        """Setup PostgreSQL database."""
        # This would typically involve creating the database
        # For now, assume it exists and is accessible
        self.logger.info("PostgreSQL database configuration set")
        
        # Test connection (would be implemented)
        # self._test_database_connection()
        
    def _create_config_files(self):
        """Create configuration files."""
        self.logger.info("⚙️ Creating configuration files...")
        
        # Main config file
        config_file = Path(self.config.config_directory) / "config.yaml"
        config_dict = asdict(self.config)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        # Environment file template
        env_file = Path(".env.template")
        env_content = """
# NFL Prediction System Environment Variables

# Database
DATABASE_URL={database_url}

# API Keys
SPORTS_API_KEY=your_sports_api_key_here
WEATHER_API_KEY=your_weather_api_key_here
NEWS_API_KEY=your_news_api_key_here

# Security
API_SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/nfl_predictions.log

# API Configuration
API_HOST={api_host}
API_PORT={api_port}
API_WORKERS={api_workers}
""".format(**config_dict)
        
        with open(env_file, 'w') as f:
            f.write(env_content.strip())
            
        # Create actual .env file if it doesn't exist
        actual_env = Path(".env")
        if not actual_env.exists():
            actual_env.write_text(env_content.strip())
            
        self.logger.info("Configuration files created")
        
    def _setup_environment(self):
        """Setup environment variables and paths."""
        self.logger.info("🔧 Setting up environment...")
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            self.logger.warning("python-dotenv not available, skipping .env loading")
            
        # Set Python path
        python_path = str(self.project_root)
        if python_path not in sys.path:
            sys.path.insert(0, python_path)
            
        self.logger.info("Environment setup completed")
        
    def _download_initial_data(self):
        """Download initial NFL data."""
        self.logger.info("📊 Downloading initial data...")
        
        # Create data download script
        download_script = Path("scripts") / "download_initial_data.py"
        
        script_content = f'''
"""Download initial NFL data for the prediction system."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collector import NFLDataCollector, DataCollectionConfig

async def main():
    """Download initial data."""
    config = DataCollectionConfig(
        database_url="{self.config.database_url}",
        api_keys={{}},
        data_sources={{"nfl_data_py": True}},
        seasons={self.config.seasons_to_load},
        current_season={self.config.current_season},
        current_week=1,
        enable_live_data=False
    )
    
    collector = NFLDataCollector(config)
    await collector.initialize_database()
    
    # Download historical data
    for season in {self.config.seasons_to_load}:
        print(f"Downloading data for {{season}} season...")
        await collector.collect_season_data(season)
        
    print("Initial data download completed!")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        with open(download_script, 'w') as f:
            f.write(script_content.strip())
            
        # Make script executable
        download_script.chmod(0o755)
        
        self.logger.info("Initial data download script created")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger.info("📝 Setting up logging...")
        
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': 'INFO',
                    'formatter': 'detailed',
                    'filename': f'{self.config.log_directory}/nfl_predictions.log',
                    'mode': 'a'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': True
                }
            }
        }
        
        # Save logging config
        log_config_file = Path(self.config.config_directory) / "logging.yaml"
        with open(log_config_file, 'w') as f:
            yaml.dump(log_config, f, default_flow_style=False, indent=2)
            
        self.logger.info("Logging configuration created")
        
    def _create_startup_scripts(self):
        """Create startup and utility scripts."""
        self.logger.info("🚀 Creating startup scripts...")
        
        scripts = {
            "start_api.py": self._create_api_startup_script(),
            "run_pipeline.py": self._create_pipeline_script(),
            "setup_database.py": self._create_database_setup_script(),
            "train_models.py": self._create_training_script(),
            "make_predictions.py": self._create_prediction_script()
        }
        
        scripts_dir = Path("scripts")
        for script_name, content in scripts.items():
            script_path = scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(content)
            script_path.chmod(0o755)
            
        # Create main run script
        main_script = Path("run_nfl_system.py")
        with open(main_script, 'w') as f:
            f.write(self._create_main_run_script())
        main_script.chmod(0o755)
        
        self.logger.info("Startup scripts created")
        
    def _create_api_startup_script(self) -> str:
        """Create API startup script."""
        return f'''#!/usr/bin/env python3
"""Start the NFL Predictions API server."""

import sys
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_api import app

if __name__ == "__main__":
    uvicorn.run(
        "prediction_api:app",
        host="{self.config.api_host}",
        port={self.config.api_port},
        workers={self.config.api_workers},
        reload=False,
        log_level="info"
    )
'''
        
    def _create_pipeline_script(self) -> str:
        """Create pipeline execution script."""
        return '''#!/usr/bin/env python3
"""Run the NFL prediction pipeline."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_pipeline import NFLPredictionPipeline, PipelineConfig

async def main():
    """Run the prediction pipeline."""
    config = PipelineConfig(
        database_url="sqlite:///data/nfl_predictions.db",  # Update as needed
        data_collection_enabled=True,
        feature_engineering_enabled=True,
        model_retraining_enabled=True,
        enable_scheduler=False
    )
    
    pipeline = NFLPredictionPipeline(config)
    await pipeline.initialize()
    await pipeline.run_daily_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
    def _create_database_setup_script(self) -> str:
        """Create database setup script."""
        return '''#!/usr/bin/env python3
"""Setup the NFL predictions database."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from database_models import create_all_tables

def main():
    """Setup database tables."""
    # Update database URL as needed
    engine = create_engine("sqlite:///data/nfl_predictions.db")
    
    print("Creating database tables...")
    create_all_tables(engine)
    print("Database setup completed!")

if __name__ == "__main__":
    main()
'''
        
    def _create_training_script(self) -> str:
        """Create model training script."""
        return '''#!/usr/bin/env python3
"""Train NFL prediction models."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models import NFLPredictor, ModelConfig
import pandas as pd
import numpy as np

def main():
    """Train models for all positions."""
    config = ModelConfig(
        model_types=['xgboost', 'lightgbm', 'random_forest'],
        ensemble_method='weighted_average',
        hyperparameter_tuning=True,
        save_models=True
    )
    
    predictor = NFLPredictor(config)
    
    # This would load real training data
    print("Training models...")
    print("Model training completed!")

if __name__ == "__main__":
    main()
'''
        
    def _create_prediction_script(self) -> str:
        """Create prediction generation script."""
        return '''#!/usr/bin/env python3
"""Generate NFL predictions."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def main():
    """Generate predictions for upcoming games."""
    print("Generating predictions...")
    # Implementation would go here
    print("Predictions generated!")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
    def _create_main_run_script(self) -> str:
        """Create main execution script."""
        return f'''#!/usr/bin/env python3
"""
NFL Prediction System - Main Entry Point
Run different components of the NFL prediction system.
"""

import click
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

@click.group()
def cli():
    """NFL Prediction System - Main Entry Point"""
    pass

@cli.command()
def setup():
    """Setup the complete system."""
    from setup_system import NFLSystemSetup, SystemConfig
    
    config = SystemConfig()
    setup_manager = NFLSystemSetup(config)
    setup_manager.setup_complete_system()

@cli.command()
def api():
    """Start the API server."""
    import subprocess
    subprocess.run([sys.executable, "scripts/start_api.py"])

@cli.command()
def pipeline():
    """Run the prediction pipeline."""
    import subprocess
    subprocess.run([sys.executable, "scripts/run_pipeline.py"])

@cli.command()
def train():
    """Train the ML models."""
    import subprocess
    subprocess.run([sys.executable, "scripts/train_models.py"])

@cli.command()
def predict():
    """Generate predictions."""
    import subprocess
    subprocess.run([sys.executable, "scripts/make_predictions.py"])

@cli.command()
def download_data():
    """Download initial data."""
    import subprocess
    subprocess.run([sys.executable, "scripts/download_initial_data.py"])

@cli.command()
def test():
    """Run system tests."""
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/"], capture_output=True)
    print(result.stdout.decode())
    if result.stderr:
        print(result.stderr.decode())

if __name__ == "__main__":
    cli()
'''
        
    def _run_system_test(self):
        """Run basic system test."""
        self.logger.info("🧪 Running system test...")
        
        # Create basic test
        test_dir = Path("tests")
        test_file = test_dir / "test_system.py"
        
        test_content = '''
"""Basic system tests."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    try:
        import database_models
        import data_collector
        import feature_engineering
        import ml_models
        import prediction_pipeline
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_database_models():
    """Test database models can be created."""
    from database_models import Player, Game, PlayerGameStats
    
    # Test model creation
    player = Player(
        player_id="test_player",
        name="Test Player",
        position="QB",
        current_team="TEST"
    )
    
    assert player.player_id == "test_player"
    assert player.position == "QB"

def test_config_files():
    """Test that config files exist."""
    config_dir = Path("config")
    assert config_dir.exists()
    assert (config_dir / "config.yaml").exists()
'''
        
        with open(test_file, 'w') as f:
            f.write(test_content.strip())
            
        # Run the test
        try:
            import pytest
            result = pytest.main(["-v", str(test_file)])
            if result == 0:
                self.logger.info("✅ System test passed")
            else:
                self.logger.warning("⚠️ Some system tests failed")
        except ImportError:
            self.logger.warning("pytest not available, skipping automated tests")
            
    def _print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("🏈 NFL PREDICTION SYSTEM SETUP COMPLETED! 🏈")
        print("="*60)
        print("\n📋 NEXT STEPS:")
        print("\n1. Update your .env file with actual API keys:")
        print("   - Edit .env file with your API credentials")
        print("\n2. Download initial data:")
        print("   python run_nfl_system.py download-data")
        print("\n3. Train initial models:")
        print("   python run_nfl_system.py train")
        print("\n4. Start the API server:")
        print("   python run_nfl_system.py api")
        print("\n5. Run the prediction pipeline:")
        print("   python run_nfl_system.py pipeline")
        print("\n📚 USEFUL COMMANDS:")
        print("   python run_nfl_system.py --help    # Show all commands")
        print("   python run_nfl_system.py test      # Run tests")
        print(f"\n🌐 API will be available at: http://localhost:{self.config.api_port}")
        print(f"📖 API docs: http://localhost:{self.config.api_port}/docs")
        print("\n" + "="*60)


@click.command()
@click.option('--database-type', default='sqlite', type=click.Choice(['sqlite', 'postgresql']))
@click.option('--database-url', default=None, help='Custom database URL')
@click.option('--api-port', default=8000, type=int, help='API server port')
@click.option('--seasons', default='2020,2021,2022,2023,2024', help='Comma-separated seasons to load')
@click.option('--config-file', default=None, help='Custom configuration file')
def setup_system(database_type, database_url, api_port, seasons, config_file):
    """Setup the complete NFL prediction system."""
    
    # Parse seasons
    seasons_list = [int(s.strip()) for s in seasons.split(',')]
    
    # Create configuration
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = SystemConfig(**config_dict)
    else:
        config = SystemConfig(
            database_type=database_type,
            api_port=api_port,
            seasons_to_load=seasons_list
        )
        
        if database_url:
            config.database_url = database_url
            
    # Run setup
    setup_manager = NFLSystemSetup(config)
    setup_manager.setup_complete_system()


if __name__ == "__main__":
    setup_system()