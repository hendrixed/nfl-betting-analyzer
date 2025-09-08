"""
Configuration Management for NFL Prediction System
Centralized configuration handling with validation and environment variable support.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    type: str = "postgresql"
    url: Optional[str] = None
    path: str = "data/nfl_predictions.db"  # Added to match config.yaml
    sqlite_path: str = "data/nfl_predictions.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    def __post_init__(self):
        if self.url is None:
            if self.type == "sqlite":
                self.url = f"sqlite:///{self.sqlite_path}"
            else:
                self.url = os.getenv("DATABASE_URL", 
                    "postgresql://user:password@localhost:5432/nfl_predictions")


@dataclass
class APIConfig:
    """API configuration and credentials."""
    # Server configuration (from config.yaml)
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    enable: bool = True
    
    # API Keys
    odds_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ODDS_API_KEY"))
    weather_api_key: Optional[str] = field(default_factory=lambda: os.getenv("WEATHER_API_KEY"))
    news_api_key: Optional[str] = field(default_factory=lambda: os.getenv("NEWS_API_KEY"))
    sports_api_key: Optional[str] = field(default_factory=lambda: os.getenv("SPORTS_API_KEY"))
    
    # API URLs
    odds_api_url: str = "https://api.the-odds-api.com/v4"
    weather_api_url: str = "https://api.openweathermap.org/data/2.5"
    news_api_url: str = "https://newsapi.org/v2"
    espn_api_url: str = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    
    # Rate limiting
    max_requests_per_minute: int = 60
    request_timeout: int = 30


@dataclass
class DataConfig:
    """Data collection and processing configuration."""
    directory: str = "data"  # Added to match config.yaml
    update_frequency: str = "daily"  # Added to match config.yaml
    seasons: List[int] = field(default_factory=lambda: [2022, 2023, 2024])
    current_season: int = 2024
    current_week: int = 1
    seasons_to_load: List[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023, 2024])  # Added to match config.yaml
    
    # Data sources
    enable_nfl_data_py: bool = True
    enable_odds_api: bool = True
    enable_weather_api: bool = True
    enable_news_api: bool = False
    enable_live_data: bool = True
    
    # Update frequencies
    data_update_frequency: str = "daily"  # hourly, daily, weekly
    live_update_interval: int = 300  # seconds
    
    # Data validation
    min_games_for_analysis: int = 4
    max_missing_data_percentage: float = 0.2


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    version: str = "v1.0"  # Changed to match config.yaml
    batch_size: int = 100  # Added to match config.yaml
    lookback_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    rolling_windows: List[int] = field(default_factory=lambda: [4, 8])
    min_games_threshold: int = 3
    
    # Feature processing
    scale_features: bool = True
    handle_missing: str = "impute"  # impute, drop, zero
    feature_selection: bool = True
    max_features: int = 100
    
    # Advanced features
    enable_opponent_adjustments: bool = True
    enable_weather_features: bool = True
    enable_situational_features: bool = True
    enable_momentum_features: bool = True


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    directory: str = "models"  # Added to match config.yaml
    types: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "random_forest"
    ])  # Changed from model_types to types to match config.yaml
    model_types: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "random_forest", "gradient_boosting"
    ])
    ensemble_method: str = "weighted_average"  # simple_average, weighted_average, stacking
    retrain_frequency: str = "weekly"  # Added to match config.yaml
    
    # Training settings
    test_size: float = 0.2
    validation_size: float = 0.2
    cross_validation_folds: int = 5
    random_state: int = 42
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = True
    tuning_method: str = "bayesian"  # grid, random, bayesian
    max_tuning_iterations: int = 100
    
    # Model persistence
    save_models: bool = True
    model_directory: str = "models/trained"
    model_versioning: bool = True
    
    # Performance thresholds
    min_r2_score: float = 0.1
    min_accuracy_score: float = 0.6
    retrain_threshold: float = 0.05  # R² drop threshold


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    # Component toggles
    data_collection_enabled: bool = True
    feature_engineering_enabled: bool = True
    model_training_enabled: bool = True
    prediction_generation_enabled: bool = True
    model_retraining_enabled: bool = True
    
    # Scheduling
    enable_scheduler: bool = False
    schedule_time: str = "06:00"  # Daily run time
    timezone: str = "UTC"
    
    # Performance
    max_workers: int = 4
    batch_size: int = 100
    prediction_horizon_days: int = 7
    
    # Monitoring
    enable_performance_tracking: bool = True
    alert_on_performance_drop: bool = True
    performance_check_frequency: str = "daily"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/nfl_predictions.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Component-specific logging
    enable_sql_logging: bool = False
    enable_api_logging: bool = True
    enable_model_logging: bool = True


@dataclass
class PredictionConfig:
    """Prediction configuration."""
    horizon_days: int = 7

@dataclass
class DirectoriesConfig:
    """Directories configuration."""
    config: str = "config"
    logs: str = "logs"
    models: str = "models"
    data: str = "data"

@dataclass
class ProcessingConfig:
    """Processing configuration."""
    max_workers: int = 4

@dataclass
class APIKeysConfig:
    """API Keys configuration."""
    news_api_key: Optional[str] = None
    sports_api_key: Optional[str] = None
    weather_api_key: Optional[str] = None

@dataclass
class SystemMetaConfig:
    """System metadata configuration."""
    version: str = "2.0.0"
    environment: str = "development"

@dataclass
class SystemConfig:
    """Complete system configuration."""
    system: SystemMetaConfig = field(default_factory=SystemMetaConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    directories: DirectoriesConfig = field(default_factory=DirectoriesConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    
    # System metadata (backward compatibility)
    version: str = "2.0.0"
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")


class ConfigManager:
    """Configuration manager with validation and file I/O."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self.config: Optional[SystemConfig] = None
        
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
        """Load configuration from file or create default."""
        if config_path:
            self.config_path = Path(config_path)
            
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                    
                if not config_dict:
                    raise ValueError("Configuration file is empty or invalid")
                    
                # Convert nested dicts to dataclass instances
                config_dict = self._convert_dict_to_config(config_dict)
                self.config = SystemConfig(**config_dict)
                
                logger.info(f"✅ Configuration loaded from {self.config_path}")
                
            except yaml.YAMLError as e:
                logger.error(f"❌ YAML parsing error in {self.config_path}: {e}")
                logger.info("Using default configuration")
                self.config = SystemConfig()
            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"❌ File access error for {self.config_path}: {e}")
                logger.info("Using default configuration")
                self.config = SystemConfig()
            except Exception as e:
                logger.error(f"❌ Unexpected error loading config from {self.config_path}: {e}")
                logger.info("Using default configuration")
                self.config = SystemConfig()
        else:
            logger.info("ℹ️  No config file found, using default configuration")
            self.config = SystemConfig()
            
        # Validate configuration
        try:
            self._validate_config()
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
        
        return self.config
    
    def save_config(self, config: Optional[SystemConfig] = None) -> None:
        """Save configuration to file."""
        if config:
            self.config = config
            
        if not self.config:
            raise ValueError("No configuration to save")
            
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = asdict(self.config)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {self.config_path}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration, loading if necessary."""
        if not self.config:
            self.load_config()
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        if not self.config:
            self.load_config()
            
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
                
        # Validate updated configuration
        self._validate_config()
    
    def _convert_dict_to_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nested dictionaries to appropriate dataclass instances."""
        converted = {}
        
        # Map of config sections to their dataclass types
        section_types = {
            'system': SystemMetaConfig,
            'database': DatabaseConfig,
            'api': APIConfig,
            'data': DataConfig,
            'features': FeatureConfig,
            'models': ModelConfig,
            'pipeline': PipelineConfig,
            'logging': LoggingConfig,
            'prediction': PredictionConfig,
            'directories': DirectoriesConfig,
            'processing': ProcessingConfig,
            'api_keys': APIKeysConfig
        }
        
        for key, value in config_dict.items():
            if key in section_types and isinstance(value, dict):
                converted[key] = section_types[key](**value)
            else:
                converted[key] = value
                
        return converted
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not self.config:
            return
            
        # Validate database configuration
        if self.config.database.type not in ["postgresql", "sqlite"]:
            raise ValueError(f"Invalid database type: {self.config.database.type}")
            
        # Validate model configuration
        valid_models = ["xgboost", "lightgbm", "random_forest", "gradient_boosting", "neural_network"]
        for model_type in self.config.models.model_types:
            if model_type not in valid_models:
                raise ValueError(f"Invalid model type: {model_type}")
                
        # Validate feature configuration
        if self.config.features.min_games_threshold < 1:
            raise ValueError("min_games_threshold must be at least 1")
            
        # Validate data configuration
        if self.config.data.current_season < 2020:
            raise ValueError("current_season must be 2020 or later")
            
        # Create required directories
        self._create_required_directories()
        
        logger.info("Configuration validation passed")
    
    def _create_required_directories(self) -> None:
        """Create required directories if they don't exist."""
        directories = [
            "data",
            "data/raw",
            "data/processed", 
            "data/features",
            "logs",
            "models",
            "models/trained",
            "models/performance",
            Path(self.config.logging.log_file).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return _config_manager.get_config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
    """Load configuration from file."""
    return _config_manager.load_config(config_path)


def save_config(config: Optional[SystemConfig] = None) -> None:
    """Save configuration to file."""
    _config_manager.save_config(config)


def update_config(**kwargs) -> None:
    """Update global configuration."""
    _config_manager.update_config(**kwargs)
