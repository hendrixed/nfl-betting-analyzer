#!/usr/bin/env python3
"""
Enhanced Logging and Error Handling System
Centralized logging configuration with structured logging, error tracking, and monitoring
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import functools
import time

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogContext:
    """Structured log context."""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    player_id: Optional[str] = None
    game_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_entry['context'] = asdict(record.context)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'context']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class NFLLogger:
    """Enhanced logger for NFL betting analyzer system."""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup logging handlers."""
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        
        # Performance log handler
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_performance.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())
        
        # Add filter for performance logs
        perf_handler.addFilter(lambda record: hasattr(record, 'performance'))
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(perf_handler)
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, context, **kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, exc_info: bool = True, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, context, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, exc_info: bool = True, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, context, exc_info=exc_info, **kwargs)
    
    def performance(self, message: str, execution_time_ms: float, context: Optional[LogContext] = None, **kwargs):
        """Log performance metrics."""
        if context:
            context.execution_time_ms = execution_time_ms
        else:
            context = LogContext(
                component=self.name,
                operation="performance_log",
                execution_time_ms=execution_time_ms
            )
        
        extra = {'performance': True, **kwargs}
        self._log(LogLevel.INFO, message, context, extra=extra)
    
    def _log(self, level: LogLevel, message: str, context: Optional[LogContext] = None, 
             exc_info: bool = False, extra: Optional[Dict] = None):
        """Internal logging method."""
        
        log_extra = extra or {}
        
        if context:
            log_extra['context'] = context
        
        getattr(self.logger, level.value.lower())(
            message, 
            exc_info=exc_info,
            extra=log_extra
        )

class ErrorTracker:
    """Track and analyze errors across the system."""
    
    def __init__(self, logger: NFLLogger):
        self.logger = logger
        self.error_counts = {}
        self.error_patterns = {}
    
    def track_error(self, error: Exception, context: Optional[LogContext] = None):
        """Track an error occurrence."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Count errors
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Log the error
        self.logger.error(
            f"Error tracked: {error_type} - {error_message}",
            context=context,
            error_type=error_type,
            error_count=self.error_counts[error_type]
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': len(self.error_counts),
            'most_common_errors': sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'timestamp': datetime.utcnow().isoformat()
        }

def performance_monitor(component: str, operation: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = NFLLogger(f"{component}.performance")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                context = LogContext(
                    component=component,
                    operation=operation,
                    execution_time_ms=execution_time
                )
                
                logger.performance(
                    f"Operation completed: {operation}",
                    execution_time,
                    context=context,
                    function_name=func.__name__,
                    success=True
                )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                context = LogContext(
                    component=component,
                    operation=operation,
                    execution_time_ms=execution_time
                )
                
                logger.error(
                    f"Operation failed: {operation} - {str(e)}",
                    context=context,
                    function_name=func.__name__,
                    success=False
                )
                raise
        
        return wrapper
    return decorator

def error_handler(component: str, operation: str, reraise: bool = True):
    """Decorator for consistent error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = NFLLogger(f"{component}.errors")
            error_tracker = ErrorTracker(logger)
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                context = LogContext(
                    component=component,
                    operation=operation
                )
                
                error_tracker.track_error(e, context)
                
                if reraise:
                    raise
                else:
                    logger.warning(
                        f"Error suppressed in {operation}: {str(e)}",
                        context=context
                    )
                    return None
        
        return wrapper
    return decorator

class LoggingConfig:
    """Centralized logging configuration."""
    
    @staticmethod
    def setup_system_logging(log_level: str = "INFO", log_dir: str = "logs"):
        """Setup system-wide logging configuration."""
        
        # Create log directory
        Path(log_dir).mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup handlers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        file_handler = logging.handlers.RotatingFileHandler(
            Path(log_dir) / "system.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Configure third-party loggers
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        return root_logger

# Convenience functions for common logging patterns
def log_prediction_start(logger: NFLLogger, player_id: str, operation: str):
    """Log prediction operation start."""
    context = LogContext(
        component="prediction",
        operation=operation,
        player_id=player_id
    )
    logger.info(f"Starting {operation} for player {player_id}", context=context)

def log_prediction_success(logger: NFLLogger, player_id: str, operation: str, 
                          confidence: float, execution_time_ms: float):
    """Log successful prediction."""
    context = LogContext(
        component="prediction",
        operation=operation,
        player_id=player_id,
        execution_time_ms=execution_time_ms
    )
    logger.info(
        f"Prediction completed for {player_id} with confidence {confidence:.3f}",
        context=context,
        confidence=confidence
    )

def log_api_request(logger: NFLLogger, endpoint: str, user_id: str, 
                   request_id: str, execution_time_ms: float):
    """Log API request."""
    context = LogContext(
        component="api",
        operation=endpoint,
        user_id=user_id,
        request_id=request_id,
        execution_time_ms=execution_time_ms
    )
    logger.info(f"API request to {endpoint} completed", context=context)

def log_database_operation(logger: NFLLogger, operation: str, table: str, 
                          rows_affected: int, execution_time_ms: float):
    """Log database operation."""
    context = LogContext(
        component="database",
        operation=operation,
        execution_time_ms=execution_time_ms
    )
    logger.info(
        f"Database {operation} on {table}: {rows_affected} rows affected",
        context=context,
        table=table,
        rows_affected=rows_affected
    )

# Example usage and testing
def main():
    """Example usage of enhanced logging system."""
    
    # Setup system logging
    LoggingConfig.setup_system_logging("INFO", "logs")
    
    # Create component loggers
    prediction_logger = NFLLogger("prediction")
    api_logger = NFLLogger("api")
    
    # Example logging with context
    context = LogContext(
        component="prediction",
        operation="generate_ultimate_prediction",
        player_id="pmahomes_qb",
        user_id="user123"
    )
    
    prediction_logger.info("Starting prediction generation", context=context)
    
    # Example performance monitoring
    @performance_monitor("prediction", "ultimate_analysis")
    def example_prediction():
        time.sleep(0.1)  # Simulate work
        return {"fantasy_points": 20.5, "confidence": 0.85}
    
    # Example error handling
    @error_handler("prediction", "data_validation", reraise=False)
    def example_validation():
        raise ValueError("Invalid player data")
    
    # Test the decorators
    result = example_prediction()
    validation_result = example_validation()
    
    prediction_logger.info("Example logging completed", context=context)
    
    print("✅ Enhanced logging system demonstration completed")
    print("📁 Check the 'logs' directory for log files")

if __name__ == "__main__":
    main()
