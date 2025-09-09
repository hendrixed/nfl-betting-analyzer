# System Overview

This document provides a high-level overview of the NFL Betting Analyzer system. It is deliberately plain ASCII to ensure compatibility with Windows default encodings.

## Goals

- Provide a reliable pipeline for collecting, processing, and predicting NFL player and team performance.
- Maintain a simple and clear configuration surface using YAML and dataclasses.
- Support modular components that can be tested independently.

## Architecture

- Database layer using SQLAlchemy ORM with models for Player, Game, PlayerGameStats, and predictions.
- Data collection layer that loads snapshots and live data into the database.
- Feature engineering layer that computes rolling averages, opponent adjustments, and situational features.
- Model layer with multiple algorithms and an optional ensemble method.
- API and CLI layers for serving predictions and operating the system.

## Data Flow

Raw data -> Database -> Feature Engineering -> Modeling -> Predictions -> API and CLI

## Key Modules

- database_models.py: ORM models and table creation helpers
- data_collector.py: snapshot ingestion and optional live data integration
- core/models/feature_engineering.py: advanced feature generation utilities
- core/models/prediction_models.py: model training and inference
- prediction_pipeline.py: orchestration and scheduling
- config/config_manager.py: configuration loading and validation

## Environments

- Development: default settings for local testing using SQLite
- Production: configurable database URL and logging levels

## Logging

The system uses Python logging with a YAML configuration in config/logging.yaml. Logs can be directed to file and console.

## Tests

The repository contains a test suite that validates imports, configuration, data quality, models, and a basic integration flow. Keep tests deterministic by setting random seeds in model code where applicable.
