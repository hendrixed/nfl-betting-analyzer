#!/usr/bin/env python3
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
