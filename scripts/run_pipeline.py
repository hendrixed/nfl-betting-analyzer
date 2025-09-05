#!/usr/bin/env python3
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
