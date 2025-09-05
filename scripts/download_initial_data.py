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
        database_url="sqlite:////home/money/CascadeProjects/nfl-betting-analyzer/data/nfl_predictions.db",
        api_keys={},
        data_sources={"nfl_data_py": True},
        seasons=[2020, 2021, 2022, 2023, 2024],
        current_season=2024,
        current_week=1,
        enable_live_data=False
    )
    
    collector = NFLDataCollector(config)
    await collector.initialize_database()
    
    # Download historical data
    for season in [2020, 2021, 2022, 2023, 2024]:
        print(f"Downloading data for {season} season...")
        await collector.collect_season_data(season)
        
    print("Initial data download completed!")

if __name__ == "__main__":
    asyncio.run(main())