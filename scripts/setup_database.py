#!/usr/bin/env python3
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
