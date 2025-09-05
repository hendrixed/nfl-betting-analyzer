#!/usr/bin/env python3
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
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
        log_level="info"
    )
