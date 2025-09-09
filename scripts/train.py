#!/usr/bin/env python3
"""
Thin wrapper for model training operations.
Business logic should be in core/models/ modules.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Train models - delegates to CLI"""
    import subprocess
    subprocess.run([sys.executable, "nfl_cli.py", "train"] + sys.argv[1:])

if __name__ == "__main__":
    main()
