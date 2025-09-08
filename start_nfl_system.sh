#!/bin/bash
# NFL Prediction System - Production Start Script

echo "🏈 Starting NFL Prediction System..."
echo "=================================="

# Activate conda environment if needed
if command -v conda &> /dev/null; then
    conda activate nfl 2>/dev/null || true
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CONDA_DEFAULT_ENV=nfl

# Check system status
echo "Checking system status..."
python production_cli.py status

echo ""
echo "🎉 System ready for predictions!"
echo ""
echo "Available commands:"
echo "  python production_cli.py predict --position QB"
echo "  python production_cli.py predict --player-name 'Josh Allen'"
echo "  python production_cli.py report --position RB"
echo "  python production_cli.py analyze --position WR"
echo ""
