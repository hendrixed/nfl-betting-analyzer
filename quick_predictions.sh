#!/bin/bash
# Quick NFL Predictions Script

echo "🎯 Quick NFL Predictions"
echo "======================"

# Top QBs
echo "🏆 Top QBs:"
python production_cli.py predict --position QB --top 3

echo ""

# Top RBs  
echo "🏆 Top RBs:"
python production_cli.py predict --position RB --top 3

echo ""

# Top WRs
echo "🏆 Top WRs:"
python production_cli.py predict --position WR --top 3

echo ""

# Top TEs
echo "🏆 Top TEs:"
python production_cli.py predict --position TE --top 3
