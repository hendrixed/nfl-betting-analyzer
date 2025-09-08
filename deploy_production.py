"""
Production Deployment Script

Deploys the NFL prediction system for production use
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

class ProductionDeployer:
    """Handle production deployment"""
    
    def __init__(self):
        self.deployment_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = Path(f"backup_{self.deployment_date}")
        
    def deploy(self):
        """Run production deployment"""
        
        print("🚀 NFL PREDICTION SYSTEM - PRODUCTION DEPLOYMENT")
        print("=" * 60)
        print(f"Deployment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        deployment_steps = [
            ("Environment Check", self._check_environment),
            ("Backup Current System", self._backup_system),
            ("Install Dependencies", self._install_dependencies),
            ("Database Optimization", self._optimize_database),
            ("System Verification", self._verify_deployment),
            ("Create Production Scripts", self._create_production_scripts)
        ]
        
        for step_name, step_function in deployment_steps:
            print(f"🔄 {step_name}...")
            
            try:
                step_function()
                print(f"✅ {step_name}: SUCCESS")
            except Exception as e:
                print(f"❌ {step_name}: FAILED - {e}")
                return False
            
            print()
        
        print("🎉 PRODUCTION DEPLOYMENT COMPLETE!")
        print("   The NFL prediction system is now ready for production use.")
        print()
        print("Next steps:")
        print("  1. Run 'python production_cli.py verify' to confirm readiness")
        print("  2. Run 'python production_cli.py demo' to test predictions")
        print("  3. Begin generating NFL predictions for betting analysis")
        
        return True
    
    def _check_environment(self):
        """Check deployment environment"""
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8+ required")
        
        # Check required files
        required_files = [
            'real_time_nfl_system.py',
            'database_models.py',
            'nfl_predictions.db',
            'production_cli.py',
            'verify_production_readiness.py',
            'live_prediction_demo.py'
        ]
        
        for file in required_files:
            if not Path(file).exists():
                raise Exception(f"Required file missing: {file}")
        
        print("   ✓ Python version compatible")
        print("   ✓ All required files present")
    
    def _backup_system(self):
        """Backup current system"""
        
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup key files
        backup_files = [
            'nfl_predictions.db',
            'production_cli.py',
            'verify_production_readiness.py',
            'live_prediction_demo.py'
        ]
        
        for file in backup_files:
            if Path(file).exists():
                shutil.copy2(file, self.backup_dir / file)
        
        # Backup models directory
        if Path('models').exists():
            shutil.copytree('models', self.backup_dir / 'models', dirs_exist_ok=True)
        
        print(f"   ✓ Backup created in {self.backup_dir}")
    
    def _install_dependencies(self):
        """Install production dependencies"""
        
        # Update requirements if needed
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--upgrade',
                'pandas', 'numpy', 'scikit-learn', 'sqlalchemy', 'click', 'nfl-data-py'
            ], check=True, capture_output=True)
            
            print("   ✓ Dependencies updated")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Dependency update warning: {e}")
    
    def _optimize_database(self):
        """Optimize database for production"""
        
        import sqlite3
        
        # Basic database optimization
        conn = sqlite3.connect('nfl_predictions.db')
        
        # Vacuum and analyze
        conn.execute('VACUUM')
        conn.execute('ANALYZE')
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_players_position ON players(position)",
            "CREATE INDEX IF NOT EXISTS idx_players_team ON players(current_team)",
            "CREATE INDEX IF NOT EXISTS idx_players_active ON players(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_stats_player ON player_game_stats(player_id)",
            "CREATE INDEX IF NOT EXISTS idx_stats_fantasy ON player_game_stats(fantasy_points_ppr)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                print(f"   ⚠️ Index creation warning: {e}")
        
        conn.commit()
        conn.close()
        
        print("   ✓ Database optimized with indexes")
    
    def _verify_deployment(self):
        """Verify deployment success"""
        
        from verify_production_readiness import ProductionReadinessVerifier
        
        verifier = ProductionReadinessVerifier()
        assessment = verifier.run_comprehensive_verification()
        
        if not assessment['production_ready']:
            raise Exception("Deployment verification failed - system not production ready")
        
        print("   ✓ Production readiness verified")
    
    def _create_production_scripts(self):
        """Create production helper scripts"""
        
        # Create start script
        start_script = """#!/bin/bash
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
"""
        
        with open('start_nfl_system.sh', 'w') as f:
            f.write(start_script)
        
        os.chmod('start_nfl_system.sh', 0o755)
        
        # Create quick prediction script
        quick_predict_script = """#!/bin/bash
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
"""
        
        with open('quick_predictions.sh', 'w') as f:
            f.write(quick_predict_script)
        
        os.chmod('quick_predictions.sh', 0o755)
        
        # Create README for production use
        readme_content = """# NFL Prediction System - Production Guide

## Quick Start

1. **Check System Status**
   ```bash
   ./start_nfl_system.sh
   ```

2. **Generate Predictions**
   ```bash
   # Specific player
   python production_cli.py predict --player-name "Josh Allen"
   
   # Top players by position
   python production_cli.py predict --position QB --top 5
   
   # Team predictions
   python production_cli.py predict --team KC --top 5
   ```

3. **Generate Reports**
   ```bash
   # Console report
   python production_cli.py report --position QB
   
   # JSON report
   python production_cli.py report --format json --position RB
   ```

4. **Performance Analysis**
   ```bash
   # Position analysis
   python production_cli.py analyze --position WR
   
   # League overview
   python production_cli.py analyze
   ```

## Available Commands

- `verify` - Verify system production readiness
- `demo` - Run live prediction demonstration
- `predict` - Generate NFL predictions
- `report` - Generate performance reports
- `status` - Check system status
- `analyze` - Analyze performance trends

## Production Features

✅ **1,488 NFL Players** - Comprehensive player database
✅ **33,642 Statistical Records** - Extensive historical data
✅ **Multi-Position Support** - QB, RB, WR, TE predictions
✅ **Fantasy Points Projections** - PPR scoring system
✅ **Betting Recommendations** - Confidence-based analysis
✅ **Performance Analytics** - Trend and volatility analysis

## Betting Analysis Workflow

1. **Daily Predictions**
   ```bash
   ./quick_predictions.sh
   ```

2. **Player Research**
   ```bash
   python production_cli.py predict --player-name "Player Name"
   python production_cli.py analyze --position QB --min-games 5
   ```

3. **Report Generation**
   ```bash
   python production_cli.py report --format json > daily_report.json
   ```

## System Monitoring

- Run `python production_cli.py status` regularly
- Check `python production_cli.py verify` for health checks
- Monitor prediction accuracy against actual results

## Support

The system is fully operational and ready for production NFL predictions and betting analysis.
"""
        
        with open('PRODUCTION_README.md', 'w') as f:
            f.write(readme_content)
        
        print("   ✓ Production scripts created")
        print("     - start_nfl_system.sh")
        print("     - quick_predictions.sh") 
        print("     - PRODUCTION_README.md")

def main():
    """Run production deployment"""
    
    deployer = ProductionDeployer()
    success = deployer.deploy()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
