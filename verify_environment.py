"""
Environment Verification Script
Verify that all dependencies are available in the NFL conda environment
"""

import sys
import os
import subprocess

def verify_conda_environment():
    """Verify we're running in the correct conda environment"""
    
    print("=== ENVIRONMENT VERIFICATION ===")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Current Conda Environment: {conda_env}")
    
    if conda_env != 'nfl':
        print("❌ WARNING: Not running in 'nfl' conda environment!")
        print("Please run: conda activate nfl")
        return False
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'nfl_data_py',
        'pandas', 
        'numpy',
        'sqlalchemy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                import sklearn
                print(f"✅ {package}: Available (v{sklearn.__version__})")
            else:
                __import__(package)
                print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    # Test NFL data access
    try:
        import nfl_data_py as nfl
        teams = nfl.import_team_desc()
        print(f"✅ NFL Data Access: {len(teams)} teams loaded")
    except Exception as e:
        print(f"❌ NFL Data Access Failed: {e}")
        return False
    
    print("\n✅ Environment verification PASSED")
    return True

def setup_project_paths():
    """Setup Python path for project modules"""
    
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Added {current_dir} to Python path")
    
    # Verify local modules can be imported
    try:
        from core.data.data_foundation import PlayerRole, MasterPlayer
        print("✅ Local modules: data_foundation imported successfully")
    except ImportError as e:
        print(f"❌ Local modules import failed: {e}")
        return False
    
    try:
        from core.database_models import Player, PlayerGameStats
        print("✅ Local modules: database_models imported successfully")
    except ImportError as e:
        print(f"❌ Database models import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = verify_conda_environment()
    if success:
        success = setup_project_paths()
    
    if success:
        print("\n🎉 Environment setup COMPLETE - Ready to continue implementation")
        exit(0)
    else:
        print("\n❌ Environment setup FAILED - Fix issues before continuing")
        exit(1)
