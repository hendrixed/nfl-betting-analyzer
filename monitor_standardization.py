"""
Monitor the ongoing historical data standardization process
"""

import os
import time
import psutil
from datetime import datetime

def monitor_standardization():
    """Monitor the standardization process"""
    
    print("📊 Monitoring Historical Data Standardization Process...")
    print("="*60)
    
    # Check if standardization process is running
    running_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'])
            if 'historical_data_standardizer' in cmdline:
                running_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if running_processes:
        print(f"✅ Found {len(running_processes)} standardization processes running")
        for proc in running_processes:
            print(f"   PID: {proc['pid']} - {proc['name']}")
    else:
        print("❌ No standardization processes found running")
    
    # Check database file size (indicates progress)
    db_file = "nfl_predictions.db"
    if os.path.exists(db_file):
        size_mb = os.path.getsize(db_file) / (1024 * 1024)
        print(f"📁 Database size: {size_mb:.1f} MB")
    else:
        print("❌ Database file not found")
    
    # Check log files for progress
    log_files = [f for f in os.listdir('.') if f.endswith('.log')]
    if log_files:
        print(f"📝 Log files found: {log_files}")
        
        # Show last few lines of most recent log
        latest_log = max(log_files, key=os.path.getctime)
        print(f"\n📋 Last 5 lines from {latest_log}:")
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                for line in lines[-5:]:
                    print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Error reading log: {e}")
    
    print(f"\n⏰ Monitor time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_standardization()
