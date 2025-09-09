#!/usr/bin/env python3
"""
NFL Betting Analyzer - Unified System Startup
Starts both Flask web interface and FastAPI prediction service
"""

import subprocess
import sys
import os
import time
import signal
import socket
from pathlib import Path

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking system dependencies...")
    
    try:
        from core.database_models import get_db_session, Player
        session = get_db_session()
        player_count = session.query(Player).count()
        session.close()
        print(f"✅ Database connection verified ({player_count} players)")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    try:
        from core.models.streamlined_models import StreamlinedNFLModels
        print("✅ Streamlined models available")
    except Exception as e:
        print(f"❌ Streamlined models unavailable: {e}")
        return False
    
    return True

def start_flask_server(port=None):
    """Start the Flask web server."""
    if port is None:
        port = find_available_port(5000)
    
    if port is None:
        print("❌ No available ports for Flask server")
        return None
    
    print(f"🚀 Starting Flask Web Interface on port {port}...")
    
    env = os.environ.copy()
    env['FLASK_PORT'] = str(port)
    
    # Modify the web server to use environment port if available
    web_server_path = Path(__file__).parent / "web" / "web_server.py"
    
    try:
        process = subprocess.Popen([
            sys.executable, str(web_server_path)
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give it a moment to start
        time.sleep(2)
        
        if process.poll() is None:  # Still running
            print(f"✅ Flask Web Interface started successfully")
            print(f"📊 Web Interface: http://localhost:{port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Flask server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start Flask server: {e}")
        return None

def start_fastapi_server(port=8000):
    """Start the FastAPI prediction service."""
    if not find_available_port(port, 1):
        port = find_available_port(8000, 10)
        if port is None:
            print("❌ No available ports for FastAPI server")
            return None
    
    print(f"🚀 Starting FastAPI Prediction Service on port {port} (api.app)...")
    
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give it a moment to start
        time.sleep(3)
        
        if process.poll() is None:  # Still running
            print(f"✅ FastAPI Prediction Service started successfully")
            print(f"🔮 API Service: http://localhost:{port}")
            print(f"📚 API Docs: http://localhost:{port}/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ FastAPI server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start FastAPI server: {e}")
        return None

def main():
    """Main startup function."""
    print("🏈 NFL Betting Analyzer - System Startup")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Dependency check failed. Please fix issues before starting.")
        return 1
    
    processes = []
    
    try:
        # Start Flask web server
        flask_process = start_flask_server()
        if flask_process:
            processes.append(('Flask Web Interface', flask_process))
        
        # Start FastAPI server
        fastapi_process = start_fastapi_server()
        if fastapi_process:
            processes.append(('FastAPI Prediction Service', fastapi_process))
        
        if not processes:
            print("❌ Failed to start any services")
            return 1
        
        print("\n" + "=" * 50)
        print("🎉 NFL Betting Analyzer Started Successfully!")
        print(f"✅ {len(processes)} service(s) running")
        print("\n🔧 Available Services:")
        
        for name, process in processes:
            print(f"   • {name}")
        
        print("\n💡 Tips:")
        print("   • Press Ctrl+C to stop all services")
        print("   • Check logs above for any warnings")
        print("   • Redis caching is optional (warnings are normal)")
        
        # Wait for interrupt
        try:
            while True:
                # Check if processes are still running
                running_processes = []
                for name, process in processes:
                    if process.poll() is None:
                        running_processes.append((name, process))
                    else:
                        print(f"⚠️  {name} stopped unexpectedly")
                
                processes = running_processes
                if not processes:
                    print("❌ All services stopped")
                    break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            
    except Exception as e:
        print(f"❌ Startup error: {e}")
        return 1
    
    finally:
        # Clean shutdown
        print("🧹 Stopping services...")
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔪 {name} force killed")
            except Exception as e:
                print(f"⚠️  Error stopping {name}: {e}")
    
    print("👋 NFL Betting Analyzer shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
