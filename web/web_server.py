"""
Simple Flask web server for NFL Betting Analyzer interface
"""
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import sys
import os
import requests
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database_models import get_db_session, Player
from core.models.streamlined_models import StreamlinedNFLModels

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models instance
models = None

def initialize_models():
    """Initialize the streamlined models."""
    global models
    try:
        session = get_db_session()
        models = StreamlinedNFLModels(session)
        logger.info("✅ Streamlined models initialized for web interface")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")

# Initialize models at startup
with app.app_context():
    initialize_models()

@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')

@app.route('/api/predict/<player_id>')
def predict_player(player_id):
    """Get prediction for a specific player."""
    try:
        if not models:
            return jsonify({
                'error': 'Models not initialized',
                'status': 'error'
            }), 503
        
        # Get prediction from streamlined models
        prediction_result = models.predict_player(player_id)
        
        if not prediction_result:
            return jsonify({
                'error': f'No prediction available for player {player_id}',
                'status': 'error'
            }), 404
        
        # Format response to match frontend expectations
        response = {
            'player_id': prediction_result.player_id,
            'player_name': prediction_result.player_name,
            'position': prediction_result.position,
            'basic_prediction': {
                'fantasy_points': round(prediction_result.predicted_value, 2),
                'confidence': round(prediction_result.confidence, 3),
                'model_used': prediction_result.model_used
            },
            'market_analysis': {
                'current_line': 17.5,  # Mock data
                'line_movement': '+1.0',
                'public_betting': 65.2,
                'edge': 0.057
            },
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error for {player_id}: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Get predictions for multiple players."""
    try:
        if not models:
            return jsonify({
                'error': 'Models not initialized',
                'status': 'error'
            }), 503
        
        data = request.get_json()
        player_ids = data.get('player_ids', [])
        
        if not player_ids:
            return jsonify({
                'error': 'No player IDs provided',
                'status': 'error'
            }), 400
        
        results = []
        for player_id in player_ids:
            prediction_result = models.predict_player(player_id.strip())
            if prediction_result:
                results.append({
                    'player_id': prediction_result.player_id,
                    'player_name': prediction_result.player_name,
                    'position': prediction_result.position,
                    'predicted_value': round(prediction_result.predicted_value, 2),
                    'confidence': round(prediction_result.confidence, 3),
                    'model_used': prediction_result.model_used
                })
        
        return jsonify({
            'predictions': results,
            'total_processed': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/players')
def get_players():
    """Get list of available players."""
    try:
        session = get_db_session()
        players = session.query(Player).filter(Player.is_active == True).limit(50).all()
        
        player_list = []
        for player in players:
            player_list.append({
                'player_id': player.player_id,
                'name': player.name,
                'position': player.position,
                'team': player.current_team
            })
        
        return jsonify({
            'players': player_list,
            'total': len(player_list),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Players fetch error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'models_loaded': models is not None,
        'version': '1.0.0'
    }
    
    if not models:
        status['status'] = 'degraded'
        status['warning'] = 'Models not initialized'
    
    return jsonify(status)

@app.route('/api/system-status')
def system_status():
    """Get detailed system status."""
    try:
        session = get_db_session()
        player_count = session.query(Player).count()
        active_players = session.query(Player).filter(Player.is_active == True).count()
        
        status = {
            'api_status': 'healthy',
            'models_status': 'loaded' if models else 'not_loaded',
            'database_status': 'connected',
            'player_count': player_count,
            'active_players': active_players,
            'features': {
                'predictions': True,
                'batch_predictions': True,
                'real_time': False,  # WebSocket not implemented in Flask version
                'caching': False     # Redis not implemented in Flask version
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    print("⚠️ The Flask web server is deprecated.")
    print("Please use the unified FastAPI app instead:")
    print("  uvicorn api.app:app --reload --port 8000")
    print("Or via CLI:")
    print("  python nfl_cli.py run-api")
    raise SystemExit(0)
