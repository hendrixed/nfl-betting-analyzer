#!/usr/bin/env python3
"""
NFL Data Browser - Comprehensive Web Interface
Browse players, teams, games, stats, schedules, rosters, and predictions
"""

from flask import Flask, render_template_string, jsonify, request
import sys
import os
import socket

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database_models import get_db_session, Player, Game
from core.models.streamlined_models import StreamlinedNFLModels

app = Flask(__name__)

# Global variables
models = None

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

def initialize_models():
    """Initialize models."""
    global models
    try:
        session = get_db_session()
        models = StreamlinedNFLModels(session)
        print("✅ Models initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize models: {e}")

# Initialize at startup
initialize_models()

# Main template
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL Data Browser</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-hover:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        .tab-active { background: #3b82f6; color: white; }
        .tab-inactive { background: #e5e7eb; color: #374151; }
    </style>
</head>
<body class="bg-gray-50 min-h-screen" x-data="nflBrowser()">
    <!-- Header -->
    <div class="gradient-bg text-white p-6 mb-8">
        <div class="container mx-auto">
            <h1 class="text-4xl font-bold mb-2">🏈 NFL Data Browser</h1>
            <p class="text-xl opacity-90">Browse Players • Teams • Games • Stats • Predictions</p>
            <div class="mt-4 flex items-center space-x-4">
                <div class="flex items-center">
                    <div class="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                    <span>Database: {{ total_players }} players, {{ total_games }} games</span>
                </div>
                <div class="flex items-center">
                    <div class="w-3 h-3 bg-blue-400 rounded-full mr-2"></div>
                    <span>Active Players: {{ active_players }}</span>
                </div>
            </div>
        </div>
    </div>

    <div class="container mx-auto px-4">
        <!-- Navigation Tabs -->
        <div class="flex flex-wrap mb-6 border-b">
            <button @click="activeTab = 'teams'" 
                    :class="activeTab === 'teams' ? 'tab-active' : 'tab-inactive'"
                    class="px-6 py-3 font-medium rounded-t-lg mr-2 transition-colors">
                🏟️ Teams
            </button>
            <button @click="activeTab = 'players'" 
                    :class="activeTab === 'players' ? 'tab-active' : 'tab-inactive'"
                    class="px-6 py-3 font-medium rounded-t-lg mr-2 transition-colors">
                🏃 Players
            </button>
            <button @click="activeTab = 'games'" 
                    :class="activeTab === 'games' ? 'tab-active' : 'tab-inactive'"
                    class="px-6 py-3 font-medium rounded-t-lg mr-2 transition-colors">
                🎮 Games
            </button>
            <button @click="activeTab = 'predictions'" 
                    :class="activeTab === 'predictions' ? 'tab-active' : 'tab-inactive'"
                    class="px-6 py-3 font-medium rounded-t-lg mr-2 transition-colors">
                🔮 Predictions
            </button>
            <button @click="activeTab = 'stats'" 
                    :class="activeTab === 'stats' ? 'tab-active' : 'tab-inactive'"
                    class="px-6 py-3 font-medium rounded-t-lg mr-2 transition-colors">
                📊 Stats
            </button>
        </div>

        <!-- Teams Tab -->
        <div x-show="activeTab === 'teams'" class="space-y-6">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-bold mb-4">NFL Teams</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4" x-show="!selectedTeam">
                    <template x-for="team in teams">
                        <button @click="selectTeam(team)" 
                                class="p-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors text-center">
                            <div class="font-bold text-lg" x-text="team"></div>
                        </button>
                    </template>
                </div>
                
                <!-- Team Detail -->
                <div x-show="selectedTeam" class="mt-6">
                    <div class="flex items-center mb-4">
                        <button @click="selectedTeam = null" class="mr-4 text-blue-600 hover:text-blue-800">← Back to Teams</button>
                        <h3 class="text-xl font-bold" x-text="selectedTeam + ' Roster'"></h3>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <template x-for="player in teamPlayers">
                            <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
                                <div class="font-semibold" x-text="player.name"></div>
                                <div class="text-sm text-gray-600" x-text="player.position"></div>
                                <button @click="selectPlayer(player)" class="mt-2 text-blue-600 hover:text-blue-800 text-sm">View Details</button>
                            </div>
                        </template>
                    </div>
                </div>
            </div>
        </div>

        <!-- Players Tab -->
        <div x-show="activeTab === 'players'" class="space-y-6">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-bold mb-4">Player Search</h2>
                <div class="flex space-x-4 mb-6">
                    <input x-model="playerSearch" @input="searchPlayers()" 
                           placeholder="Search players..." 
                           class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                    <select x-model="positionFilter" @change="searchPlayers()" 
                            class="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                        <option value="">All Positions</option>
                        <option value="QB">QB</option>
                        <option value="RB">RB</option>
                        <option value="WR">WR</option>
                        <option value="TE">TE</option>
                        <option value="K">K</option>
                        <option value="DST">DST</option>
                    </select>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <template x-for="player in searchResults">
                        <div class="border rounded-lg p-4 hover:shadow-md transition-shadow card-hover">
                            <div class="font-semibold" x-text="player.name"></div>
                            <div class="text-sm text-gray-600" x-text="player.position + ' - ' + player.team"></div>
                            <div class="text-xs text-gray-500 mt-1" x-text="player.is_active ? 'Active' : 'Inactive'"></div>
                            <button @click="selectPlayer(player)" class="mt-2 text-blue-600 hover:text-blue-800 text-sm">View Details</button>
                        </div>
                    </template>
                </div>
            </div>
        </div>

        <!-- Games Tab -->
        <div x-show="activeTab === 'games'" class="space-y-6">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-bold mb-4">NFL Games</h2>
                <div class="space-y-4">
                    <template x-for="game in games">
                        <div class="border rounded-lg p-4 hover:shadow-md transition-shadow">
                            <div class="flex justify-between items-center">
                                <div>
                                    <div class="font-semibold" x-text="game.away_team + ' @ ' + game.home_team"></div>
                                    <div class="text-sm text-gray-600" x-text="game.game_date"></div>
                                    <div class="text-xs text-gray-500" x-text="'Week ' + game.week + ' - ' + game.season"></div>
                                </div>
                                <div class="text-right">
                                    <div x-show="game.home_score !== null" class="font-bold">
                                        <span x-text="game.away_score"></span> - <span x-text="game.home_score"></span>
                                    </div>
                                    <div x-show="game.home_score === null" class="text-gray-500">Scheduled</div>
                                </div>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
        </div>

        <!-- Predictions Tab -->
        <div x-show="activeTab === 'predictions'" class="space-y-6">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-bold mb-4">Player Predictions</h2>
                <div class="mb-6">
                    <input x-model="predictionInput" 
                           placeholder="Enter player ID (e.g., pmahomes)" 
                           class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                    <button @click="getPrediction()" 
                            class="mt-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                        Get Prediction
                    </button>
                </div>
                
                <div x-show="predictionResult" class="mt-6 p-4 bg-gray-50 rounded-lg">
                    <h3 class="font-bold mb-2">Prediction Result:</h3>
                    <pre x-text="JSON.stringify(predictionResult, null, 2)" class="text-sm"></pre>
                </div>
            </div>
        </div>

        <!-- Stats Tab -->
        <div x-show="activeTab === 'stats'" class="space-y-6">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-bold mb-4">Database Statistics</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div class="text-center">
                        <div class="text-3xl font-bold text-blue-600">{{ total_players }}</div>
                        <div class="text-gray-600">Total Players</div>
                    </div>
                    <div class="text-center">
                        <div class="text-3xl font-bold text-green-600">{{ active_players }}</div>
                        <div class="text-gray-600">Active Players</div>
                    </div>
                    <div class="text-center">
                        <div class="text-3xl font-bold text-purple-600">{{ total_games }}</div>
                        <div class="text-gray-600">Total Games</div>
                    </div>
                    <div class="text-center">
                        <div class="text-3xl font-bold text-orange-600">{{ qb_count }}</div>
                        <div class="text-gray-600">Active QBs</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function nflBrowser() {
            return {
                activeTab: 'teams',
                teams: {{ teams | tojsonfilter }},
                selectedTeam: null,
                teamPlayers: [],
                playerSearch: '',
                positionFilter: '',
                searchResults: [],
                games: {{ recent_games | tojsonfilter }},
                predictionInput: '',
                predictionResult: null,

                async selectTeam(team) {
                    this.selectedTeam = team;
                    try {
                        const response = await fetch(`/api/team/${team}/players`);
                        this.teamPlayers = await response.json();
                    } catch (error) {
                        console.error('Error fetching team players:', error);
                        this.teamPlayers = [];
                    }
                },

                async searchPlayers() {
                    if (this.playerSearch.length < 2 && !this.positionFilter) return;
                    
                    try {
                        const params = new URLSearchParams();
                        if (this.playerSearch) params.append('search', this.playerSearch);
                        if (this.positionFilter) params.append('position', this.positionFilter);
                        
                        const response = await fetch(`/api/players/search?${params}`);
                        this.searchResults = await response.json();
                    } catch (error) {
                        console.error('Error searching players:', error);
                        this.searchResults = [];
                    }
                },

                async getPrediction() {
                    if (!this.predictionInput) return;
                    
                    try {
                        const response = await fetch(`/api/predict/${this.predictionInput}`);
                        this.predictionResult = await response.json();
                    } catch (error) {
                        console.error('Error getting prediction:', error);
                        this.predictionResult = { error: error.message };
                    }
                },

                selectPlayer(player) {
                    alert(`Player Details: ${player.name} (${player.position}) - ${player.team}`);
                }
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main NFL data browser page."""
    try:
        session = get_db_session()
        
        # Get database statistics
        total_players = session.query(Player).count()
        active_players = session.query(Player).filter(Player.is_active == True).count()
        total_games = session.query(Game).count()
        qb_count = session.query(Player).filter(Player.position == 'QB', Player.is_active == True).count()
        
        # Get teams
        teams_query = session.query(Player.current_team).filter(
            Player.current_team.isnot(None),
            Player.is_active == True
        ).distinct().all()
        teams = sorted([team[0] for team in teams_query if team[0]])
        
        # Get recent games
        recent_games = session.query(Game).order_by(Game.game_date.desc()).limit(20).all()
        games_data = []
        for game in recent_games:
            games_data.append({
                'game_id': game.game_id,
                'home_team': game.home_team,
                'away_team': game.away_team,
                'game_date': str(game.game_date),
                'week': game.week,
                'season': game.season,
                'home_score': game.home_score,
                'away_score': game.away_score
            })
        
        session.close()
        
        return render_template_string(MAIN_TEMPLATE,
            total_players=total_players,
            active_players=active_players,
            total_games=total_games,
            qb_count=qb_count,
            teams=teams,
            recent_games=games_data
        )
        
    except Exception as e:
        return f"Error loading data: {e}"

@app.route('/api/team/<team>/players')
def get_team_players(team):
    """Get players for a specific team."""
    try:
        session = get_db_session()
        players = session.query(Player).filter(
            Player.current_team == team,
            Player.is_active == True
        ).order_by(Player.position, Player.name).all()
        
        players_data = []
        for player in players:
            players_data.append({
                'player_id': player.player_id,
                'name': player.name,
                'position': player.position,
                'team': player.current_team
            })
        
        session.close()
        return jsonify(players_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/players/search')
def search_players():
    """Search players by name and/or position."""
    try:
        search_term = request.args.get('search', '')
        position = request.args.get('position', '')
        
        session = get_db_session()
        query = session.query(Player).filter(Player.is_active == True)
        
        if search_term:
            query = query.filter(Player.name.like(f'%{search_term}%'))
        if position:
            query = query.filter(Player.position == position)
        
        players = query.limit(50).all()
        
        players_data = []
        for player in players:
            players_data.append({
                'player_id': player.player_id,
                'name': player.name,
                'position': player.position,
                'team': player.current_team,
                'is_active': player.is_active
            })
        
        session.close()
        return jsonify(players_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<player_id>')
def predict_player(player_id):
    """Get prediction for a player."""
    try:
        if not models:
            return jsonify({'error': 'Models not available'}), 503
        
        result = models.predict_player(player_id)
        if result:
            return jsonify({
                'player_id': result.player_id,
                'player_name': result.player_name,
                'position': result.position,
                'predicted_value': round(result.predicted_value, 2),
                'confidence': round(result.confidence, 3),
                'model_used': result.model_used
            })
        else:
            return jsonify({'error': 'No prediction available'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🏈 Starting NFL Data Browser...")
    
    port = find_available_port(5000)
    if port is None:
        print("❌ No available ports found")
        exit(1)
    
    print(f"📊 NFL Data Browser: http://localhost:{port}")
    print(f"🔍 Browse teams, players, games, stats, and predictions")
    
    app.run(host='0.0.0.0', port=port, debug=True)
