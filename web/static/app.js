// NFL Betting Analyzer - Web Interface JavaScript

class NFLDashboard {
    constructor() {
        this.currentTab = 'teams';
        this.data = {
            teams: [],
            players: [],
            games: [],
            stats: []
        };
        this.init();
    }

    init() {
        this.setupTabs();
        this.loadInitialData();
    }

    setupTabs() {
        const tabs = document.querySelectorAll('.nav-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }

    switchTab(tabName) {
        // Update active tab
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update active content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-content`).classList.add('active');

        this.currentTab = tabName;
        this.loadTabData(tabName);
    }

    async loadInitialData() {
        this.loadTabData('teams');
    }

    async loadTabData(tabName) {
        const contentEl = document.getElementById(`${tabName}-content`);
        contentEl.innerHTML = '<div class="loading">Loading...</div>';

        try {
            let data;
            switch(tabName) {
                case 'teams':
                    data = await this.fetchTeams();
                    this.renderTeams(data.teams || []);
                    break;
                case 'players':
                    data = await this.fetchPlayers();
                    this.renderPlayers(data.players || []);
                    break;
                case 'games':
                    data = await this.fetchGames();
                    this.renderGames(data.games || []);
                    break;
                case 'stats':
                    data = await this.fetchStats();
                    this.renderStats(data.stats || []);
                    break;
            }
        } catch (error) {
            contentEl.innerHTML = `<div class="error">Error loading ${tabName}: ${error.message}</div>`;
        }
    }

    async fetchTeams() {
        const response = await fetch('/api/teams');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    }

    async fetchPlayers(team = null, position = null) {
        let url = '/api/players?limit=50';
        if (team) url += `&team=${team}`;
        if (position) url += `&position=${position}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    }

    async fetchGames() {
        const response = await fetch('/api/games?limit=20');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    }

    async fetchStats() {
        const response = await fetch('/api/stats?limit=50');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    }

    renderTeams(teams) {
        const contentEl = document.getElementById('teams-content');
        if (!teams.length) {
            contentEl.innerHTML = '<div class="error">No teams found</div>';
            return;
        }

        const html = `
            <div class="data-grid">
                <div class="card">
                    <h3>NFL Teams (${teams.length})</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Team</th>
                                <th>Abbreviation</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${teams.map(team => `
                                <tr>
                                    <td>${team.name}</td>
                                    <td>${team.abbreviation}</td>
                                    <td>
                                        <button class="btn" onclick="dashboard.loadTeamRoster('${team.id}')">
                                            View Roster
                                        </button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        contentEl.innerHTML = html;
    }

    renderPlayers(players) {
        const contentEl = document.getElementById('players-content');
        if (!players.length) {
            contentEl.innerHTML = '<div class="error">No players found</div>';
            return;
        }

        const html = `
            <div class="data-grid">
                <div class="card">
                    <h3>NFL Players (${players.length})</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Position</th>
                                <th>Team</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${players.map(player => `
                                <tr>
                                    <td>${player.name}</td>
                                    <td>${player.position}</td>
                                    <td>${player.team}</td>
                                    <td>${player.status}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        contentEl.innerHTML = html;
    }

    renderGames(games) {
        const contentEl = document.getElementById('games-content');
        if (!games.length) {
            contentEl.innerHTML = '<div class="error">No games found</div>';
            return;
        }

        const html = `
            <div class="data-grid">
                <div class="card">
                    <h3>NFL Games (${games.length})</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Away Team</th>
                                <th>Home Team</th>
                                <th>Week</th>
                                <th>Season</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${games.map(game => `
                                <tr>
                                    <td>${game.game_date ? new Date(game.game_date).toLocaleDateString() : 'TBD'}</td>
                                    <td>${game.away_team}</td>
                                    <td>${game.home_team}</td>
                                    <td>${game.week}</td>
                                    <td>${game.season}</td>
                                    <td>${game.away_score !== null && game.home_score !== null ? 
                                        `${game.away_score} - ${game.home_score}` : 'Not played'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        contentEl.innerHTML = html;
    }

    renderStats(stats) {
        const contentEl = document.getElementById('stats-content');
        if (!stats.length) {
            contentEl.innerHTML = '<div class="error">No stats found</div>';
            return;
        }

        const html = `
            <div class="data-grid">
                <div class="card">
                    <h3>Player Statistics (${stats.length})</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Player ID</th>
                                <th>Pass Yds</th>
                                <th>Pass TDs</th>
                                <th>Rush Yds</th>
                                <th>Rush TDs</th>
                                <th>Rec Yds</th>
                                <th>Rec TDs</th>
                                <th>Fantasy Pts</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${stats.map(stat => `
                                <tr>
                                    <td>${stat.game_date ? new Date(stat.game_date).toLocaleDateString() : 'N/A'}</td>
                                    <td>${stat.player_id}</td>
                                    <td>${stat.passing_yards || 0}</td>
                                    <td>${stat.passing_tds || 0}</td>
                                    <td>${stat.rushing_yards || 0}</td>
                                    <td>${stat.rushing_tds || 0}</td>
                                    <td>${stat.receiving_yards || 0}</td>
                                    <td>${stat.receiving_tds || 0}</td>
                                    <td>${stat.fantasy_points_ppr ? stat.fantasy_points_ppr.toFixed(1) : '0.0'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        contentEl.innerHTML = html;
    }

    async loadTeamRoster(teamId) {
        try {
            const response = await fetch(`/api/teams/${teamId}/roster`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            
            // Switch to players tab and show roster
            this.switchTab('players');
            this.renderPlayers(data.roster || []);
        } catch (error) {
            console.error('Error loading team roster:', error);
        }
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new NFLDashboard();
});
