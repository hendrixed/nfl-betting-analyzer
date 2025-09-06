#!/usr/bin/env python3
"""
Cross-Validation and Backtesting Framework
Comprehensive model validation and historical performance testing for NFL betting predictions.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import create_engine, text
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationBacktestingFramework:
    """Comprehensive validation and backtesting framework."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Validation configuration
        self.time_series_splits = 5
        self.min_train_size = 20
        self.test_size = 5
        
        # Backtesting configuration
        self.backtest_periods = ['2023', '2022', '2021']
        self.betting_bankroll = 10000  # Starting bankroll for backtesting
        self.max_bet_size = 0.05       # Max 5% of bankroll per bet
        
        # Performance tracking
        self.validation_results = {}
        self.backtest_results = {}
        
        # Initialize validation database
        self._init_validation_db()
        
    def _init_validation_db(self):
        """Initialize validation tracking database."""
        
        # Model validation results
        validation_table_sql = """
        CREATE TABLE IF NOT EXISTS model_validation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            player_id TEXT,
            target_variable TEXT,
            validation_method TEXT,
            cv_score REAL,
            rmse REAL,
            mae REAL,
            r2_score REAL,
            validation_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Backtesting results
        backtest_table_sql = """
        CREATE TABLE IF NOT EXISTS backtesting_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            backtest_period TEXT,
            total_bets INTEGER,
            winning_bets INTEGER,
            total_profit REAL,
            roi REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            win_rate REAL,
            avg_bet_size REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Individual bet results
        bet_results_sql = """
        CREATE TABLE IF NOT EXISTS individual_bet_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            backtest_id INTEGER,
            player_id TEXT,
            bet_type TEXT,
            line REAL,
            prediction REAL,
            actual_result REAL,
            bet_size REAL,
            profit_loss REAL,
            game_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(validation_table_sql))
            conn.execute(text(backtest_table_sql))
            conn.execute(text(bet_results_sql))
            conn.commit()
    
    def time_series_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                   model=None, player_id: str = None) -> Dict[str, float]:
        """Perform time series cross-validation."""
        
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Ensure we have enough data
        if len(X) < self.min_train_size + self.test_size:
            logger.warning(f"Insufficient data for cross-validation: {len(X)} samples")
            return {}
        
        # Time series split
        tscv = TimeSeriesSplit(
            n_splits=min(self.time_series_splits, len(X) // self.test_size - 1),
            test_size=self.test_size
        )
        
        cv_scores = []
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Handle missing values
            X_train_clean = X_train.fillna(X_train.median())
            X_test_clean = X_test.fillna(X_train.median())
            
            # Fit model
            model.fit(X_train_clean, y_train)
            
            # Predict
            y_pred = model.predict(X_test_clean)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
        
        results = {
            'cv_score_mean': np.mean(cv_scores),
            'cv_score_std': np.std(cv_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'n_splits': len(cv_scores)
        }
        
        # Store results
        self._store_validation_results(
            model_name=type(model).__name__,
            player_id=player_id,
            target_variable='fantasy_points_ppr',
            validation_method='time_series_cv',
            results=results
        )
        
        return results
    
    def learning_curve_analysis(self, X: pd.DataFrame, y: pd.Series, 
                               model=None) -> Dict[str, Any]:
        """Analyze learning curves to understand model performance vs training size."""
        
        if model is None:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Define training sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Calculate learning curve
        from sklearn.model_selection import learning_curve
        
        X_clean = X.fillna(X.median())
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_clean, y, 
            train_sizes=train_sizes,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
    
    def hyperparameter_validation(self, X: pd.DataFrame, y: pd.Series, 
                                 param_name: str, param_range: List) -> Dict[str, Any]:
        """Validate hyperparameter choices."""
        
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        X_clean = X.fillna(X.median())
        
        train_scores, val_scores = validation_curve(
            model, X_clean, y,
            param_name=param_name,
            param_range=param_range,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        
        return {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
    
    def backtest_betting_strategy(self, strategy_name: str, 
                                 predictions_df: pd.DataFrame,
                                 actual_results_df: pd.DataFrame) -> Dict[str, Any]:
        """Backtest a betting strategy with historical data."""
        
        # Initialize backtesting variables
        bankroll = self.betting_bankroll
        total_bets = 0
        winning_bets = 0
        bet_history = []
        bankroll_history = [bankroll]
        
        # Merge predictions with actual results
        merged_df = predictions_df.merge(
            actual_results_df, 
            on=['player_id', 'game_date'], 
            how='inner'
        )
        
        for _, row in merged_df.iterrows():
            prediction = row['prediction']
            actual = row['actual_result']
            line = row.get('line', prediction)
            confidence = row.get('confidence', 0.6)
            
            # Determine bet based on strategy
            bet_info = self._calculate_bet_size(prediction, line, confidence, bankroll)
            
            if bet_info['should_bet']:
                total_bets += 1
                bet_size = bet_info['bet_size']
                
                # Determine if bet won
                if bet_info['bet_type'] == 'over':
                    won = actual > line
                else:  # under
                    won = actual < line
                
                # Calculate profit/loss (assuming -110 odds)
                if won:
                    profit = bet_size * 0.909  # Win $0.909 for every $1 bet at -110
                    winning_bets += 1
                else:
                    profit = -bet_size
                
                bankroll += profit
                bankroll_history.append(bankroll)
                
                # Record bet
                bet_history.append({
                    'player_id': row['player_id'],
                    'bet_type': bet_info['bet_type'],
                    'line': line,
                    'prediction': prediction,
                    'actual_result': actual,
                    'bet_size': bet_size,
                    'profit_loss': profit,
                    'game_date': row['game_date'],
                    'won': won
                })
        
        # Calculate performance metrics
        if total_bets > 0:
            total_profit = bankroll - self.betting_bankroll
            roi = (total_profit / self.betting_bankroll) * 100
            win_rate = (winning_bets / total_bets) * 100
            avg_bet_size = np.mean([bet['bet_size'] for bet in bet_history])
            
            # Calculate max drawdown
            peak = self.betting_bankroll
            max_drawdown = 0
            for balance in bankroll_history:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio (simplified)
            returns = np.diff(bankroll_history) / bankroll_history[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            results = {
                'strategy_name': strategy_name,
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'total_profit': total_profit,
                'roi': roi,
                'win_rate': win_rate,
                'avg_bet_size': avg_bet_size,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'final_bankroll': bankroll,
                'bet_history': bet_history
            }
            
            # Store results
            self._store_backtest_results(results)
            
            return results
        
        return {'error': 'No bets placed during backtest period'}
    
    def _calculate_bet_size(self, prediction: float, line: float, 
                           confidence: float, bankroll: float) -> Dict[str, Any]:
        """Calculate bet size using Kelly Criterion with confidence adjustment."""
        
        # Determine bet direction
        if abs(prediction - line) < 0.5:  # Too close to bet
            return {'should_bet': False}
        
        bet_type = 'over' if prediction > line else 'under'
        
        # Calculate edge (simplified)
        edge = abs(prediction - line) / max(line, 1) * confidence
        
        # Minimum edge threshold
        if edge < 0.02:  # 2% minimum edge
            return {'should_bet': False}
        
        # Kelly Criterion calculation (simplified for -110 odds)
        # Kelly = (bp - q) / b, where b = 0.909, p = win probability, q = 1-p
        win_prob = 0.5 + edge  # Simplified probability calculation
        kelly_fraction = (0.909 * win_prob - (1 - win_prob)) / 0.909
        
        # Apply confidence and safety factors
        kelly_fraction *= confidence * 0.25  # Conservative Kelly
        
        # Calculate bet size
        bet_size = min(
            kelly_fraction * bankroll,
            self.max_bet_size * bankroll,
            bankroll * 0.02  # Never bet more than 2% of bankroll
        )
        
        return {
            'should_bet': bet_size > 0,
            'bet_size': max(0, bet_size),
            'bet_type': bet_type,
            'edge': edge,
            'kelly_fraction': kelly_fraction
        }
    
    def _store_validation_results(self, model_name: str, player_id: str, 
                                 target_variable: str, validation_method: str, 
                                 results: Dict[str, float]):
        """Store validation results in database."""
        
        insert_sql = """
        INSERT INTO model_validation_results 
        (model_name, player_id, target_variable, validation_method, 
         cv_score, rmse, mae, r2_score, validation_date)
        VALUES (:model_name, :player_id, :target_variable, :validation_method,
                :cv_score, :rmse, :mae, :r2_score, :validation_date)
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), {
                'model_name': model_name,
                'player_id': player_id,
                'target_variable': target_variable,
                'validation_method': validation_method,
                'cv_score': results.get('cv_score_mean', 0),
                'rmse': results.get('rmse_mean', 0),
                'mae': results.get('mae_mean', 0),
                'r2_score': results.get('r2_mean', 0),
                'validation_date': datetime.now()
            })
            conn.commit()
    
    def _store_backtest_results(self, results: Dict[str, Any]):
        """Store backtesting results in database."""
        
        insert_sql = """
        INSERT INTO backtesting_results 
        (strategy_name, total_bets, winning_bets, total_profit, roi, 
         max_drawdown, sharpe_ratio, win_rate, avg_bet_size)
        VALUES (:strategy_name, :total_bets, :winning_bets, :total_profit, :roi,
                :max_drawdown, :sharpe_ratio, :win_rate, :avg_bet_size)
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), {
                'strategy_name': results['strategy_name'],
                'total_bets': results['total_bets'],
                'winning_bets': results['winning_bets'],
                'total_profit': results['total_profit'],
                'roi': results['roi'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio'],
                'win_rate': results['win_rate'],
                'avg_bet_size': results['avg_bet_size']
            })
            conn.commit()
    
    def generate_mock_backtest_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate mock data for backtesting demonstration."""
        
        # Mock predictions
        dates = pd.date_range(start='2023-09-01', end='2023-12-31', freq='W')
        players = ['pmahomes_qb', 'cmccaffrey_rb', 'thill_wr', 'tkelce_te']
        
        predictions_data = []
        actual_data = []
        
        for date in dates:
            for player in players:
                # Generate mock prediction
                base_prediction = np.random.uniform(15, 25)
                prediction = base_prediction + np.random.normal(0, 2)
                confidence = np.random.uniform(0.6, 0.9)
                line = base_prediction + np.random.uniform(-2, 2)
                
                # Generate actual result (with some correlation to prediction)
                actual = prediction + np.random.normal(0, 3)
                
                predictions_data.append({
                    'player_id': player,
                    'game_date': date,
                    'prediction': prediction,
                    'confidence': confidence,
                    'line': line
                })
                
                actual_data.append({
                    'player_id': player,
                    'game_date': date,
                    'actual_result': actual
                })
        
        return pd.DataFrame(predictions_data), pd.DataFrame(actual_data)
    
    def run_comprehensive_validation(self, player_id: str) -> Dict[str, Any]:
        """Run comprehensive validation for a player."""
        
        # Get player data
        query = """
        SELECT * FROM player_game_stats 
        WHERE player_id = :player_id 
        ORDER BY created_at DESC
        LIMIT 30
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), {'player_id': player_id}).fetchall()
        
        if not results:
            return {'error': f'No data found for {player_id}'}
        
        # Get column names dynamically
        column_query = "PRAGMA table_info(player_game_stats)"
        with self.engine.connect() as conn:
            column_info = conn.execute(text(column_query)).fetchall()
        
        column_names = [col[1] for col in column_info]
        df = pd.DataFrame(results, columns=column_names)
        
        # Prepare features and target
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['id', 'fantasy_points_ppr']]
        
        if 'fantasy_points_ppr' not in df.columns or len(feature_columns) == 0:
            return {'error': 'Insufficient data for validation'}
        
        X = df[feature_columns]
        y = df['fantasy_points_ppr']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(y_clean) < 10:
            return {'error': 'Insufficient clean data for validation'}
        
        validation_results = {}
        
        # Time series cross-validation
        logger.info(f"Running time series cross-validation for {player_id}")
        cv_results = self.time_series_cross_validation(X_clean, y_clean, player_id=player_id)
        validation_results['cross_validation'] = cv_results
        
        # Learning curve analysis
        logger.info(f"Running learning curve analysis for {player_id}")
        lc_results = self.learning_curve_analysis(X_clean, y_clean)
        validation_results['learning_curve'] = lc_results
        
        # Hyperparameter validation
        logger.info(f"Running hyperparameter validation for {player_id}")
        hp_results = self.hyperparameter_validation(
            X_clean, y_clean, 
            'n_estimators', [10, 50, 100, 200]
        )
        validation_results['hyperparameter_validation'] = hp_results
        
        return validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        
        query = """
        SELECT model_name, validation_method, 
               AVG(cv_score) as avg_cv_score,
               AVG(rmse) as avg_rmse,
               AVG(r2_score) as avg_r2,
               COUNT(*) as num_validations
        FROM model_validation_results 
        GROUP BY model_name, validation_method
        ORDER BY avg_cv_score DESC
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query)).fetchall()
        
        validation_summary = []
        for row in results:
            validation_summary.append({
                'model_name': row[0],
                'validation_method': row[1],
                'avg_cv_score': row[2],
                'avg_rmse': row[3],
                'avg_r2': row[4],
                'num_validations': row[5]
            })
        
        # Get backtesting summary
        backtest_query = """
        SELECT strategy_name, 
               AVG(roi) as avg_roi,
               AVG(win_rate) as avg_win_rate,
               AVG(sharpe_ratio) as avg_sharpe,
               COUNT(*) as num_backtests
        FROM backtesting_results 
        GROUP BY strategy_name
        ORDER BY avg_roi DESC
        """
        
        with self.engine.connect() as conn:
            backtest_results = conn.execute(text(backtest_query)).fetchall()
        
        backtest_summary = []
        for row in backtest_results:
            backtest_summary.append({
                'strategy_name': row[0],
                'avg_roi': row[1],
                'avg_win_rate': row[2],
                'avg_sharpe': row[3],
                'num_backtests': row[4]
            })
        
        return {
            'validation_results': validation_summary,
            'backtest_results': backtest_summary,
            'total_validations': len(validation_summary),
            'total_backtests': len(backtest_summary)
        }

def main():
    """Test the validation and backtesting framework."""
    print("🔬 VALIDATION & BACKTESTING FRAMEWORK")
    print("=" * 70)
    
    # Initialize framework
    framework = ValidationBacktestingFramework()
    
    # Test comprehensive validation
    print("📊 Running Comprehensive Validation:")
    
    test_players = ['pmahomes_qb', 'cmccaffrey_rb']
    
    for player_id in test_players:
        print(f"\n   🧪 {player_id}:")
        
        validation_results = framework.run_comprehensive_validation(player_id)
        
        if 'error' in validation_results:
            print(f"     Error: {validation_results['error']}")
            continue
        
        # Cross-validation results
        if 'cross_validation' in validation_results:
            cv = validation_results['cross_validation']
            print(f"     Cross-Validation R²: {cv.get('cv_score_mean', 0):.3f} ± {cv.get('cv_score_std', 0):.3f}")
            print(f"     RMSE: {cv.get('rmse_mean', 0):.3f} ± {cv.get('rmse_std', 0):.3f}")
            print(f"     Splits: {cv.get('n_splits', 0)}")
        
        # Learning curve results
        if 'learning_curve' in validation_results:
            lc = validation_results['learning_curve']
            print(f"     Learning Curve: {len(lc.get('train_sizes', []))} training sizes analyzed")
        
        # Hyperparameter results
        if 'hyperparameter_validation' in validation_results:
            hp = validation_results['hyperparameter_validation']
            best_idx = np.argmax(hp.get('val_scores_mean', [0]))
            best_param = hp.get('param_range', [None])[best_idx] if best_idx < len(hp.get('param_range', [])) else None
            print(f"     Best {hp.get('param_name', 'parameter')}: {best_param}")
    
    # Test backtesting
    print("\n💰 Running Backtesting:")
    
    # Generate mock data
    predictions_df, actual_df = framework.generate_mock_backtest_data()
    print(f"   Generated {len(predictions_df)} predictions and {len(actual_df)} actual results")
    
    # Run backtest
    backtest_results = framework.backtest_betting_strategy(
        'Conservative Kelly', predictions_df, actual_df
    )
    
    if 'error' not in backtest_results:
        print(f"   Total Bets: {backtest_results['total_bets']}")
        print(f"   Win Rate: {backtest_results['win_rate']:.1f}%")
        print(f"   ROI: {backtest_results['roi']:.1f}%")
        print(f"   Max Drawdown: {backtest_results['max_drawdown']:.1f}%")
        print(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"   Final Bankroll: ${backtest_results['final_bankroll']:.2f}")
    else:
        print(f"   Backtest Error: {backtest_results['error']}")
    
    # Get validation summary
    print("\n📈 Validation Summary:")
    summary = framework.get_validation_summary()
    
    print(f"   Total Validations: {summary['total_validations']}")
    print(f"   Total Backtests: {summary['total_backtests']}")
    
    if summary['validation_results']:
        print("\n   🏆 Top Validation Results:")
        for result in summary['validation_results'][:3]:
            print(f"     {result['model_name']} ({result['validation_method']}): "
                  f"R² = {result['avg_cv_score']:.3f}")
    
    if summary['backtest_results']:
        print("\n   💎 Top Backtest Results:")
        for result in summary['backtest_results'][:3]:
            print(f"     {result['strategy_name']}: "
                  f"ROI = {result['avg_roi']:.1f}%, "
                  f"Win Rate = {result['avg_win_rate']:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ Validation and backtesting framework operational!")

if __name__ == "__main__":
    main()
