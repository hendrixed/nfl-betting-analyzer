#!/usr/bin/env python3
"""
Advanced Feature Engineering System
Sophisticated feature creation, selection, and transformation for NFL betting models.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import create_engine, text
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """Advanced feature engineering for NFL betting predictions."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Feature engineering configuration
        self.rolling_windows = [3, 5, 8, 12]  # Game windows for rolling stats
        self.lag_features = [1, 2, 3, 5]      # Lag periods for historical features
        
        # Feature transformations
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        
        # Feature importance tracking
        self.feature_importance_scores = {}
        
    def create_rolling_features(self, df: pd.DataFrame, 
                               target_columns: List[str]) -> pd.DataFrame:
        """Create rolling statistical features."""
        
        df_enhanced = df.copy()
        
        # Sort by player and date for proper rolling calculations
        df_enhanced = df_enhanced.sort_values(['player_id', 'created_at'])
        
        for window in self.rolling_windows:
            for col in target_columns:
                if col in df_enhanced.columns:
                    # Rolling mean
                    df_enhanced[f'{col}_rolling_mean_{window}'] = (
                        df_enhanced.groupby('player_id')[col]
                        .rolling(window=window, min_periods=1)
                        .mean().reset_index(0, drop=True)
                    )
                    
                    # Rolling standard deviation
                    df_enhanced[f'{col}_rolling_std_{window}'] = (
                        df_enhanced.groupby('player_id')[col]
                        .rolling(window=window, min_periods=1)
                        .std().reset_index(0, drop=True)
                    )
                    
                    # Rolling max
                    df_enhanced[f'{col}_rolling_max_{window}'] = (
                        df_enhanced.groupby('player_id')[col]
                        .rolling(window=window, min_periods=1)
                        .max().reset_index(0, drop=True)
                    )
                    
                    # Rolling trend (slope)
                    df_enhanced[f'{col}_rolling_trend_{window}'] = (
                        df_enhanced.groupby('player_id')[col]
                        .rolling(window=window, min_periods=2)
                        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                        .reset_index(0, drop=True)
                    )
        
        return df_enhanced
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_columns: List[str]) -> pd.DataFrame:
        """Create lagged features for time series analysis."""
        
        df_enhanced = df.copy()
        df_enhanced = df_enhanced.sort_values(['player_id', 'created_at'])
        
        for lag in self.lag_features:
            for col in target_columns:
                if col in df_enhanced.columns:
                    # Simple lag
                    df_enhanced[f'{col}_lag_{lag}'] = (
                        df_enhanced.groupby('player_id')[col].shift(lag)
                    )
                    
                    # Lag difference (change from lag periods ago)
                    df_enhanced[f'{col}_lag_diff_{lag}'] = (
                        df_enhanced[col] - df_enhanced[f'{col}_lag_{lag}']
                    )
                    
                    # Lag ratio (current / lag)
                    df_enhanced[f'{col}_lag_ratio_{lag}'] = (
                        df_enhanced[col] / (df_enhanced[f'{col}_lag_{lag}'] + 0.1)
                    )
        
        return df_enhanced
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        
        df_enhanced = df.copy()
        
        # Position-specific interactions
        if 'passing_yards' in df_enhanced.columns and 'passing_attempts' in df_enhanced.columns:
            df_enhanced['yards_per_attempt'] = (
                df_enhanced['passing_yards'] / (df_enhanced['passing_attempts'] + 0.1)
            )
        
        if 'rushing_yards' in df_enhanced.columns and 'rushing_attempts' in df_enhanced.columns:
            df_enhanced['yards_per_carry'] = (
                df_enhanced['rushing_yards'] / (df_enhanced['rushing_attempts'] + 0.1)
            )
        
        if 'receiving_yards' in df_enhanced.columns and 'receptions' in df_enhanced.columns:
            df_enhanced['yards_per_reception'] = (
                df_enhanced['receiving_yards'] / (df_enhanced['receptions'] + 0.1)
            )
        
        if 'receptions' in df_enhanced.columns and 'targets' in df_enhanced.columns:
            df_enhanced['catch_rate'] = (
                df_enhanced['receptions'] / (df_enhanced['targets'] + 0.1)
            )
        
        # Efficiency ratios
        if 'passing_touchdowns' in df_enhanced.columns and 'passing_attempts' in df_enhanced.columns:
            df_enhanced['td_rate'] = (
                df_enhanced['passing_touchdowns'] / (df_enhanced['passing_attempts'] + 0.1)
            )
        
        # Game script indicators
        if 'passing_attempts' in df_enhanced.columns and 'rushing_attempts' in df_enhanced.columns:
            df_enhanced['pass_rush_ratio'] = (
                df_enhanced['passing_attempts'] / 
                (df_enhanced['passing_attempts'] + df_enhanced['rushing_attempts'] + 0.1)
            )
        
        return df_enhanced
    
    def create_opponent_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent-adjusted performance features."""
        
        df_enhanced = df.copy()
        
        # Calculate league averages for normalization
        numeric_columns = df_enhanced.select_dtypes(include=[np.number]).columns
        league_averages = df_enhanced[numeric_columns].mean()
        
        # Create opponent-adjusted features
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 
                   'passing_touchdowns', 'rushing_touchdowns', 'receiving_touchdowns']:
            if col in df_enhanced.columns:
                # Performance vs league average
                df_enhanced[f'{col}_vs_avg'] = (
                    df_enhanced[col] - league_averages.get(col, 0)
                )
                
                # Normalized performance
                league_std = df_enhanced[col].std()
                if league_std > 0:
                    df_enhanced[f'{col}_normalized'] = (
                        (df_enhanced[col] - league_averages.get(col, 0)) / league_std
                    )
        
        return df_enhanced
    
    def create_streaks_and_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create streak and momentum features."""
        
        df_enhanced = df.copy()
        df_enhanced = df_enhanced.sort_values(['player_id', 'created_at'])
        
        # Fantasy points streaks
        if 'fantasy_points_ppr' in df_enhanced.columns:
            # Above/below average streaks
            avg_points = df_enhanced['fantasy_points_ppr'].mean()
            df_enhanced['above_avg'] = (df_enhanced['fantasy_points_ppr'] > avg_points).astype(int)
            
            # Calculate streaks
            df_enhanced['streak'] = (
                df_enhanced.groupby('player_id')['above_avg']
                .apply(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
                .reset_index(0, drop=True)
            )
            
            # Momentum (recent performance trend)
            df_enhanced['momentum_3'] = (
                df_enhanced.groupby('player_id')['fantasy_points_ppr']
                .rolling(window=3, min_periods=1)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                .reset_index(0, drop=True)
            )
        
        return df_enhanced
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal and temporal features."""
        
        df_enhanced = df.copy()
        
        # Convert created_at to datetime if it's not already
        if 'created_at' in df_enhanced.columns:
            df_enhanced['created_at'] = pd.to_datetime(df_enhanced['created_at'])
            
            # Week of season (use week column if available, otherwise derive from date)
            if 'week' in df_enhanced.columns:
                df_enhanced['week_of_season'] = df_enhanced['week']
            else:
                # Derive week from date (approximate)
                df_enhanced['week_of_season'] = df_enhanced['created_at'].dt.isocalendar().week - 35
                df_enhanced['week_of_season'] = df_enhanced['week_of_season'].clip(1, 18)
            
            # Early/mid/late season indicators
            week_col = 'week_of_season'
            df_enhanced['early_season'] = (df_enhanced[week_col] <= 6).astype(int)
            df_enhanced['mid_season'] = ((df_enhanced[week_col] > 6) & (df_enhanced[week_col] <= 12)).astype(int)
            df_enhanced['late_season'] = (df_enhanced[week_col] > 12).astype(int)
            
            # Playoff implications (weeks 15-18)
            df_enhanced['playoff_weeks'] = (df_enhanced[week_col] >= 15).astype(int)
        
        return df_enhanced
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """Select the most important features."""
        
        # Remove non-numeric columns and handle missing values
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        if len(X_numeric.columns) == 0:
            return X, []
        
        # Feature selection
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(X_numeric.columns)))
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X_numeric.columns)))
        elif method == 'random_forest':
            # Use Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_numeric, y)
            
            # Get feature importance scores
            importance_scores = pd.Series(rf.feature_importances_, index=X_numeric.columns)
            selected_features = importance_scores.nlargest(min(k, len(X_numeric.columns))).index.tolist()
            
            return X_numeric[selected_features], selected_features
        else:
            return X_numeric, X_numeric.columns.tolist()
        
        # Fit selector and transform
        X_selected = selector.fit_transform(X_numeric, y)
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        
        # Store feature importance scores
        if hasattr(selector, 'scores_'):
            self.feature_importance_scores[method] = dict(zip(X_numeric.columns, selector.scores_))
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X_numeric.index), selected_features
    
    def transform_features(self, X: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        """Apply feature transformations."""
        
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) == 0:
            return X
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(n_quantiles=100, random_state=42)
        else:
            return X_numeric
        
        # Fit and transform
        X_scaled = scaler.fit_transform(X_numeric.fillna(X_numeric.median()))
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        return pd.DataFrame(X_scaled, columns=X_numeric.columns, index=X_numeric.index)
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2, 
                                  max_features: int = 20) -> pd.DataFrame:
        """Create polynomial features for top features."""
        
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) == 0 or degree < 2:
            return X_numeric
        
        # Select top features to avoid explosion
        top_features = X_numeric.columns[:min(max_features, len(X_numeric.columns))]
        X_top = X_numeric[top_features]
        
        # Create polynomial features
        poly_features = pd.DataFrame(index=X_top.index)
        
        # Degree 2 interactions
        if degree >= 2:
            for i, col1 in enumerate(top_features):
                for col2 in top_features[i:]:
                    if col1 != col2:
                        poly_features[f'{col1}_x_{col2}'] = X_top[col1] * X_top[col2]
                    else:
                        poly_features[f'{col1}_squared'] = X_top[col1] ** 2
        
        # Combine with original features
        return pd.concat([X_numeric, poly_features], axis=1)
    
    def apply_pca(self, X: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction."""
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_filled = X_numeric.fillna(X_numeric.median())
        
        if len(X_filled.columns) < n_components:
            return X_numeric
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_filled)
        
        # Store PCA transformer
        self.pca_transformers[n_components] = pca
        
        # Create DataFrame with PCA components
        pca_columns = [f'pca_component_{i+1}' for i in range(n_components)]
        return pd.DataFrame(X_pca, columns=pca_columns, index=X_numeric.index)
    
    def engineer_comprehensive_features(self, player_id: str, 
                                      target: str = 'fantasy_points_ppr') -> pd.DataFrame:
        """Create comprehensive feature set for a player."""
        
        # Get player data
        query = """
        SELECT * FROM player_game_stats 
        WHERE player_id = :player_id 
        ORDER BY created_at DESC
        LIMIT 50
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), {'player_id': player_id}).fetchall()
        
        if not results:
            return pd.DataFrame()
        
        # Get column names dynamically
        column_query = "PRAGMA table_info(player_game_stats)"
        with self.engine.connect() as conn:
            column_info = conn.execute(text(column_query)).fetchall()
        
        column_names = [col[1] for col in column_info]
        df = pd.DataFrame(results, columns=column_names)
        
        # Define target columns for feature engineering
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = [col for col in numeric_columns if col not in ['id', 'week', 'season']]
        
        logger.info(f"Creating comprehensive features for {player_id}")
        
        # Apply all feature engineering techniques
        df_enhanced = df.copy()
        
        # 1. Rolling features
        df_enhanced = self.create_rolling_features(df_enhanced, target_columns)
        
        # 2. Lag features
        df_enhanced = self.create_lag_features(df_enhanced, target_columns)
        
        # 3. Interaction features
        df_enhanced = self.create_interaction_features(df_enhanced)
        
        # 4. Opponent-adjusted features
        df_enhanced = self.create_opponent_adjusted_features(df_enhanced)
        
        # 5. Streaks and momentum
        df_enhanced = self.create_streaks_and_momentum(df_enhanced)
        
        # 6. Seasonal features
        df_enhanced = self.create_seasonal_features(df_enhanced)
        
        logger.info(f"Created {len(df_enhanced.columns)} total features")
        
        return df_enhanced
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """Generate feature importance report."""
        
        report = {
            'feature_selection_methods': list(self.feature_importance_scores.keys()),
            'total_scalers_fitted': len(self.scalers),
            'pca_transformers': list(self.pca_transformers.keys()),
            'feature_importance_scores': self.feature_importance_scores
        }
        
        # Get top features across methods
        if self.feature_importance_scores:
            all_features = set()
            for method_scores in self.feature_importance_scores.values():
                all_features.update(method_scores.keys())
            
            # Calculate average importance across methods
            avg_importance = {}
            for feature in all_features:
                scores = []
                for method_scores in self.feature_importance_scores.values():
                    if feature in method_scores:
                        scores.append(method_scores[feature])
                
                if scores:
                    avg_importance[feature] = np.mean(scores)
            
            # Get top 20 features
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            report['top_features'] = top_features
        
        return report

def main():
    """Test the advanced feature engineering system."""
    print("🔧 ADVANCED FEATURE ENGINEERING SYSTEM")
    print("=" * 70)
    
    # Initialize system
    feature_eng = AdvancedFeatureEngineering()
    
    # Test comprehensive feature engineering
    print("🛠️ Testing Comprehensive Feature Engineering:")
    
    test_players = ['pmahomes_qb', 'cmccaffrey_rb', 'thill_wr', 'tkelce_te']
    
    for player_id in test_players:
        print(f"\n   📊 {player_id}:")
        
        # Create comprehensive features
        features_df = feature_eng.engineer_comprehensive_features(player_id)
        
        if not features_df.empty:
            print(f"     Original columns: {len([col for col in features_df.columns if not any(x in col for x in ['rolling', 'lag', 'normalized', 'streak'])])}")
            print(f"     Total features: {len(features_df.columns)}")
            print(f"     Rolling features: {len([col for col in features_df.columns if 'rolling' in col])}")
            print(f"     Lag features: {len([col for col in features_df.columns if 'lag' in col])}")
            print(f"     Interaction features: {len([col for col in features_df.columns if any(x in col for x in ['_x_', 'rate', 'ratio', 'per_'])])}")
            
            # Test feature selection
            if 'fantasy_points_ppr' in features_df.columns:
                X = features_df.drop(['fantasy_points_ppr', 'player_id', 'team', 'opponent'], axis=1, errors='ignore')
                y = features_df['fantasy_points_ppr']
                
                # Remove rows with NaN target
                mask = ~y.isna()
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(y_clean) > 5:
                    # Test different feature selection methods
                    for method in ['mutual_info', 'f_regression', 'random_forest']:
                        try:
                            X_selected, selected_features = feature_eng.select_features(
                                X_clean, y_clean, method=method, k=20
                            )
                            print(f"     {method}: {len(selected_features)} features selected")
                        except Exception as e:
                            print(f"     {method}: Error - {str(e)[:50]}...")
        else:
            print(f"     No data available")
    
    # Generate feature importance report
    print("\n📈 Feature Importance Report:")
    report = feature_eng.get_feature_importance_report()
    
    print(f"   Methods used: {len(report['feature_selection_methods'])}")
    print(f"   Scalers fitted: {report['total_scalers_fitted']}")
    
    if 'top_features' in report:
        print(f"\n   🏆 Top 10 Features (Average Importance):")
        for i, (feature, importance) in enumerate(report['top_features'][:10], 1):
            print(f"     {i:2d}. {feature[:40]:<40} {importance:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Advanced feature engineering system operational!")

if __name__ == "__main__":
    main()
