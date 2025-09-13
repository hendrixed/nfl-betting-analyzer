"""
Streamlined NFL Prediction Models
Simplified, working implementation for NFL betting predictions.
"""

import logging
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Make sklearn optional so that importing this module doesn't fail in lean environments
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = GradientBoostingRegressor = Ridge = None  # type: ignore
    train_test_split = cross_val_score = None  # type: ignore
    mean_squared_error = mean_absolute_error = r2_score = None  # type: ignore
    StandardScaler = None  # type: ignore

from sqlalchemy.orm import Session
from sqlalchemy import func
from ..database_models import get_db_session, Player, PlayerGameStats

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Model training and evaluation result"""
    position: str
    target_stat: str
    model_type: str
    r2_score: float
    rmse: float
    mae: float
    cv_score: float
    feature_importance: Dict[str, float]
    sample_count: int


@dataclass
class PredictionResult:
    """Player prediction result"""
    player_id: str
    player_name: str
    position: str
    predicted_value: float
    confidence: float
    model_used: str


class StreamlinedNFLModels:
    """Streamlined NFL prediction models that actually work"""
    
    def __init__(self, session: Session):
        self.session = session
        self.models_dir = Path("models/streamlined")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        if SKLEARN_AVAILABLE:
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0, random_state=42)
            }
        else:
            self.models = {}
    
    def train_all_models(self, target_stat: str = 'fantasy_points_ppr') -> Dict[str, List[ModelResult]]:
        """Train models for all positions"""
        logger.info(f"Training streamlined models for {target_stat}")
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available; skipping training.")
            return {}
        
        results = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            logger.info(f"Training models for {position}")
            
            try:
                position_results = self._train_position_models(position, target_stat)
                results[position] = position_results
                
                # Save the best model
                if position_results:
                    best_result = max(position_results, key=lambda x: x.r2_score)
                    self._save_model_result(best_result)
                    
            except Exception as e:
                logger.error(f"Error training {position} models: {e}")
                continue
        
        return results
    
    def _train_position_models(self, position: str, target_stat: str) -> List[ModelResult]:
        """Train all model types for a specific position"""
        if not SKLEARN_AVAILABLE:
            return []
        
        # Get training data
        X, y, feature_names, player_count = self._prepare_training_data(position, target_stat)
        
        if len(X) < 10:
            logger.warning(f"Insufficient data for {position}: {len(X)} samples")
            return []
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = []
        
        for model_name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                cv_score = np.mean(cv_scores)
                
                # Feature importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)
                else:
                    importance = np.ones(len(feature_names)) / len(feature_names)
                
                # Normalize importance
                if np.sum(importance) > 0:
                    importance = importance / np.sum(importance)
                
                feature_importance = dict(zip(feature_names, importance))
                
                result = ModelResult(
                    position=position,
                    target_stat=target_stat,
                    model_type=model_name,
                    r2_score=r2,
                    rmse=rmse,
                    mae=mae,
                    cv_score=cv_score,
                    feature_importance=feature_importance,
                    sample_count=len(X)
                )
                
                results.append(result)
                
                logger.info(f"  {model_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} for {position}: {e}")
                continue
        
        return results
    
    def _prepare_training_data(self, position: str, target_stat: str) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        """Prepare training data for a position"""
        
        # Get players with sufficient game history
        players_with_stats = self.session.query(
            Player.player_id,
            Player.name,
            func.count(PlayerGameStats.stat_id).label('game_count'),
            func.avg(PlayerGameStats.fantasy_points_ppr).label('avg_fantasy_points'),
            func.avg(PlayerGameStats.fantasy_points_standard).label('avg_fantasy_standard'),
            func.avg(PlayerGameStats.passing_yards).label('avg_passing_yards'),
            func.avg(PlayerGameStats.passing_touchdowns).label('avg_passing_tds'),
            func.avg(PlayerGameStats.passing_interceptions).label('avg_passing_ints'),
            func.avg(PlayerGameStats.rushing_yards).label('avg_rushing_yards'),
            func.avg(PlayerGameStats.rushing_touchdowns).label('avg_rushing_tds'),
            func.avg(PlayerGameStats.receiving_yards).label('avg_receiving_yards'),
            func.avg(PlayerGameStats.receiving_touchdowns).label('avg_receiving_tds'),
            func.avg(PlayerGameStats.receptions).label('avg_receptions'),
            func.avg(PlayerGameStats.targets).label('avg_targets'),
            func.max(PlayerGameStats.fantasy_points_ppr).label('fantasy_max'),
            func.min(PlayerGameStats.fantasy_points_ppr).label('fantasy_min')
        ).join(PlayerGameStats).filter(
            Player.position == position,
            Player.is_active == True,
            PlayerGameStats.fantasy_points_ppr.isnot(None)
        ).group_by(Player.player_id, Player.name).having(
            func.count(PlayerGameStats.stat_id) >= 5
        ).all()
        
        if not players_with_stats:
            return np.array([]), np.array([]), [], 0
        
        # Build feature matrix
        features = []
        targets = []
        
        for player_data in players_with_stats:
            # Position-specific features
            fantasy_range = (player_data.fantasy_max or 0) - (player_data.fantasy_min or 0)
            
            if position == 'QB':
                feature_row = [
                    player_data.avg_passing_yards or 0,
                    player_data.avg_passing_tds or 0,
                    player_data.avg_passing_ints or 0,
                    player_data.avg_rushing_yards or 0,
                    player_data.game_count,
                    fantasy_range
                ]
            elif position == 'RB':
                feature_row = [
                    player_data.avg_rushing_yards or 0,
                    player_data.avg_rushing_tds or 0,
                    player_data.avg_receiving_yards or 0,
                    player_data.avg_receptions or 0,
                    player_data.game_count,
                    fantasy_range
                ]
            elif position in ['WR', 'TE']:
                feature_row = [
                    player_data.avg_receiving_yards or 0,
                    player_data.avg_receiving_tds or 0,
                    player_data.avg_receptions or 0,
                    player_data.avg_targets or 0,
                    player_data.game_count,
                    fantasy_range
                ]
            else:
                continue
            
            # Target variable
            if target_stat == 'fantasy_points_ppr':
                target_value = player_data.avg_fantasy_points
            elif target_stat == 'fantasy_points_standard':
                target_value = player_data.avg_fantasy_standard
            elif target_stat == 'passing_yards':
                target_value = player_data.avg_passing_yards
            elif target_stat == 'rushing_yards':
                target_value = player_data.avg_rushing_yards
            elif target_stat == 'receiving_yards':
                target_value = player_data.avg_receiving_yards
            else:
                target_value = player_data.avg_fantasy_points
            
            if target_value and target_value > 0:
                features.append(feature_row)
                targets.append(target_value)
        
        # Feature names
        if position == 'QB':
            feature_names = ['avg_passing_yards', 'avg_passing_tds', 'avg_passing_ints', 
                           'avg_rushing_yards', 'game_count', 'fantasy_range']
        elif position == 'RB':
            feature_names = ['avg_rushing_yards', 'avg_rushing_tds', 'avg_receiving_yards',
                           'avg_receptions', 'game_count', 'fantasy_range']
        else:  # WR/TE
            feature_names = ['avg_receiving_yards', 'avg_receiving_tds', 'avg_receptions',
                           'avg_targets', 'game_count', 'fantasy_range']
        
        return np.array(features), np.array(targets), feature_names, len(players_with_stats)
    
    def predict_player(self, player_id: str, target_stat: str = 'fantasy_points_ppr') -> Optional[PredictionResult]:
        """Predict performance for a specific player"""
        if not SKLEARN_AVAILABLE:
            return None
        
        # Get player info
        player = self.session.query(Player).filter(Player.player_id == player_id).first()
        if not player:
            return None
        
        # Load model
        model_data = self._load_model(player.position, target_stat)
        if not model_data:
            return None
        
        try:
            # Get player's recent stats for prediction
            recent_stats = self.session.query(
                func.avg(PlayerGameStats.fantasy_points_ppr).label('avg_fantasy_points'),
                func.avg(PlayerGameStats.passing_yards).label('avg_passing_yards'),
                func.avg(PlayerGameStats.passing_touchdowns).label('avg_passing_tds'),
                func.avg(PlayerGameStats.passing_interceptions).label('avg_passing_ints'),
                func.avg(PlayerGameStats.rushing_yards).label('avg_rushing_yards'),
                func.avg(PlayerGameStats.rushing_touchdowns).label('avg_rushing_tds'),
                func.avg(PlayerGameStats.receiving_yards).label('avg_receiving_yards'),
                func.avg(PlayerGameStats.receiving_touchdowns).label('avg_receiving_tds'),
                func.avg(PlayerGameStats.receptions).label('avg_receptions'),
                func.avg(PlayerGameStats.targets).label('avg_targets'),
                func.count(PlayerGameStats.stat_id).label('game_count'),
                func.max(PlayerGameStats.fantasy_points_ppr).label('fantasy_max'),
                func.min(PlayerGameStats.fantasy_points_ppr).label('fantasy_min')
            ).filter(
                PlayerGameStats.player_id == player_id
            ).first()
            
            if not recent_stats or not recent_stats.game_count:
                return None
            
            # Build feature vector
            fantasy_range = (recent_stats.fantasy_max or 0) - (recent_stats.fantasy_min or 0)
            
            if player.position == 'QB':
                features = [
                    recent_stats.avg_passing_yards or 0,
                    recent_stats.avg_passing_tds or 0,
                    recent_stats.avg_passing_ints or 0,
                    recent_stats.avg_rushing_yards or 0,
                    recent_stats.game_count,
                    fantasy_range
                ]
            elif player.position == 'RB':
                features = [
                    recent_stats.avg_rushing_yards or 0,
                    recent_stats.avg_rushing_tds or 0,
                    recent_stats.avg_receiving_yards or 0,
                    recent_stats.avg_receptions or 0,
                    recent_stats.game_count,
                    fantasy_range
                ]
            elif player.position in ['WR', 'TE']:
                features = [
                    recent_stats.avg_receiving_yards or 0,
                    recent_stats.avg_receiving_tds or 0,
                    recent_stats.avg_receptions or 0,
                    recent_stats.avg_targets or 0,
                    recent_stats.game_count,
                    fantasy_range
                ]
            else:
                return None
            
            # Make prediction
            feature_array = np.array(features).reshape(1, -1)
            feature_scaled = model_data['scaler'].transform(feature_array)
            prediction = model_data['model'].predict(feature_scaled)[0]
            
            # Clamp prediction for known targets
            try:
                if target_stat == 'fantasy_points_ppr':
                    # Fantasy outputs should be within a sensible range
                    prediction = float(max(0.0, min(80.0, float(prediction))))
                else:
                    # Generic safety clamp for other numeric targets
                    prediction = float(prediction)
            except Exception:
                pass
            
            # Calculate confidence based on model performance and clamp to [0, 0.99]
            try:
                confidence = float(model_data['r2_score']) if model_data.get('r2_score') is not None else 0.0
            except Exception:
                confidence = 0.0
            confidence = float(max(0.0, min(0.99, confidence)))
            
            return PredictionResult(
                player_id=player_id,
                player_name=player.name,
                position=player.position,
                predicted_value=prediction,
                confidence=confidence,
                model_used=f"{player.position}_{model_data['model_type']}"
            )
            
        except Exception as e:
            logger.error(f"Error predicting for {player.name}: {e}")
            return None
    
    def _save_model_result(self, result: ModelResult):
        """Save the best model for a position"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Retrain the model with all data for saving
        X, y, feature_names, _ = self._prepare_training_data(result.position, result.target_stat)
        
        if len(X) < 5:
            return
        
        # Scale and train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = self.models[result.model_type]
        model.fit(X_scaled, y)
        
        # Save model data
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'model_type': result.model_type,
            'r2_score': result.r2_score,
            'position': result.position,
            'target_stat': result.target_stat
        }
        
        filename = f"{result.position}_{result.target_stat}_{result.model_type}.pkl"
        filepath = self.models_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved model: {filename}")
            # Also write sidecar JSON metadata to avoid requiring sklearn for summaries
            meta = {
                'position': result.position,
                'model_type': result.model_type,
                'target_stat': result.target_stat,
                'r2_score': result.r2_score,
                'features': len(feature_names),
                'feature_names': feature_names,
                'samples': len(X),
                'saved_at': datetime.now().isoformat()
            }
            meta_path = filepath.with_suffix(filepath.suffix + ".meta.json")
            try:
                with open(meta_path, 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, ensure_ascii=True, indent=2)
            except Exception as me:
                logger.debug(f"Failed to write metadata for {filename}: {me}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self, position: str, target_stat: str) -> Optional[Dict]:
        """Load the best model for a position"""
        if not SKLEARN_AVAILABLE:
            return None
        
        # Try to find the best model file
        best_model = None
        best_r2 = -1
        
        for model_file in self.models_dir.glob(f"{position}_{target_stat}_*.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                if model_data.get('r2_score', 0) > best_r2:
                    best_model = model_data
                    best_r2 = model_data.get('r2_score', 0)
                
            except Exception as e:
                logger.debug(f"Error loading {model_file}: {e}")
                continue
        
        return best_model
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models by position"""
        summary: Dict[str, Any] = {}
        
        # 1) Prefer reading sidecar JSON metadata if available (does not require sklearn)
        for meta_file in self.models_dir.glob("*.meta.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
                position = meta.get('position', 'Unknown')
                current_best = summary.get(position)
                if (current_best is None) or (float(meta.get('r2_score', 0)) > float(current_best.get('r2_score', 0))):
                    summary[position] = {
                        'model_type': meta.get('model_type', 'Unknown'),
                        'r2_score': float(meta.get('r2_score', 0)),
                        'target_stat': meta.get('target_stat', 'fantasy_points_ppr'),
                        'features': int(meta.get('features', 0))
                    }
            except Exception as e:
                logger.debug(f"Error reading metadata {meta_file}: {e}")
                continue
        
        # 2) If no metadata, provide minimal summary by parsing filenames
        if not summary:
            for model_file in self.models_dir.glob("*.pkl"):
                try:
                    name = model_file.stem  # e.g., QB_fantasy_points_ppr_ridge
                    parts = name.split("_")
                    if len(parts) >= 3:
                        position = parts[0]
                        model_type = parts[-1]
                        target_stat = "_".join(parts[1:-1])
                        # Minimal info; r2 unknown without unpickling
                        existing = summary.get(position)
                        if existing is None:
                            summary[position] = {
                                'model_type': model_type,
                                'r2_score': 0.0,
                                'target_stat': target_stat,
                                'features': 0
                            }
                    else:
                        continue
                except Exception:
                    continue
        
        # 3) If sklearn is available, try to refine with actual r2/features by unpickling
        if SKLEARN_AVAILABLE:
            for model_file in self.models_dir.glob("*.pkl"):
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    position = model_data.get('position', 'Unknown')
                    if position not in summary or float(model_data.get('r2_score', 0)) > float(summary[position]['r2_score']):
                        summary[position] = {
                            'model_type': model_data.get('model_type', 'Unknown'),
                            'r2_score': float(model_data.get('r2_score', 0)),
                            'target_stat': model_data.get('target_stat', 'fantasy_points_ppr'),
                            'features': len(model_data.get('feature_names', []))
                        }
                except Exception as e:
                    logger.debug(f"Error refining summary from {model_file}: {e}")
                    continue
        
        return summary


def main():
    """Test the streamlined models"""
    session = get_db_session()
    models = StreamlinedNFLModels(session)
    
    try:
        # Train models
        results = models.train_all_models('fantasy_points_ppr')
        
        print("Training Results:")
        for position, position_results in results.items():
            print(f"\n{position}:")
            for result in position_results:
                print(f"  {result.model_type}: R² = {result.r2_score:.3f}, "
                      f"RMSE = {result.rmse:.3f}, Samples = {result.sample_count}")
        
        # Test prediction
        test_player = session.query(Player).filter(
            Player.is_active == True,
            Player.position == 'QB'
        ).first()
        
        if test_player:
            prediction = models.predict_player(test_player.player_id)
            if prediction:
                print(f"\nTest Prediction:")
                print(f"  Player: {prediction.player_name}")
                print(f"  Predicted: {prediction.predicted_value:.2f}")
                print(f"  Confidence: {prediction.confidence:.2f}")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
