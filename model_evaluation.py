"""
NFL Model Evaluation & Backtesting System
Comprehensive evaluation framework for validating prediction accuracy and performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy import stats
from sqlalchemy import create_engine, select, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session

# Import our models
from database_models import (
    Player, Game, PlayerGameStats, PlayerPrediction, 
    GamePrediction, ModelPerformance, BettingLine
)

# Configure logging
logging.basicConfig(level=logging.INFO)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    database_url: str
    evaluation_start_date: date
    evaluation_end_date: date
    positions: List[str] = field(default_factory=lambda: ['QB', 'RB', 'WR', 'TE'])
    stat_types: List[str] = field(default_factory=lambda: [
        'passing_yards', 'rushing_yards', 'receiving_yards', 
        'passing_tds', 'rushing_tds', 'receiving_tds', 'receptions', 'fantasy_points'
    ])
    confidence_threshold: float = 0.6
    plot_results: bool = True
    save_results: bool = True
    results_directory: str = "evaluation_results"


@dataclass
class PredictionResult:
    """Container for prediction vs actual result."""
    player_id: str
    game_id: str
    position: str
    stat_type: str
    predicted_value: float
    actual_value: float
    confidence: float
    game_date: date
    model_version: str
    absolute_error: float = 0.0
    percentage_error: float = 0.0
    
    def __post_init__(self):
        self.absolute_error = abs(self.predicted_value - self.actual_value)
        if self.actual_value != 0:
            self.percentage_error = abs(self.absolute_error / self.actual_value) * 100
        else:
            self.percentage_error = 0.0 if self.predicted_value == 0 else 100.0


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    stat_type: str
    position: str
    n_predictions: int
    
    # Regression metrics
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    
    # Betting metrics
    over_under_accuracy: float = 0.0
    betting_roi: float = 0.0
    sharp_ratio: float = 0.0
    
    # Confidence calibration
    confidence_vs_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Additional metrics
    predictions_within_10_percent: float = 0.0
    predictions_within_20_percent: float = 0.0
    median_error: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0


class NFLModelEvaluator:
    """Comprehensive model evaluation and backtesting system."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the model evaluator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.engine = create_engine(config.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Results storage
        self.prediction_results: List[PredictionResult] = []
        self.evaluation_metrics: Dict[str, EvaluationMetrics] = {}
        
        # Create results directory
        Path(config.results_directory).mkdir(parents=True, exist_ok=True)
        
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run complete model evaluation."""
        try:
            self.logger.info("🔍 Starting comprehensive model evaluation...")
            
            # Load prediction and actual data
            self._load_evaluation_data()
            
            # Calculate metrics for each position and stat type
            self._calculate_evaluation_metrics()
            
            # Evaluate betting performance
            betting_results = self._evaluate_betting_performance()
            
            # Analyze model calibration
            calibration_results = self._analyze_model_calibration()
            
            # Generate performance insights
            insights = self._generate_performance_insights()
            
            # Create visualizations
            if self.config.plot_results:
                self._create_evaluation_plots()
                
            # Save results
            if self.config.save_results:
                results = self._save_evaluation_results(betting_results, calibration_results, insights)
            else:
                results = {
                    'metrics': self.evaluation_metrics,
                    'betting_performance': betting_results,
                    'calibration': calibration_results,
                    'insights': insights
                }
                
            self.logger.info("✅ Model evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            raise
            
    def _load_evaluation_data(self):
        """Load prediction and actual data for evaluation."""
        self.logger.info("📊 Loading evaluation data...")
        
        with self.Session() as session:
            # Query predictions and actuals
            query = session.query(
                PlayerPrediction,
                PlayerGameStats,
                Player.position,
                Game.game_date
            ).join(
                PlayerGameStats,
                and_(
                    PlayerPrediction.player_id == PlayerGameStats.player_id,
                    PlayerPrediction.game_id == PlayerGameStats.game_id
                )
            ).join(
                Player, PlayerPrediction.player_id == Player.player_id
            ).join(
                Game, PlayerPrediction.game_id == Game.game_id
            ).filter(
                and_(
                    Game.game_date >= self.config.evaluation_start_date,
                    Game.game_date <= self.config.evaluation_end_date,
                    PlayerPrediction.confidence_overall >= self.config.confidence_threshold
                )
            )
            
            results = query.all()
            self.logger.info(f"Loaded {len(results)} prediction-actual pairs")
            
            # Process results
            for pred, actual, position, game_date in results:
                if position not in self.config.positions:
                    continue
                    
                # Extract predictions and actuals for each stat type
                stat_mappings = {
                    'passing_yards': (pred.predicted_passing_yards, actual.passing_yards),
                    'passing_tds': (pred.predicted_passing_tds, actual.passing_touchdowns),
                    'rushing_yards': (pred.predicted_rushing_yards, actual.rushing_yards),
                    'rushing_tds': (pred.predicted_rushing_tds, actual.rushing_touchdowns),
                    'receiving_yards': (pred.predicted_receiving_yards, actual.receiving_yards),
                    'receiving_tds': (pred.predicted_receiving_tds, actual.receiving_touchdowns),
                    'receptions': (pred.predicted_receptions, actual.receptions),
                    'fantasy_points': (pred.predicted_fantasy_points, actual.fantasy_points_ppr)
                }
                
                for stat_type, (predicted, actual_val) in stat_mappings.items():
                    if predicted is not None and actual_val is not None:
                        result = PredictionResult(
                            player_id=pred.player_id,
                            game_id=pred.game_id,
                            position=position,
                            stat_type=stat_type,
                            predicted_value=float(predicted),
                            actual_value=float(actual_val),
                            confidence=pred.confidence_overall,
                            game_date=game_date,
                            model_version=pred.model_version
                        )
                        self.prediction_results.append(result)
                        
        self.logger.info(f"Processed {len(self.prediction_results)} individual predictions")
        
    def _calculate_evaluation_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        self.logger.info("📈 Calculating evaluation metrics...")
        
        # Group by position and stat type
        grouped_results = {}
        for result in self.prediction_results:
            key = f"{result.position}_{result.stat_type}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
            
        # Calculate metrics for each group
        for key, results in grouped_results.items():
            position, stat_type = key.split('_', 1)
            
            if len(results) < 5:  # Minimum sample size
                continue
                
            # Extract arrays
            predicted = np.array([r.predicted_value for r in results])
            actual = np.array([r.actual_value for r in results])
            confidence = np.array([r.confidence for r in results])
            
            # Calculate regression metrics
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            
            # Calculate MAPE (handle division by zero)
            mask = actual != 0
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
            else:
                mape = 0.0
                
            # Percentage-based accuracy
            errors = np.abs(predicted - actual)
            relative_errors = np.where(actual != 0, errors / np.abs(actual), 0) * 100
            
            within_10_pct = np.mean(relative_errors <= 10) * 100
            within_20_pct = np.mean(relative_errors <= 20) * 100
            
            # Create metrics object
            metrics = EvaluationMetrics(
                stat_type=stat_type,
                position=position,
                n_predictions=len(results),
                mae=mae,
                rmse=rmse,
                r2=r2,
                mape=mape,
                predictions_within_10_percent=within_10_pct,
                predictions_within_20_percent=within_20_pct,
                median_error=np.median(errors),
                max_error=np.max(errors),
                min_error=np.min(errors)
            )
            
            self.evaluation_metrics[key] = metrics
            
        self.logger.info(f"Calculated metrics for {len(self.evaluation_metrics)} categories")
        
    def _evaluate_betting_performance(self) -> Dict[str, Any]:
        """Evaluate performance for betting applications."""
        self.logger.info("💰 Evaluating betting performance...")
        
        betting_results = {}
        
        with self.Session() as session:
            # Get betting lines for comparison
            betting_lines = session.query(BettingLine).filter(
                BettingLine.line_type == 'player_prop'
            ).all()
            
            # Create lookup for betting lines
            line_lookup = {}
            for line in betting_lines:
                key = f"{line.player_id}_{line.game_id}_{line.prop_type}"
                line_lookup[key] = line.line_value
                
        # Evaluate over/under accuracy
        over_under_results = []
        betting_edges = []
        
        for result in self.prediction_results:
            line_key = f"{result.player_id}_{result.game_id}_{result.stat_type}"
            
            if line_key in line_lookup:
                betting_line = line_lookup[line_key]
                
                # Predicted over/under
                predicted_over = result.predicted_value > betting_line
                actual_over = result.actual_value > betting_line
                
                # Accuracy
                correct = predicted_over == actual_over
                over_under_results.append(correct)
                
                # Calculate edge
                edge = (result.predicted_value - betting_line) / betting_line
                betting_edges.append({
                    'edge': edge,
                    'correct': correct,
                    'confidence': result.confidence
                })
                
        # Calculate betting metrics
        if over_under_results:
            over_under_accuracy = np.mean(over_under_results) * 100
            
            # ROI calculation (simplified)
            roi = self._calculate_betting_roi(betting_edges)
            
            # Sharp ratio (edge accuracy at high confidence)
            high_conf_edges = [e for e in betting_edges if e['confidence'] >= 0.8]
            sharp_ratio = np.mean([e['correct'] for e in high_conf_edges]) * 100 if high_conf_edges else 0
            
            betting_results = {
                'over_under_accuracy': over_under_accuracy,
                'betting_roi': roi,
                'sharp_ratio': sharp_ratio,
                'total_betting_opportunities': len(over_under_results),
                'high_confidence_bets': len(high_conf_edges)
            }
        else:
            betting_results = {
                'over_under_accuracy': 0,
                'betting_roi': 0,
                'sharp_ratio': 0,
                'total_betting_opportunities': 0,
                'high_confidence_bets': 0
            }
            
        return betting_results
        
    def _calculate_betting_roi(self, betting_edges: List[Dict]) -> float:
        """Calculate betting ROI based on edges and outcomes."""
        if not betting_edges:
            return 0.0
            
        total_return = 0.0
        total_bet = 0.0
        
        for edge_data in betting_edges:
            # Simple Kelly criterion-like betting
            edge = edge_data['edge']
            correct = edge_data['correct']
            confidence = edge_data['confidence']
            
            # Only bet if we have positive edge and high confidence
            if abs(edge) >= 0.05 and confidence >= 0.7:
                bet_size = min(abs(edge) * confidence, 0.1)  # Max 10% of bankroll
                total_bet += bet_size
                
                if correct:
                    # Assume -110 odds (1.91 decimal)
                    total_return += bet_size * 1.91
                # If incorrect, we lose the bet (return = 0)
                
        roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else 0.0
        return roi
        
    def _analyze_model_calibration(self) -> Dict[str, Any]:
        """Analyze how well-calibrated model confidence scores are."""
        self.logger.info("🎯 Analyzing model calibration...")
        
        # Group predictions by confidence bins
        confidence_bins = np.arange(0.5, 1.01, 0.1)
        calibration_data = []
        
        for result in self.prediction_results:
            # Find confidence bin
            bin_idx = np.digitize(result.confidence, confidence_bins) - 1
            bin_idx = max(0, min(len(confidence_bins) - 2, bin_idx))
            
            # Calculate if prediction was "accurate" (within 20% of actual)
            relative_error = abs(result.percentage_error)
            accurate = relative_error <= 20
            
            calibration_data.append({
                'confidence_bin': confidence_bins[bin_idx],
                'confidence': result.confidence,
                'accurate': accurate,
                'relative_error': relative_error
            })
            
        # Calculate calibration metrics
        calibration_df = pd.DataFrame(calibration_data)
        calibration_results = {}
        
        for bin_val in confidence_bins[:-1]:
            bin_data = calibration_df[calibration_df['confidence_bin'] == bin_val]
            
            if len(bin_data) > 0:
                accuracy = bin_data['accurate'].mean()
                avg_confidence = bin_data['confidence'].mean()
                count = len(bin_data)
                
                calibration_results[f"bin_{bin_val:.1f}"] = {
                    'expected_accuracy': avg_confidence,
                    'actual_accuracy': accuracy,
                    'calibration_error': abs(avg_confidence - accuracy),
                    'count': count
                }
                
        # Overall calibration metrics
        overall_calibration_error = np.mean([
            data['calibration_error'] for data in calibration_results.values()
        ])
        
        return {
            'bin_results': calibration_results,
            'overall_calibration_error': overall_calibration_error,
            'total_predictions': len(calibration_data)
        }
        
    def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate actionable insights from evaluation results."""
        self.logger.info("💡 Generating performance insights...")
        
        insights = {
            'best_performing': {},
            'worst_performing': {},
            'position_rankings': {},
            'stat_type_rankings': {},
            'recommendations': []
        }
        
        # Find best and worst performing categories
        metrics_list = list(self.evaluation_metrics.values())
        
        if metrics_list:
            # Sort by R² score
            sorted_by_r2 = sorted(metrics_list, key=lambda x: x.r2, reverse=True)
            insights['best_performing'] = {
                'category': f"{sorted_by_r2[0].position}_{sorted_by_r2[0].stat_type}",
                'r2': sorted_by_r2[0].r2,
                'mae': sorted_by_r2[0].mae,
                'n_predictions': sorted_by_r2[0].n_predictions
            }
            
            insights['worst_performing'] = {
                'category': f"{sorted_by_r2[-1].position}_{sorted_by_r2[-1].stat_type}",
                'r2': sorted_by_r2[-1].r2,
                'mae': sorted_by_r2[-1].mae,
                'n_predictions': sorted_by_r2[-1].n_predictions
            }
            
            # Position rankings
            position_performance = {}
            for metrics in metrics_list:
                if metrics.position not in position_performance:
                    position_performance[metrics.position] = []
                position_performance[metrics.position].append(metrics.r2)
                
            for position, r2_scores in position_performance.items():
                position_performance[position] = np.mean(r2_scores)
                
            insights['position_rankings'] = dict(
                sorted(position_performance.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Generate recommendations
            recommendations = []
            
            # Low performing categories
            low_performers = [m for m in metrics_list if m.r2 < 0.3]
            if low_performers:
                recommendations.append(
                    f"Consider improving models for {len(low_performers)} underperforming categories "
                    f"(R² < 0.3)"
                )
                
            # High error categories
            high_error = [m for m in metrics_list if m.mape > 30]
            if high_error:
                recommendations.append(
                    f"Focus on reducing prediction errors for {len(high_error)} categories "
                    f"with MAPE > 30%"
                )
                
            # Sample size issues
            small_samples = [m for m in metrics_list if m.n_predictions < 20]
            if small_samples:
                recommendations.append(
                    f"Collect more data for {len(small_samples)} categories with insufficient samples"
                )
                
            insights['recommendations'] = recommendations
            
        return insights
        
    def _create_evaluation_plots(self):
        """Create visualization plots for evaluation results."""
        self.logger.info("📊 Creating evaluation plots...")
        
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            
            # Plot 1: R² by Position and Stat Type
            ax1 = plt.subplot(2, 3, 1)
            self._plot_r2_heatmap(ax1)
            
            # Plot 2: MAE Distribution
            ax2 = plt.subplot(2, 3, 2)
            self._plot_mae_distribution(ax2)
            
            # Plot 3: Prediction vs Actual Scatter
            ax3 = plt.subplot(2, 3, 3)
            self._plot_prediction_scatter(ax3)
            
            # Plot 4: Confidence Calibration
            ax4 = plt.subplot(2, 3, 4)
            self._plot_confidence_calibration(ax4)
            
            # Plot 5: Error Distribution by Position
            ax5 = plt.subplot(2, 3, 5)
            self._plot_error_by_position(ax5)
            
            # Plot 6: Performance Timeline
            ax6 = plt.subplot(2, 3, 6)
            self._plot_performance_timeline(ax6)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(self.config.results_directory) / "evaluation_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Evaluation plots saved to {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create plots: {e}")
            
    def _plot_r2_heatmap(self, ax):
        """Plot R² scores as heatmap."""
        # Prepare data for heatmap
        positions = list(set(m.position for m in self.evaluation_metrics.values()))
        stat_types = list(set(m.stat_type for m in self.evaluation_metrics.values()))
        
        r2_matrix = np.zeros((len(positions), len(stat_types)))
        r2_matrix.fill(np.nan)
        
        for i, pos in enumerate(positions):
            for j, stat in enumerate(stat_types):
                key = f"{pos}_{stat}"
                if key in self.evaluation_metrics:
                    r2_matrix[i, j] = self.evaluation_metrics[key].r2
                    
        sns.heatmap(
            r2_matrix,
            xticklabels=stat_types,
            yticklabels=positions,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            ax=ax
        )
        ax.set_title('R² Scores by Position and Stat Type')
        
    def _plot_mae_distribution(self, ax):
        """Plot MAE distribution."""
        mae_values = [m.mae for m in self.evaluation_metrics.values()]
        ax.hist(mae_values, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Mean Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of MAE Across All Categories')
        ax.axvline(np.mean(mae_values), color='red', linestyle='--', label=f'Mean: {np.mean(mae_values):.2f}')
        ax.legend()
        
    def _plot_prediction_scatter(self, ax):
        """Plot predicted vs actual scatter plot."""
        # Sample results for visualization
        sample_results = self.prediction_results[::max(1, len(self.prediction_results) // 1000)]
        
        predicted = [r.predicted_value for r in sample_results]
        actual = [r.actual_value for r in sample_results]
        
        ax.scatter(actual, predicted, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(min(predicted), min(actual))
        max_val = max(max(predicted), max(actual))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values (Sample)')
        ax.legend()
        
    def _plot_confidence_calibration(self, ax):
        """Plot confidence calibration curve."""
        # This would use the calibration analysis results
        ax.plot([0.5, 1.0], [0.5, 1.0], 'r--', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('Actual Accuracy')
        ax.set_title('Confidence Calibration')
        ax.legend()
        
    def _plot_error_by_position(self, ax):
        """Plot error distribution by position."""
        position_errors = {}
        for result in self.prediction_results:
            if result.position not in position_errors:
                position_errors[result.position] = []
            position_errors[result.position].append(result.percentage_error)
            
        positions = list(position_errors.keys())
        errors = [position_errors[pos] for pos in positions]
        
        ax.boxplot(errors, labels=positions)
        ax.set_ylabel('Percentage Error (%)')
        ax.set_title('Error Distribution by Position')
        
    def _plot_performance_timeline(self, ax):
        """Plot performance over time."""
        # Group by month and calculate average R²
        monthly_performance = {}
        for key, metrics in self.evaluation_metrics.items():
            # This would need game dates grouped by month
            # Simplified for demonstration
            month = "2024-01"  # Placeholder
            if month not in monthly_performance:
                monthly_performance[month] = []
            monthly_performance[month].append(metrics.r2)
            
        months = list(monthly_performance.keys())
        avg_r2 = [np.mean(monthly_performance[month]) for month in months]
        
        ax.plot(months, avg_r2, marker='o')
        ax.set_ylabel('Average R² Score')
        ax.set_title('Model Performance Over Time')
        ax.tick_params(axis='x', rotation=45)
        
    def _save_evaluation_results(self, betting_results, calibration_results, insights) -> Dict[str, Any]:
        """Save evaluation results to files."""
        results_dir = Path(self.config.results_directory)
        
        # Convert metrics to serializable format
        metrics_dict = {}
        for key, metrics in self.evaluation_metrics.items():
            metrics_dict[key] = {
                'stat_type': metrics.stat_type,
                'position': metrics.position,
                'n_predictions': metrics.n_predictions,
                'mae': float(metrics.mae),
                'rmse': float(metrics.rmse),
                'r2': float(metrics.r2),
                'mape': float(metrics.mape),
                'predictions_within_10_percent': float(metrics.predictions_within_10_percent),
                'predictions_within_20_percent': float(metrics.predictions_within_20_percent),
                'median_error': float(metrics.median_error),
                'max_error': float(metrics.max_error),
                'min_error': float(metrics.min_error)
            }
            
        # Compile all results
        all_results = {
            'evaluation_config': {
                'start_date': self.config.evaluation_start_date.isoformat(),
                'end_date': self.config.evaluation_end_date.isoformat(),
                'positions': self.config.positions,
                'stat_types': self.config.stat_types,
                'confidence_threshold': self.config.confidence_threshold
            },
            'metrics': metrics_dict,
            'betting_performance': betting_results,
            'calibration': calibration_results,
            'insights': insights,
            'summary': {
                'total_predictions': len(self.prediction_results),
                'evaluation_categories': len(self.evaluation_metrics),
                'average_r2': np.mean([m.r2 for m in self.evaluation_metrics.values()]),
                'average_mae': np.mean([m.mae for m in self.evaluation_metrics.values()]),
                'evaluation_date': datetime.now().isoformat()
            }
        }
        
        # Save to JSON
        results_file = results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
            
        # Save detailed results to CSV
        results_df = pd.DataFrame([
            {
                'player_id': r.player_id,
                'game_id': r.game_id,
                'position': r.position,
                'stat_type': r.stat_type,
                'predicted_value': r.predicted_value,
                'actual_value': r.actual_value,
                'absolute_error': r.absolute_error,
                'percentage_error': r.percentage_error,
                'confidence': r.confidence,
                'game_date': r.game_date,
                'model_version': r.model_version
            }
            for r in self.prediction_results
        ])
        
        csv_file = results_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {results_file} and {csv_file}")
        
        return all_results


# Utility functions for backtesting
def run_rolling_backtest(
    database_url: str,
    start_date: date,
    end_date: date,
    window_weeks: int = 4
) -> Dict[str, Any]:
    """Run rolling window backtest."""
    
    results = []
    current_date = start_date
    
    while current_date <= end_date:
        window_end = current_date + timedelta(weeks=window_weeks)
        
        config = EvaluationConfig(
            database_url=database_url,
            evaluation_start_date=current_date,
            evaluation_end_date=min(window_end, end_date),
            plot_results=False,
            save_results=False
        )
        
        evaluator = NFLModelEvaluator(config)
        window_results = evaluator.run_comprehensive_evaluation()
        
        results.append({
            'start_date': current_date,
            'end_date': min(window_end, end_date),
            'results': window_results
        })
        
        current_date += timedelta(weeks=1)  # Roll forward by 1 week
        
    return {
        'rolling_results': results,
        'window_weeks': window_weeks,
        'total_windows': len(results)
    }


# Example usage
def main():
    """Example usage of the evaluation system."""
    
    config = EvaluationConfig(
        database_url="postgresql://user:password@localhost/nfl_predictions",
        evaluation_start_date=date(2024, 9, 1),
        evaluation_end_date=date(2024, 12, 31),
        positions=['QB', 'RB', 'WR', 'TE'],
        confidence_threshold=0.6,
        plot_results=True,
        save_results=True
    )
    
    evaluator = NFLModelEvaluator(config)
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n🏈 NFL Model Evaluation Results 🏈")
    print("=" * 50)
    
    print(f"\nTotal Predictions Evaluated: {results['summary']['total_predictions']}")
    print(f"Average R² Score: {results['summary']['average_r2']:.3f}")
    print(f"Average MAE: {results['summary']['average_mae']:.2f}")
    
    print(f"\nBetting Performance:")
    print(f"  Over/Under Accuracy: {results['betting_performance']['over_under_accuracy']:.1f}%")
    print(f"  Betting ROI: {results['betting_performance']['betting_roi']:.1f}%")
    print(f"  Sharp Ratio: {results['betting_performance']['sharp_ratio']:.1f}%")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(results['insights']['recommendations'][:3], 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()