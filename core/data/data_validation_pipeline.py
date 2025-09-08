"""
Data Validation and Pipeline Management
Implements data quality checks, outlier detection, and automated workflows.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Data validation rule definition."""
    name: str
    column: str
    rule_type: str  # 'range', 'not_null', 'unique', 'pattern', 'outlier'
    parameters: Dict[str, Any]
    severity: str = 'error'  # 'error', 'warning', 'info'
    description: str = ""

class DataValidator:
    """Comprehensive data validation system."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.validation_rules = self._initialize_validation_rules()
        self.validation_results = {}
        
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize comprehensive validation rules."""
        return [
            # Player data validation
            ValidationRule(
                name="player_id_not_null",
                column="player_id",
                rule_type="not_null",
                parameters={},
                description="Player ID must not be null"
            ),
            ValidationRule(
                name="position_valid",
                column="position",
                rule_type="pattern",
                parameters={"allowed_values": ["QB", "RB", "WR", "TE", "K", "DEF"]},
                description="Position must be valid NFL position"
            ),
            
            # Game stats validation
            ValidationRule(
                name="fantasy_points_range",
                column="fantasy_points_ppr",
                rule_type="range",
                parameters={"min_value": 0, "max_value": 60},
                description="Fantasy points should be between 0 and 60"
            ),
            ValidationRule(
                name="passing_yards_range",
                column="passing_yards",
                rule_type="range",
                parameters={"min_value": 0, "max_value": 600},
                description="Passing yards should be reasonable"
            ),
            ValidationRule(
                name="rushing_yards_range",
                column="rushing_yards",
                rule_type="range",
                parameters={"min_value": -50, "max_value": 400},
                description="Rushing yards should be reasonable (negative for sacks)"
            ),
            ValidationRule(
                name="receiving_yards_range",
                column="receiving_yards",
                rule_type="range",
                parameters={"min_value": 0, "max_value": 300},
                description="Receiving yards should be reasonable"
            ),
            
            # Outlier detection
            ValidationRule(
                name="fantasy_points_outlier",
                column="fantasy_points_ppr",
                rule_type="outlier",
                parameters={"method": "iqr", "threshold": 3.0},
                severity="warning",
                description="Detect fantasy points outliers"
            ),
            ValidationRule(
                name="passing_attempts_outlier",
                column="passing_attempts",
                rule_type="outlier",
                parameters={"method": "zscore", "threshold": 3.0},
                severity="warning",
                description="Detect passing attempts outliers"
            ),
        ]
    
    def validate_table(self, table_name: str) -> Dict[str, Any]:
        """Validate a specific table."""
        logger.info(f"Validating table: {table_name}")
        
        try:
            with self.Session() as session:
                # Get table data
                query = text(f"SELECT * FROM {table_name} LIMIT 10000")
                result = session.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if df.empty:
                    return {"error": f"No data found in {table_name}"}
                
                validation_results = {
                    "table": table_name,
                    "total_records": len(df),
                    "validation_timestamp": datetime.now().isoformat(),
                    "rules_passed": 0,
                    "rules_failed": 0,
                    "warnings": 0,
                    "errors": [],
                    "warnings_list": [],
                    "summary": {}
                }
                
                # Apply validation rules
                for rule in self.validation_rules:
                    if rule.column in df.columns:
                        result = self._apply_validation_rule(df, rule)
                        
                        if result["passed"]:
                            validation_results["rules_passed"] += 1
                        else:
                            validation_results["rules_failed"] += 1
                            
                            if rule.severity == "error":
                                validation_results["errors"].append(result)
                            elif rule.severity == "warning":
                                validation_results["warnings"] += 1
                                validation_results["warnings_list"].append(result)
                
                # Data quality summary
                validation_results["summary"] = self._generate_data_quality_summary(df)
                
                return validation_results
                
        except Exception as e:
            logger.error(f"Error validating table {table_name}: {e}")
            return {"error": str(e)}
    
    def _apply_validation_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Apply a single validation rule."""
        try:
            if rule.rule_type == "not_null":
                null_count = df[rule.column].isnull().sum()
                passed = null_count == 0
                
                return {
                    "rule": rule.name,
                    "passed": passed,
                    "message": f"Found {null_count} null values in {rule.column}",
                    "severity": rule.severity,
                    "details": {"null_count": int(null_count)}
                }
                
            elif rule.rule_type == "range":
                min_val = rule.parameters.get("min_value")
                max_val = rule.parameters.get("max_value")
                
                out_of_range = 0
                if min_val is not None:
                    out_of_range += (df[rule.column] < min_val).sum()
                if max_val is not None:
                    out_of_range += (df[rule.column] > max_val).sum()
                
                passed = out_of_range == 0
                
                return {
                    "rule": rule.name,
                    "passed": passed,
                    "message": f"Found {out_of_range} values outside range [{min_val}, {max_val}]",
                    "severity": rule.severity,
                    "details": {"out_of_range_count": int(out_of_range)}
                }
                
            elif rule.rule_type == "pattern":
                allowed_values = rule.parameters.get("allowed_values", [])
                invalid_count = (~df[rule.column].isin(allowed_values)).sum()
                passed = invalid_count == 0
                
                return {
                    "rule": rule.name,
                    "passed": passed,
                    "message": f"Found {invalid_count} invalid values in {rule.column}",
                    "severity": rule.severity,
                    "details": {"invalid_count": int(invalid_count)}
                }
                
            elif rule.rule_type == "outlier":
                outliers = self._detect_outliers(df[rule.column], rule.parameters)
                passed = len(outliers) == 0
                
                return {
                    "rule": rule.name,
                    "passed": passed,
                    "message": f"Found {len(outliers)} outliers in {rule.column}",
                    "severity": rule.severity,
                    "details": {"outlier_count": len(outliers), "outlier_indices": outliers.tolist()}
                }
                
            else:
                return {
                    "rule": rule.name,
                    "passed": False,
                    "message": f"Unknown rule type: {rule.rule_type}",
                    "severity": "error"
                }
                
        except Exception as e:
            return {
                "rule": rule.name,
                "passed": False,
                "message": f"Error applying rule: {str(e)}",
                "severity": "error"
            }
    
    def _detect_outliers(self, series: pd.Series, parameters: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using specified method."""
        method = parameters.get("method", "iqr")
        threshold = parameters.get("threshold", 3.0)
        
        # Remove null values
        clean_series = series.dropna()
        
        if method == "iqr":
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            return outliers.index.values
            
        elif method == "zscore":
            z_scores = np.abs((clean_series - clean_series.mean()) / clean_series.std())
            outliers = clean_series[z_scores > threshold]
            return outliers.index.values
            
        else:
            return np.array([])
    
    def _generate_data_quality_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality summary."""
        summary = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_data": {},
            "data_types": {},
            "basic_stats": {}
        }
        
        # Missing data analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            summary["missing_data"][col] = {
                "count": int(missing_count),
                "percentage": float(missing_pct)
            }
        
        # Data types
        for col in df.columns:
            summary["data_types"][col] = str(df[col].dtype)
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary["basic_stats"][col] = {
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                "std": float(df[col].std()) if not df[col].isnull().all() else None,
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None,
                "unique_values": int(df[col].nunique())
            }
        
        return summary
    
    def validate_all_tables(self) -> Dict[str, Any]:
        """Validate all important tables."""
        tables_to_validate = [
            "players",
            "player_game_stats", 
            "games"
        ]
        
        all_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "tables_validated": len(tables_to_validate),
            "overall_status": "passed",
            "table_results": {}
        }
        
        for table in tables_to_validate:
            try:
                result = self.validate_table(table)
                all_results["table_results"][table] = result
                
                # Check if any critical errors
                if "error" in result or result.get("rules_failed", 0) > 0:
                    all_results["overall_status"] = "failed"
                    
            except Exception as e:
                logger.error(f"Error validating {table}: {e}")
                all_results["table_results"][table] = {"error": str(e)}
                all_results["overall_status"] = "failed"
        
        return all_results

class AutomatedWorkflowManager:
    """Manages automated data processing and model retraining workflows."""
    
    def __init__(self, database_url: str, config: Dict[str, Any]):
        self.database_url = database_url
        self.config = config
        self.validator = DataValidator(database_url)
        self.workflow_history = []
        
    def run_daily_workflow(self) -> Dict[str, Any]:
        """Run daily automated workflow."""
        logger.info("🔄 Starting daily automated workflow")
        
        workflow_result = {
            "workflow_type": "daily",
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "overall_status": "success"
        }
        
        try:
            # Step 1: Data validation
            logger.info("Step 1: Data validation")
            validation_result = self.validator.validate_all_tables()
            workflow_result["steps"].append({
                "step": "data_validation",
                "status": "success" if validation_result["overall_status"] == "passed" else "warning",
                "result": validation_result
            })
            
            # Step 2: Check for new data
            logger.info("Step 2: Checking for new data")
            new_data_check = self._check_for_new_data()
            workflow_result["steps"].append({
                "step": "new_data_check",
                "status": "success",
                "result": new_data_check
            })
            
            # Step 3: Model performance monitoring
            logger.info("Step 3: Model performance monitoring")
            performance_check = self._monitor_model_performance()
            workflow_result["steps"].append({
                "step": "performance_monitoring",
                "status": "success",
                "result": performance_check
            })
            
            # Step 4: Automated retraining (if needed)
            if self._should_retrain_models(new_data_check, performance_check):
                logger.info("Step 4: Automated model retraining")
                retrain_result = self._trigger_model_retraining()
                workflow_result["steps"].append({
                    "step": "model_retraining",
                    "status": "success" if retrain_result["success"] else "error",
                    "result": retrain_result
                })
            
            workflow_result["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error in daily workflow: {e}")
            workflow_result["overall_status"] = "error"
            workflow_result["error"] = str(e)
        
        # Store workflow history
        self.workflow_history.append(workflow_result)
        
        return workflow_result
    
    def _check_for_new_data(self) -> Dict[str, Any]:
        """Check for new data since last workflow run."""
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # Check for new player stats
                result = conn.execute(text("""
                    SELECT COUNT(*) as new_stats
                    FROM player_game_stats 
                    WHERE created_at > datetime('now', '-1 day')
                """))
                
                new_stats = result.scalar()
                
                # Check for new players
                result = conn.execute(text("""
                    SELECT COUNT(*) as new_players
                    FROM players 
                    WHERE created_at > datetime('now', '-1 day')
                """))
                
                new_players = result.scalar()
                
                return {
                    "new_player_stats": new_stats,
                    "new_players": new_players,
                    "requires_retraining": new_stats > 100 or new_players > 5
                }
                
        except Exception as e:
            logger.error(f"Error checking for new data: {e}")
            return {"error": str(e)}
    
    def _monitor_model_performance(self) -> Dict[str, Any]:
        """Monitor model performance metrics."""
        try:
            # This would check recent prediction accuracy
            # For now, return a placeholder
            return {
                "models_checked": 0,
                "performance_degradation": False,
                "avg_accuracy": 0.75,
                "requires_retraining": False
            }
            
        except Exception as e:
            logger.error(f"Error monitoring model performance: {e}")
            return {"error": str(e)}
    
    def _should_retrain_models(self, new_data_check: Dict, performance_check: Dict) -> bool:
        """Determine if models should be retrained."""
        return (
            new_data_check.get("requires_retraining", False) or
            performance_check.get("requires_retraining", False)
        )
    
    def _trigger_model_retraining(self) -> Dict[str, Any]:
        """Trigger automated model retraining."""
        try:
            # This would trigger the comprehensive model training
            logger.info("Triggering automated model retraining")
            
            # For now, return a placeholder
            return {
                "success": True,
                "models_retrained": 0,
                "training_time": "0 minutes",
                "new_performance": {}
            }
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Example usage of data validation and workflow management."""
    
    # Initialize validator
    validator = DataValidator("sqlite:///data/nfl_predictions.db")
    
    # Run validation
    print("🔍 Running comprehensive data validation...")
    results = validator.validate_all_tables()
    
    print(f"Validation Status: {results['overall_status']}")
    print(f"Tables Validated: {results['tables_validated']}")
    
    for table, result in results["table_results"].items():
        if "error" not in result:
            print(f"\n{table}:")
            print(f"  Records: {result['total_records']:,}")
            print(f"  Rules Passed: {result['rules_passed']}")
            print(f"  Rules Failed: {result['rules_failed']}")
            print(f"  Warnings: {result['warnings']}")
    
    # Initialize workflow manager
    config = {"retrain_threshold": 100}
    workflow_manager = AutomatedWorkflowManager("sqlite:///data/nfl_predictions.db", config)
    
    # Run daily workflow
    print("\n🔄 Running daily workflow...")
    workflow_result = workflow_manager.run_daily_workflow()
    
    print(f"Workflow Status: {workflow_result['overall_status']}")
    print(f"Steps Completed: {len(workflow_result['steps'])}")

if __name__ == "__main__":
    main()
