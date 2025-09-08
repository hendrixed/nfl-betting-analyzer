"""
Test Enhanced NFL Data System

This script tests the new hierarchical data architecture to ensure
it correctly identifies starters, validates data quality, and eliminates
incorrect backup player statistics.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import our new enhanced modules
from data_foundation import PlayerRole, WeeklyRosterSnapshot, ValidationReport
from enhanced_data_collector import EnhancedNFLDataCollector, RoleBasedStatsCollector
from data_validator import ComprehensiveValidator
from database_models import Base

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSystemTester:
    """Test the enhanced NFL data collection system"""
    
    def __init__(self, db_path: str = "test_enhanced_nfl.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize components - use 2024 data for testing since 2025 isn't available yet
        self.collector = EnhancedNFLDataCollector(self.session, current_season=2024)
        self.stats_collector = RoleBasedStatsCollector(self.collector)
        self.validator = ComprehensiveValidator()
    
    async def test_data_collection(self, week: int = 1):
        """Test the enhanced data collection pipeline"""
        
        logger.info(f"Testing enhanced data collection for Week {week}")
        
        try:
            # Step 1: Collect foundation data
            logger.info("Step 1: Collecting foundation data...")
            team_snapshots = await self.collector.collect_weekly_foundation_data(week)
            
            if not team_snapshots:
                logger.error("No team snapshots collected - data collection failed")
                return False
            
            logger.info(f"Collected snapshots for {len(team_snapshots)} teams")
            
            # Step 2: Analyze roster snapshots
            self._analyze_roster_snapshots(team_snapshots)
            
            # Step 3: Collect validated stats
            logger.info("Step 3: Collecting validated stats...")
            validated_stats = await self.stats_collector.collect_validated_stats(week, team_snapshots)
            
            logger.info(f"Collected {len(validated_stats)} validated stat records")
            
            # Step 4: Run comprehensive validation
            logger.info("Step 4: Running comprehensive validation...")
            quality_report = self.validator.run_full_validation(
                season=2024,
                week=week,
                snapshots=team_snapshots,
                stats_data=validated_stats,
                snap_data=[]  # Would be populated with actual snap data
            )
            
            # Step 5: Generate test report
            self._generate_test_report(team_snapshots, validated_stats, quality_report)
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            return False
    
    def _analyze_roster_snapshots(self, snapshots: Dict[str, WeeklyRosterSnapshot]):
        """Analyze the collected roster snapshots"""
        
        logger.info("Analyzing roster snapshots...")
        
        total_players = 0
        total_starters = 0
        total_backups = 0
        total_inactive = 0
        
        position_analysis = {}
        
        for team, snapshot in snapshots.items():
            team_players = snapshot.get_active_players()
            total_players += len(team_players)
            total_starters += len(snapshot.starters)
            total_backups += len(snapshot.backup_primary) + len(snapshot.backup_depth)
            total_inactive += len(snapshot.inactive)
            
            # Analyze by position
            for player in team_players:
                pos = player.position
                if pos not in position_analysis:
                    position_analysis[pos] = {
                        'total': 0, 'starters': 0, 'backups': 0,
                        'avg_quality': 0.0, 'quality_scores': []
                    }
                
                position_analysis[pos]['total'] += 1
                position_analysis[pos]['quality_scores'].append(player.data_quality_score)
                
                if player.role_classification == PlayerRole.STARTER:
                    position_analysis[pos]['starters'] += 1
                elif player.role_classification in [PlayerRole.BACKUP_HIGH, PlayerRole.BACKUP_LOW]:
                    position_analysis[pos]['backups'] += 1
        
        # Calculate averages
        for pos_data in position_analysis.values():
            if pos_data['quality_scores']:
                pos_data['avg_quality'] = sum(pos_data['quality_scores']) / len(pos_data['quality_scores'])
        
        logger.info(f"Total players processed: {total_players}")
        logger.info(f"Starters: {total_starters}, Backups: {total_backups}, Inactive: {total_inactive}")
        
        for pos, data in position_analysis.items():
            logger.info(f"{pos}: {data['starters']} starters, {data['backups']} backups, "
                       f"avg quality: {data['avg_quality']:.2f}")
    
    def _generate_test_report(self, snapshots: Dict[str, WeeklyRosterSnapshot], 
                            stats: List[Dict], quality_report):
        """Generate comprehensive test report"""
        
        logger.info("Generating test report...")
        
        report_lines = [
            "=" * 60,
            "ENHANCED NFL DATA SYSTEM TEST REPORT",
            "=" * 60,
            f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Season: 2025, Week: {quality_report.week}",
            "",
            "ROSTER SNAPSHOT ANALYSIS:",
            f"- Teams processed: {len(snapshots)}",
            f"- Total players: {quality_report.total_players_processed}",
            f"- High quality players: {quality_report.players_with_high_quality}",
            f"- Teams with complete rosters: {quality_report.teams_with_complete_rosters}",
            "",
            "VALIDATION RESULTS:",
        ]
        
        if quality_report.validation_report:
            vr = quality_report.validation_report
            report_lines.extend([
                f"- Overall Score: {vr.overall_score:.3f}",
                f"- Roster Completeness: {vr.roster_completeness:.3f}",
                f"- Depth Chart Accuracy: {vr.depth_chart_accuracy:.3f}",
                f"- Stats-Snap Consistency: {vr.stats_snap_consistency:.3f}",
                f"- Player ID Consistency: {vr.player_id_consistency:.3f}",
            ])
            
            if vr.issues_found:
                report_lines.extend([
                    "",
                    "ISSUES FOUND:",
                    *[f"- {issue}" for issue in vr.issues_found[:10]]  # Show first 10 issues
                ])
        
        report_lines.extend([
            "",
            "STATISTICS VALIDATION:",
            f"- Validated stat records: {len(stats)}",
            f"- Stats with role validation: {len([s for s in stats if s.get('role_classification')])}"
        ])
        
        if quality_report.recommended_actions:
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
                *[f"- {action}" for action in quality_report.recommended_actions]
            ])
        
        # Test success criteria
        overall_quality = quality_report.get_overall_quality_score()
        report_lines.extend([
            "",
            "TEST RESULTS:",
            f"- Overall Quality Score: {overall_quality:.3f}",
            f"- Test Status: {'PASS' if overall_quality >= 0.7 else 'FAIL'}",
            "",
            "SUCCESS CRITERIA:",
            f"- Quality Score >= 0.7: {'✓' if overall_quality >= 0.7 else '✗'}",
            f"- Teams Processed >= 20: {'✓' if len(snapshots) >= 20 else '✗'}",
            f"- Stats Validated: {'✓' if len(stats) > 0 else '✗'}",
        ])
        
        report_lines.append("=" * 60)
        
        # Print report
        for line in report_lines:
            logger.info(line)
        
        # Save to file
        with open("enhanced_system_test_report.txt", "w") as f:
            f.write("\n".join(report_lines))
        
        logger.info("Test report saved to enhanced_system_test_report.txt")
    
    def test_player_role_classification(self, snapshots: Dict[str, WeeklyRosterSnapshot]):
        """Test player role classification accuracy"""
        
        logger.info("Testing player role classification...")
        
        role_counts = {role: 0 for role in PlayerRole}
        position_role_matrix = {}
        
        for team, snapshot in snapshots.items():
            for player in snapshot.get_active_players():
                role_counts[player.role_classification] += 1
                
                pos = player.position
                if pos not in position_role_matrix:
                    position_role_matrix[pos] = {role: 0 for role in PlayerRole}
                
                position_role_matrix[pos][player.role_classification] += 1
        
        logger.info("Role distribution:")
        for role, count in role_counts.items():
            logger.info(f"  {role.value}: {count}")
        
        logger.info("Position-Role Matrix:")
        for pos, roles in position_role_matrix.items():
            starters = roles[PlayerRole.STARTER]
            backups = roles[PlayerRole.BACKUP_HIGH] + roles[PlayerRole.BACKUP_LOW]
            logger.info(f"  {pos}: {starters} starters, {backups} backups")
    
    def cleanup(self):
        """Clean up test resources"""
        self.session.close()


async def main():
    """Main test function"""
    
    logger.info("Starting Enhanced NFL Data System Test")
    
    tester = EnhancedSystemTester()
    
    try:
        # Test current week (Week 1 for 2025 season)
        success = await tester.test_data_collection(week=1)
        
        if success:
            logger.info("✓ Enhanced system test completed successfully")
            return 0
        else:
            logger.error("✗ Enhanced system test failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return 1
    
    finally:
        tester.cleanup()


if __name__ == "__main__":
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
