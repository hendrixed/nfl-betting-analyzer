"""
Test Historical Data Standardizer

This script tests the historical data standardization system on a single season
to validate functionality before running on all historical data.
"""

import logging
import asyncio
import sys
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database_models import create_all_tables
from historical_data_standardizer import HistoricalDataStandardizer
from config_manager import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_historical_standardizer():
    """Test the historical data standardizer on 2024 season"""
    
    logger.info("Starting Historical Data Standardizer Test")
    logger.info("=" * 60)
    
    try:
        # Setup database
        db_path = "test_historical_standardizer.db"
        engine = create_engine(f"sqlite:///{db_path}")
        create_all_tables(engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Initialize standardizer for 2024 season only
        standardizer = HistoricalDataStandardizer(
            session=session,
            target_seasons=[2024]  # Test with single season first
        )
        
        logger.info("Test 1: Analyzing existing data structure")
        existing_data = await standardizer._analyze_existing_data()
        
        if existing_data['seasons_with_data']:
            season_data = existing_data['seasons_with_data'][2024]
            logger.info(f"✓ 2024 season data found:")
            logger.info(f"  - Total records: {season_data['total_records']}")
            logger.info(f"  - Unique players: {season_data['unique_players']}")
            logger.info(f"  - Weeks covered: {season_data['weeks_covered']}")
            logger.info(f"  - Missing data: {season_data['missing_data_percentage']:.1%}")
            logger.info(f"  - Available columns: {len(season_data['stat_columns'])}")
        else:
            logger.warning("✗ No 2024 season data found")
            return
        
        logger.info("\nTest 2: Player identity standardization")
        player_mapping = await standardizer._standardize_player_identities()
        
        total_players = len(set(player_mapping.keys()))
        unique_masters = len(set(player_mapping.values()))
        consolidation_rate = (total_players - unique_masters) / total_players if total_players > 0 else 0
        
        logger.info(f"✓ Player identity mapping created:")
        logger.info(f"  - Original player IDs: {total_players}")
        logger.info(f"  - Master player IDs: {unique_masters}")
        logger.info(f"  - Consolidation rate: {consolidation_rate:.1%}")
        
        logger.info("\nTest 3: Statistical terminology standardization")
        stat_mappings = await standardizer._standardize_statistical_terminology()
        
        if 2024 in stat_mappings:
            season_mappings = stat_mappings[2024]
            logger.info(f"✓ Statistical terminology mapping created:")
            logger.info(f"  - Standardized stats mapped: {len(season_mappings)}")
            
            # Show sample mappings
            sample_mappings = list(season_mappings.items())[:5]
            for standard, original in sample_mappings:
                logger.info(f"  - {standard} <- {original}")
        else:
            logger.warning("✗ No statistical mappings created for 2024")
        
        logger.info("\nTest 4: Full season standardization")
        season_report = await standardizer._standardize_season_data(
            season=2024,
            player_mapping=player_mapping,
            stat_mappings=stat_mappings
        )
        
        logger.info(f"✓ Season standardization completed:")
        logger.info(f"  - Records processed: {season_report['records_processed']}")
        logger.info(f"  - Records standardized: {season_report['records_standardized']}")
        logger.info(f"  - Player mappings applied: {season_report['player_mappings_applied']}")
        logger.info(f"  - Stat mappings applied: {season_report['stat_mappings_applied']}")
        logger.info(f"  - Quality score: {season_report['quality_score']:.3f}")
        
        if season_report['issues']:
            logger.warning(f"  - Issues found: {len(season_report['issues'])}")
            for issue in season_report['issues'][:3]:  # Show first 3 issues
                logger.warning(f"    * {issue}")
        
        logger.info("\nTest 5: Data validation")
        validation_report = await standardizer._validate_standardized_data()
        
        logger.info(f"✓ Data validation completed:")
        logger.info(f"  - Cross-season consistency: {validation_report['cross_season_consistency']:.3f}")
        logger.info(f"  - Statistical completeness: {validation_report['statistical_completeness']:.3f}")
        logger.info(f"  - Player identity consistency: {validation_report['player_identity_consistency']:.3f}")
        logger.info(f"  - Overall validation score: {validation_report['overall_validation_score']:.3f}")
        
        logger.info("\nTest 6: Final standardization report")
        final_report = await standardizer._generate_standardization_report()
        
        logger.info(f"✓ Final report generated:")
        logger.info(f"  - Seasons processed: {len(final_report.seasons_processed)}")
        logger.info(f"  - Players standardized: {final_report.players_standardized}")
        logger.info(f"  - Stats records processed: {final_report.stats_records_processed}")
        logger.info(f"  - Overall quality score: {final_report.data_quality_score:.3f}")
        
        # Test database queries
        logger.info("\nTest 7: Database integration validation")
        
        # Check stored mappings
        from database_models import PlayerIdentityMapping, StatTerminologyMapping, HistoricalDataStandardization
        
        identity_mappings = session.query(PlayerIdentityMapping).count()
        stat_mappings_count = session.query(StatTerminologyMapping).count()
        standardization_records = session.query(HistoricalDataStandardization).count()
        
        logger.info(f"✓ Database integration validated:")
        logger.info(f"  - Player identity mappings stored: {identity_mappings}")
        logger.info(f"  - Statistical mappings stored: {stat_mappings_count}")
        logger.info(f"  - Standardization records: {standardization_records}")
        
        # Generate test summary
        logger.info("\n" + "=" * 60)
        logger.info("HISTORICAL DATA STANDARDIZER TEST SUMMARY")
        logger.info("=" * 60)
        
        test_results = {
            'data_analysis': bool(existing_data['seasons_with_data']),
            'player_identity': len(player_mapping) > 0,
            'stat_terminology': len(stat_mappings.get(2024, {})) > 0,
            'season_standardization': season_report['quality_score'] > 0.5,
            'data_validation': validation_report['overall_validation_score'] > 0.5,
            'database_integration': identity_mappings > 0 and stat_mappings_count > 0
        }
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        for test_name, passed in test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  - {test_name.replace('_', ' ').title()}: {status}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED - Historical Data Standardizer is ready for production!")
            logger.info("✅ Proceed with full historical data standardization (2020-2024)")
        else:
            logger.warning(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED - Review issues before proceeding")
            logger.warning("❌ Fix issues before running full standardization")
        
        # Performance metrics
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  - Processing rate: {season_report['records_standardized']/max(1, season_report['records_processed']):.1%}")
        logger.info(f"  - Data quality score: {season_report['quality_score']:.3f}")
        logger.info(f"  - Player consolidation: {consolidation_rate:.1%}")
        
        session.close()
        
        return test_results
        
    except Exception as e:
        logger.error(f"Historical data standardizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def run_full_standardization():
    """Run full historical data standardization on all seasons"""
    
    logger.info("Starting FULL Historical Data Standardization")
    logger.info("=" * 60)
    logger.info("Processing seasons: 2020, 2021, 2022, 2023, 2024")
    
    try:
        # Setup database
        db_path = "nfl_predictions_standardized.db"
        engine = create_engine(f"sqlite:///{db_path}")
        create_all_tables(engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Initialize standardizer for all seasons
        standardizer = HistoricalDataStandardizer(
            session=session,
            target_seasons=[2020, 2021, 2022, 2023, 2024]
        )
        
        # Run full standardization
        final_report = await standardizer.standardize_all_historical_data()
        
        logger.info("FULL STANDARDIZATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Seasons processed: {len(final_report.seasons_processed)}")
        logger.info(f"Players standardized: {final_report.players_standardized}")
        logger.info(f"Stats records processed: {final_report.stats_records_processed}")
        logger.info(f"Terminology mappings applied: {final_report.terminology_mappings_applied}")
        logger.info(f"Identity conflicts resolved: {final_report.identity_conflicts_resolved}")
        logger.info(f"Overall data quality score: {final_report.data_quality_score:.3f}")
        
        if final_report.issues_found:
            logger.warning(f"Issues found: {len(final_report.issues_found)}")
            for issue in final_report.issues_found[:5]:
                logger.warning(f"  - {issue}")
        
        if final_report.recommendations:
            logger.info("Recommendations:")
            for rec in final_report.recommendations:
                logger.info(f"  - {rec}")
        
        session.close()
        
        return final_report
        
    except Exception as e:
        logger.error(f"Full historical data standardization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Run full standardization
        result = asyncio.run(run_full_standardization())
    else:
        # Run test on 2024 season only
        result = asyncio.run(test_historical_standardizer())
    
    if result:
        logger.info("Historical data standardization completed successfully")
    else:
        logger.error("Historical data standardization failed")
        sys.exit(1)
