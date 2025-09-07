"""
Run Full Historical Data Standardization (2020-2024)

This script runs the complete historical data standardization process
across all NFL seasons from 2020 to 2024.
"""

import asyncio
import logging
import sys
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from historical_data_standardizer import HistoricalDataStandardizer
from database_models import Base
from config_manager import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_standardization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def run_full_standardization():
    """Run standardization for all seasons 2020-2024"""
    
    logger.info("=" * 80)
    logger.info("STARTING FULL HISTORICAL DATA STANDARDIZATION (2020-2024)")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    # Database setup
    config = get_config()
    db_path = "nfl_predictions.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    
    # Target seasons
    seasons = [2020, 2021, 2022, 2023, 2024]
    
    total_seasons = len(seasons)
    completed_seasons = 0
    failed_seasons = []
    
    standardization_results = {}
    
    try:
        with Session() as session:
            standardizer = HistoricalDataStandardizer(session)
            
            for season in seasons:
                logger.info(f"\n{'='*60}")
                logger.info(f"PROCESSING SEASON {season} ({completed_seasons + 1}/{total_seasons})")
                logger.info(f"{'='*60}")
                
                season_start_time = datetime.now()
                
                try:
                    # Run standardization for this season
                    result = await standardizer.standardize_season(season)
                    
                    standardization_results[season] = result
                    completed_seasons += 1
                    
                    season_duration = datetime.now() - season_start_time
                    
                    logger.info(f"✅ Season {season} completed successfully")
                    logger.info(f"   Duration: {season_duration}")
                    logger.info(f"   Records processed: {result.get('records_processed', 0)}")
                    logger.info(f"   Records standardized: {result.get('records_standardized', 0)}")
                    logger.info(f"   Quality score: {result.get('data_quality_score', 0):.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Season {season} failed: {e}")
                    failed_seasons.append(season)
                    standardization_results[season] = {"error": str(e)}
                
                # Progress update
                progress = ((completed_seasons + len(failed_seasons)) / total_seasons) * 100
                logger.info(f"\nOverall Progress: {progress:.1f}% ({completed_seasons} completed, {len(failed_seasons)} failed)")
    
    except Exception as e:
        logger.error(f"Critical error in standardization process: {e}")
        return False
    
    # Final summary
    total_duration = datetime.now() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("FULL STANDARDIZATION COMPLETE")
    logger.info("=" * 80)
    
    logger.info(f"Total Duration: {total_duration}")
    logger.info(f"Seasons Processed: {total_seasons}")
    logger.info(f"Successful: {completed_seasons}")
    logger.info(f"Failed: {len(failed_seasons)}")
    
    if failed_seasons:
        logger.warning(f"Failed Seasons: {failed_seasons}")
    
    # Detailed results
    logger.info("\nDETAILED RESULTS:")
    logger.info("-" * 50)
    
    total_records_processed = 0
    total_records_standardized = 0
    
    for season, result in standardization_results.items():
        if "error" not in result:
            records_processed = result.get('records_processed', 0)
            records_standardized = result.get('records_standardized', 0)
            quality_score = result.get('data_quality_score', 0)
            
            total_records_processed += records_processed
            total_records_standardized += records_standardized
            
            logger.info(f"Season {season}:")
            logger.info(f"  Records Processed: {records_processed:,}")
            logger.info(f"  Records Standardized: {records_standardized:,}")
            logger.info(f"  Quality Score: {quality_score:.2f}")
            logger.info(f"  Success Rate: {(records_standardized/records_processed*100):.1f}%" if records_processed > 0 else "  Success Rate: 0%")
        else:
            logger.info(f"Season {season}: FAILED - {result['error']}")
    
    logger.info(f"\nTOTAL SUMMARY:")
    logger.info(f"  Total Records Processed: {total_records_processed:,}")
    logger.info(f"  Total Records Standardized: {total_records_standardized:,}")
    logger.info(f"  Overall Success Rate: {(total_records_standardized/total_records_processed*100):.1f}%" if total_records_processed > 0 else "  Overall Success Rate: 0%")
    
    # Success criteria
    success_rate = (completed_seasons / total_seasons) * 100
    
    if success_rate >= 80:
        logger.info(f"🎉 STANDARDIZATION SUCCESSFUL ({success_rate:.1f}% completion rate)")
        return True
    else:
        logger.warning(f"⚠️  STANDARDIZATION PARTIALLY SUCCESSFUL ({success_rate:.1f}% completion rate)")
        return False

async def validate_standardized_data():
    """Validate the standardized data quality"""
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING STANDARDIZED DATA")
    logger.info("=" * 60)
    
    config = get_config()
    db_path = "nfl_predictions.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    
    try:
        with Session() as session:
            standardizer = HistoricalDataStandardizer(session)
            
            # Get standardization summary
            from sqlalchemy import text
            
            summary_query = text("""
                SELECT 
                    season,
                    records_processed,
                    records_standardized,
                    data_quality_score,
                    player_mappings_created,
                    terminology_mappings_created,
                    created_at
                FROM historical_data_standardization
                ORDER BY season
            """)
            
            results = session.execute(summary_query).fetchall()
            
            if not results:
                logger.warning("No standardization records found in database")
                return False
            
            logger.info("Standardization Summary from Database:")
            logger.info("-" * 40)
            
            total_processed = 0
            total_standardized = 0
            
            for result in results:
                season = result[0]
                processed = result[1]
                standardized = result[2]
                quality = result[3]
                player_mappings = result[4]
                term_mappings = result[5]
                created = result[6]
                
                total_processed += processed
                total_standardized += standardized
                
                logger.info(f"Season {season}:")
                logger.info(f"  Processed: {processed:,}")
                logger.info(f"  Standardized: {standardized:,}")
                logger.info(f"  Quality Score: {quality:.2f}")
                logger.info(f"  Player Mappings: {player_mappings}")
                logger.info(f"  Term Mappings: {term_mappings}")
                logger.info(f"  Completed: {created}")
            
            logger.info(f"\nOverall Totals:")
            logger.info(f"  Total Processed: {total_processed:,}")
            logger.info(f"  Total Standardized: {total_standardized:,}")
            logger.info(f"  Success Rate: {(total_standardized/total_processed*100):.1f}%" if total_processed > 0 else "  Success Rate: 0%")
            
            # Validate data integrity
            integrity_query = text("""
                SELECT COUNT(*) as standardized_records
                FROM player_game_stats
                WHERE standardized_player_id IS NOT NULL
                AND standardized_stats IS NOT NULL
            """)
            
            integrity_result = session.execute(integrity_query).fetchone()
            standardized_count = integrity_result[0] if integrity_result else 0
            
            logger.info(f"\nData Integrity Check:")
            logger.info(f"  Records with standardized data: {standardized_count:,}")
            
            if standardized_count >= total_standardized * 0.95:  # 95% threshold
                logger.info("✅ Data integrity validation PASSED")
                return True
            else:
                logger.warning("⚠️  Data integrity validation FAILED")
                return False
                
    except Exception as e:
        logger.error(f"Error validating standardized data: {e}")
        return False

async def main():
    """Main execution function"""
    
    try:
        # Run full standardization
        standardization_success = await run_full_standardization()
        
        if standardization_success:
            # Validate results
            validation_success = await validate_standardized_data()
            
            if validation_success:
                logger.info("\n🎉 FULL STANDARDIZATION AND VALIDATION COMPLETE!")
                logger.info("The historical NFL data (2020-2024) has been successfully standardized.")
                logger.info("You can now use the enhanced prediction system with improved data quality.")
            else:
                logger.warning("\n⚠️  Standardization completed but validation failed.")
                logger.warning("Please review the data integrity issues before proceeding.")
        else:
            logger.error("\n❌ Standardization failed.")
            logger.error("Please review the errors and retry failed seasons individually.")
    
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
