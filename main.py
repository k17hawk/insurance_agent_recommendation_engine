from agent_recommender.pipeline.stg_1_data_pipeline import main as run_data_ingestion
from agent_recommender.pipeline.stg_2_data_validation import main as run_data_validation
from agent_recommender import logger


if __name__ == "__main__":
    try:
        # Stage 1: Data Ingestion
        logger.info("=" * 50)
        logger.info("Running Stage 1: Data Ingestion")
        logger.info("=" * 50)
        run_data_ingestion()
        
        # Stage 2: Data Validation
        logger.info("=" * 50)
        logger.info("Running Stage 2: Data Validation")
        logger.info("=" * 50)
        run_data_validation()
        
        logger.info("All stages completed successfully!")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise e