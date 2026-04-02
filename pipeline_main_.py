from agent_recommender.pipeline.stg_1_data_ingestion import main as run_data_ingestion
from agent_recommender.pipeline.stg_2_data_validation import main as run_data_validation
from agent_recommender.pipeline.stg_3_data_transformation import main as run_data_transformation
from agent_recommender.pipeline.stg_4_model_training import main as run_model_training
from agent_recommender.pipeline.stg_5_model_eval import main as run_model_evaluation
from agent_recommender.pipeline.stg_6_model_push import main as run_model_push
from agent_recommender import logger

if __name__ == "__main__":
    try:
        logger.info("=" * 50)
        logger.info("Running Stage 1: Data Ingestion")
        logger.info("=" * 50)
        run_data_ingestion()
        
        logger.info("=" * 50)
        logger.info("Running Stage 2: Data Validation")
        logger.info("=" * 50)
        run_data_validation()
        
        logger.info("=" * 50)
        logger.info("Running Stage 3: Data Transformation")
        logger.info("=" * 50)
        run_data_transformation()
        
        logger.info("=" * 50)
        logger.info("Running Stage 4: Model Training")
        logger.info("=" * 50)
        run_model_training()
        
        logger.info("=" * 50)
        logger.info("Running Stage 5: Model Evaluation")
        logger.info("=" * 50)
        run_model_evaluation()
        
        logger.info("=" * 50)
        logger.info("Running Stage 6: Model Push")
        logger.info("=" * 50)
        run_model_push()
        
        logger.info("=" * 50)
        logger.info("All stages completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise e