from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.model_training import ModelTraining
from agent_recommender import logger
from agent_recommender.utils.utility import create_directories

STAGE_NAME = "Model Training stage"


def main():
    config = ConfigurationManager()
    model_training_config = config.get_model_training_config()
    
    # Create directories
    create_directories([model_training_config.model_dir])
    create_directories([model_training_config.reports_dir])
    
    model_training = ModelTraining(config=model_training_config)
    
    model_training.load_data() \
        .filter_columns() \
        .create_dataloaders() \
        .build_model() \
        .setup_training() \
        .train() \
        .find_optimal_threshold() \
        .evaluate() \
        .save_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e