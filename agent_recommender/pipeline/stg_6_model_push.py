from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.model_push import ModelPush
from agent_recommender import logger
from agent_recommender.utils.utility import create_directories

STAGE_NAME = "Model Push stage"


def main():
    config = ConfigurationManager()
    model_push_config = config.get_model_push_config()
    
    # Create directories
    create_directories([model_push_config.push_dir])
    
    model_push = ModelPush(config=model_push_config)
    
    model_push.push_model() \
        .create_latest_symlink() \
        .save_version_info() \
        .validate_push()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e