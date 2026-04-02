from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.model_push import ModelPush
from agent_recommender import logger
from agent_recommender.utils.utility import create_directories

STAGE_NAME = "Model Push Stage"

def main():
    config = ConfigurationManager()
    push_config = config.get_model_push_config()
    create_directories([push_config.push_dir])
    pusher = ModelPush(config=push_config)
    pusher.push_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e