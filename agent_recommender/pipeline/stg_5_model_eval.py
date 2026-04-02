from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.model_evaluation import ModelEvaluation
from agent_recommender import logger

STAGE_NAME = "Model Evaluation Stage"

def main():
    config = ConfigurationManager()
    eval_config = config.get_model_evaluation_config()
    evaluator = ModelEvaluation(config=eval_config)
    evaluator.evaluate()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e