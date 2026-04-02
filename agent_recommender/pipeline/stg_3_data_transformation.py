from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.data_transformation import DataTransformation
from agent_recommender import logger

STAGE_NAME = "Data Transformation Stage"

def main():
    config = ConfigurationManager()
    transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(config=transformation_config)
    data_transformation.load_data() \
                     .merge_historical_data() \
                     .split_data() \
                     .transform_train_test() \
                     .split_positive_negative() \
                     .save_transformed_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e