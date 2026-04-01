from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.data_ingestion import DataIngestion
from agent_recommender import logger

STAGE_NAME = "Data Ingestion stage"


def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.load_raw_data()
    data_ingestion.handle_reentry_leads()
    data_ingestion.handle_orphan_brokers()
    data_ingestion.save_preprocessed_data()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e