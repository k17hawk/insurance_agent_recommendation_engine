from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.model_evaluation import ModelEvaluation
from agent_recommender import logger
from agent_recommender.utils.utility import create_directories

STAGE_NAME = "Model Evaluation stage"


def main():
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()
    
    # Create directories
    create_directories([model_evaluation_config.reports_dir])
    
    model_evaluation = ModelEvaluation(config=model_evaluation_config)
    
    model_evaluation.load_model() \
        .load_test_data() \
        .generate_predictions() \
        .calculate_metrics() \
        .plot_confusion_matrix() \
        .plot_roc_curve() \
        .plot_pr_curve() \
        .plot_prediction_distribution() \
        .generate_classification_report() \
        .save_metrics()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e