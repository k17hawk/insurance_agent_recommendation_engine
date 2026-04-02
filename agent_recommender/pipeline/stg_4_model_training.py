from agent_recommender.config.configuration import ConfigurationManager
from agent_recommender.components.model_training import ModelTraining
from agent_recommender import logger

STAGE_NAME = "Model Training Stage"

def main():
    config = ConfigurationManager()
    training_config = config.get_model_training_config()
    
    trainer = ModelTraining(config=training_config)
    
    # Step 1: Load data
    trainer.load_data()
    
    # Step 2: Filter columns (ensure all features exist and are numeric)
    trainer.filter_columns()
    
    # Step 3: Create dataloaders
    trainer.create_dataloaders()
    
    # Step 4: Build model
    trainer.build_model()
    
    # Step 5: Setup loss, optimizer, scheduler
    trainer.setup_training()
    
    # Step 6: Train the model (returns model and history)
    model, history = trainer.train()
    
    # Step 7: Find optimal threshold on validation set
    trainer.find_optimal_threshold()
    
    # Step 8: Evaluate on test set
    trainer.evaluate()
    
    # Step 9: Save final model and config
    trainer.save_model()
    
    logger.info(f"Training completed. Best model saved to {training_config.model_dir}")
    
    return model, history

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e