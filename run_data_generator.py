
from data_generator.pipeline.generator_pipeline import TrainingPipeline
from data_generator.utils.utils import print_pipeline_status


def main():
    """Main entry point for the synthetic data generator"""
    try:
        # Initialize pipeline
        print_pipeline_status("Initializing Training Pipeline", "INFO")
        pipeline = TrainingPipeline()
        
        # Run pipeline
        print_pipeline_status("Starting data generation", "INFO")
        artifact = pipeline.run()
        
        # Run validation
        print_pipeline_status("Running validation checks", "INFO")
        pipeline.run_validation(artifact)
        
        print_pipeline_status("Pipeline completed successfully!", "SUCCESS")
        
    except Exception as e:
        print_pipeline_status(f"Pipeline failed: {str(e)}", "ERROR")
        raise


if __name__ == "__main__":
    main()