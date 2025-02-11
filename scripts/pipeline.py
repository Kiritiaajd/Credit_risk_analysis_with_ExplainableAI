import logging
from data_preprocessing import load_data, preprocess_data
from feature_engineering import engineer_features
from model_training import train_model, evaluate_model, save_model
from explainable_ai import explain_model

# Set up logging
logging.basicConfig(filename='end_to_end_pipeline.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Started end-to-end pipeline')

# Main function to run the full pipeline
def main():
    try:
        # Define paths and configurations
        data_file_path = r'C:\Github\Credit_risk_analysis_with_ExplainableAI\data\processed\credit_risk_processed.csv'
        model_json_path = r'C:\Github\Credit_risk_analysis_with_ExplainableAI\models\xgb_model.json'
        model_pkl_path = r'C:\Github\Credit_risk_analysis_with_ExplainableAI\models\xgb_model.pkl'

        # Load and preprocess data
        df = load_data(data_file_path)
        X, y = preprocess_data(df)

        # Feature engineering (add new features, transformations, etc.)
        X = engineer_features(X)
        logging.info("Feature engineering completed")

        # Train model
        model = train_model(X, y)
        logging.info("Model training completed")

        # Evaluate model performance
        evaluate_model(model, X, y)
        logging.info("Model evaluation completed")

        # Save the model
        save_model(model, model_json_path, model_pkl_path)
        logging.info(f"Model saved at {model_json_path} and {model_pkl_path}")

        # Model explanation (optional step for explainable AI)
        explain_model(model, X)
        logging.info("Model explanation completed")

    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
