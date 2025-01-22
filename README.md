# Credit Risk Analysis and Explainable AI Dashboard

This project develops a machine learning model to analyze credit risk and predict creditworthiness, integrating Explainable AI (XAI) for transparency and interpretability. The system includes an API for real-time predictions and a user-friendly dashboard for visualizing results.

## Features

- **Credit Risk Prediction:** Accurately predicts the probability of loan defaults using customer data and machine learning.
- **Explainable AI Integration:** Leverages SHAP (SHapley Additive exPlanations) to provide clear and interpretable insights into model predictions.
- **API Deployment:** Offers a REST API for seamless integration with other applications and real-time credit risk assessments.
- **Interactive Dashboard:** Built with Streamlit, the dashboard visualizes creditworthiness scores and SHAP explanations for easy understanding.
- **Cloud Deployment:** Designed for easy deployment on free cloud platforms like Heroku or Render.

## Project Structure
credit-risk-analysis/
├── data/
│   ├── raw/              # Raw input data (e.g., CSV, Excel)
│   └── processed/         # Cleaned and processed data
├── notebooks/            # Jupyter notebooks for EDA, model development, and experimentation
├── src/
│   ├── init.py        # Initialization file for Python modules
│   ├── config.py          # Configuration settings for the project (e.g., database connection, API keys)
│   ├── preprocessing/     # Data preprocessing pipeline
│   │   ├── data_cleaning.py  # Scripts for cleaning raw data (e.g., handling missing values, outliers)
│   │   └── feature_engineer.py # Feature engineering scripts (e.g., creating new features, transformations)
│   ├── models/            # Model training and evaluation
│   │   ├── train.py        # Model training script (e.g., model selection, hyperparameter tuning)
│   │   └── evaluate.py      # Model evaluation script (e.g., performance metrics, model comparison)
│   ├── api/               # API implementation
│   │   ├── app.py         # Flask/FastAPI application for serving predictions
│   │   └── requirements.txt # API dependencies
├── dashboard/           # Streamlit-based dashboard
│   ├── app.py            # Streamlit dashboard implementation
│   └── utils/            # Helper scripts for the dashboard
├── tests/                # Unit tests for the project
│   ├── test_data.py      # Tests for data cleaning and feature engineering
│   ├── test_model.py     # Tests for model training and evaluation
├── requirements.txt      # Project-wide Python dependencies
├── README.md            # Project documentation (this file)
└── setup.py