<img width="943" alt="image" src="https://github.com/user-attachments/assets/71aa3601-d68b-45e3-9694-067476e8c168" />


## Project Structure

```bash
credit-risk-analysis/
├── data/
│   ├── raw/              # Raw input data (e.g., CSV, Excel)
│   └── processed/        # Cleaned and processed data
├── notebooks/            # Jupyter notebooks for EDA, model development, and experimentation
├── src/
│   ├── __init__.py       # Initialization file for Python modules
│   ├── config.py         # Configuration settings (e.g., database connection, API keys)
│   ├── preprocessing/    # Data preprocessing pipeline
│   │   ├── data_cleaning.py  # Scripts for cleaning raw data (e.g., handling missing values, outliers)
│   │   └── feature_engineering.py # Feature engineering scripts (e.g., creating new features)
│   ├── models/           # Model training and evaluation
│   │   ├── train.py      # Model training script (e.g., model selection, hyperparameter tuning)
│   │   └── evaluate.py   # Model evaluation script (e.g., performance metrics, model comparison)
│   ├── api/              # API implementation
│   │   ├── app.py        # Flask/FastAPI application for serving predictions
│   │   └── requirements.txt # API dependencies
├── dashboard/            # Streamlit-based dashboard
│   ├── app.py            # Streamlit dashboard implementation
│   └── utils/            # Helper scripts for the dashboard
├── tests/                # Unit tests for the project
│   ├── test_data.py      # Tests for data cleaning and feature engineering
│   ├── test_model.py     # Tests for model training and evaluation
├── requirements.txt      # Project-wide Python dependencies
├── README.md             # Project documentation (this file)
└── setup.py              # Setup script for installing dependencies
