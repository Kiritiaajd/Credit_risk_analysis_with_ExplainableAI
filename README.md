# Credit Risk Analysis with Explainable AI

## Overview
This project focuses on **Credit Risk Analysis** by developing machine learning models to assess customer creditworthiness. It integrates **Explainable AI (XAI)** techniques to enhance model transparency, ensuring stakeholders can interpret risk assessments. The project uses **XGBoost** for model training and **SHAP** for explainability.

## Objectives
- Develop predictive models for **credit risk assessment**.
- Improve **loan approval decision-making** with AI-driven insights.
- Enhance **model interpretability** using Explainable AI (XAI).
- Build **interactive dashboards** for real-time visualization of credit risk insights.

## Technologies Used
- **Programming:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn
- **Data Visualization:** Power BI, Matplotlib
- **Deployment:** Flask, AWS

## Dataset
- **Source:** Publicly available financial datasets (or proprietary bank datasets)
- **Features:** Customer demographics, income, credit history, loan status, and financial behavior
- **Target Variable:** Loan Default (0: No Default, 1: Default)

## Implementation Steps

### 1. Data Preprocessing
- Handle missing values using mean/median imputation.
- Encode categorical variables with **One-Hot Encoding**.
- Normalize numerical features using **Min-Max Scaling**.

### 2. Exploratory Data Analysis (EDA)
- Visualize correlations between features using **heatmaps**.
- Detect outliers and apply **Z-score filtering**.
- Analyze class distribution (default vs. non-default customers).

### 3. Model Training
- Train an **XGBoost classifier** to predict credit risk.
- Use **GridSearchCV** for hyperparameter tuning.
- Evaluate model performance using:
  - Accuracy
  - Precision, Recall, and F1-score
  - ROC-AUC Score

### 4. Explainability with SHAP
- Compute SHAP values to interpret feature importance.
- Generate **SHAP summary plots** to visualize impact.
- Create **force plots** to explain individual predictions.

### 5. Deployment & Visualization
- Deploy model using **Flask API**.
- Design **Power BI dashboards** for real-time credit risk monitoring.

## How to Run
### Prerequisites
Ensure you have the following installed:
```sh
pip install -r requirement.txt
```

### Run Model Training
```sh
python train_model.py
```

### Run Flask API
```sh
python app.py
```

### Access API
```sh
http://127.0.0.1:5000/predict
```


<img width="524" alt="image" src="https://github.com/user-attachments/assets/9d916ccc-67cb-465c-a1f5-7170fe41c87f" />
<img width="843" alt="image" src="https://github.com/user-attachments/assets/dc427a5a-e4bf-49a5-ae4e-df3512fc9f43" />




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
│   │   ├── data_cleaning.py  # Scripts for cleaning raw data (e.g., handling missing values, and outliers)
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
