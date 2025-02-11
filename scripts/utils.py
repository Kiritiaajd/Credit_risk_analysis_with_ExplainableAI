import os
import logging
import pandas as pd
import numpy as np

# Function to check if a directory exists and create it if it doesn't
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory created: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")

# Function to load CSV data
def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file from {file_path}: {e}")
        raise

# Function to save a DataFrame to a CSV file
def save_csv_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise

# Function to handle missing values (imputation or removal)
def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        df = df.fillna(df.mean())
        logging.info("Missing values imputed using mean.")
    elif strategy == 'drop':
        df = df.dropna()
        logging.info("Missing values dropped.")
    else:
        logging.error("Invalid strategy for missing values. Choose 'mean' or 'drop'.")
        raise ValueError("Invalid strategy for missing values. Choose 'mean' or 'drop'.")
    return df

# Function to normalize/scale features (e.g., using MinMax scaling)
from sklearn.preprocessing import MinMaxScaler
def scale_features(df, feature_columns):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    logging.info(f"Features {feature_columns} scaled using MinMaxScaler.")
    return df

# Function to split a dataset into training and testing sets
from sklearn.model_selection import train_test_split
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Data split into training and testing sets. Test size: {test_size}")
    return X_train, X_test, y_train, y_test

# Function to log the model performance metrics
from sklearn.metrics import accuracy_score, classification_report
def log_model_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{report}")
