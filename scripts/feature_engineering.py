import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(file_path):
    """
    Preprocesses the input CSV file by handling missing values, encoding categorical features,
    and scaling numerical features.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Handling missing values in 'loan_int_rate' by filling with median
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
    
    # Detecting outliers using IQR method for 'loan_int_rate'
    Q1 = df['loan_int_rate'].quantile(0.25)
    Q3 = df['loan_int_rate'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identifying and printing the number of outliers
    outliers = df[(df['loan_int_rate'] < lower_bound) | (df['loan_int_rate'] > upper_bound)]
    print(f"Number of outliers in 'loan_int_rate': {outliers.shape[0]}")
    
    # One-hot encoding categorical features
    categorical_features = ['cb_person_default_on_file', 'person_home_ownership', 'loan_intent']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Feature Engineering: Creating new features
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['loan_to_income_percent'] = df['loan_percent_income'] * 100
    df['employment_stability'] = df['person_emp_length'] / df['cb_person_cred_hist_length']
    
    # Scaling numerical features
    numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                          'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                          'income_to_loan_ratio', 'loan_to_income_percent', 'employment_stability']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

if __name__ == "__main__":
    file_path = 'C:\\Github\\Credit_risk_analysis_with_ExplainableAI\\data\\raw\\credit_risk_dataset.csv'
    processed_df = preprocess_data(file_path)
    print("Data preprocessing and feature engineering completed.")