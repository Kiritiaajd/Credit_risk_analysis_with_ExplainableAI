import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = xgb.XGBClassifier()
model.load_model(r'C:\Github\Credit_risk_analysis_with_ExplainableAI\models\xgb_model.json')

# Feature columns
feature_columns = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
    'loan_status', 'loan_percent_income', 'cb_person_default_on_file',
    'cb_person_cred_hist_length', 'person_home_ownership_OTHER', 'person_home_ownership_OWN',
    'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]

# SHAP explanation
def get_shap_values(input_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    return shap_values

@app.route('/explain', methods=['POST'])
def explain_model():
    try:
        # Get input data from the request
        data = request.get_json()
        input_df = pd.DataFrame([data], columns=feature_columns)

        # Check if input has correct number of features
        if input_df.shape[1] != len(feature_columns):
            return jsonify({"error": "Invalid number of features."}), 400

        # Get SHAP values
        shap_values = get_shap_values(input_df)

        return jsonify({
            'shap_values': shap_values.tolist(),
            'explanation': 'SHAP values indicate feature importance and contribution to the prediction.'
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
