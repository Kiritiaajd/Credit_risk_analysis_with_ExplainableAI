import shap
import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load a sample dataset (since we don't have your actual credit dataset)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Generate a force plot and summary plot for interpretation
shap_plots = {
    "force_plot": shap.plots.force(shap_values[0], matplotlib=True, show=False),
    "summary_plot": shap.summary_plot(shap_values, X_test, feature_names=data.feature_names, show=False)
}
