# Unit tests for API
import unittest
import json
from app import app

class CreditRiskTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_valid_input(self):
        valid_input = {
            "person_age": 35,
            "person_income": 50000,
            "person_emp_length": 5,
            "loan_amnt": 15000,
            "loan_int_rate": 12.5,
            "loan_status": 0,
            "loan_percent_income": 0.3,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 10,
            "person_home_ownership_OTHER": 0,
            "person_home_ownership_OWN": 1,
            "person_home_ownership_RENT": 0,
            "loan_intent_DEBTCONSOLIDATION": 1,
            "loan_intent_EDUCATION": 0,
            "loan_intent_HOMEIMPROVEMENT": 0,
            "loan_intent_MEDICAL": 0,
            "loan_intent_PERSONAL": 0,
            "loan_intent_VENTURE": 0
        }
        response = self.app.post('/predict', data=json.dumps(valid_input), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json)

    def test_predict_invalid_input(self):
        invalid_input = {"person_age": 10}  # Invalid age
        response = self.app.post('/predict', data=json.dumps(invalid_input), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

if __name__ == "__main__":
    unittest.main()