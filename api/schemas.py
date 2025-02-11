# API request and response schemas

from pydantic import BaseModel, conint, confloat
from typing import Literal

# schema for input validation
class CreditRiskInput(BaseModel):
    person_age: conint(ge=18, le=100)   # Age between 18 and 100
    person_income: confloat(ge=0)  # Income must be non-negative
    person_emp_length: confloat(ge=0, le=50)  # Employment length between 0 and 50 years
    loan_amnt: confloat(ge=100, le=1000000)  # Loan amount between 100 and 1M
    loan_int_rate: confloat(ge=0, le=100)  # Interest rate between 0% and 100%
    loan_status: conint(ge=0, le=1)  # 0 for no default, 1 for default
    loan_percent_income: confloat(ge=0, le=1)  # Percentage of income used for the loan
    cb_person_default_on_file: Literal["Y", "N"]  # Whether the person has defaulted before
    cb_person_cred_hist_length: conint(ge=0, le=50)  # Credit history length in years
    
    # One-hot encoded categorical variables
    person_home_ownership_OTHER: conint(ge=0, le=1)
    person_home_ownership_OWN: conint(ge=0, le=1)
    person_home_ownership_RENT: conint(ge=0, le=1)
    loan_intent_DEBTCONSOLIDATION: conint(ge=0, le=1)
    loan_intent_EDUCATION: conint(ge=0, le=1)
    loan_intent_HOMEIMPROVEMENT: conint(ge=0, le=1)
    loan_intent_MEDICAL: conint(ge=0, le=1)
    loan_intent_PERSONAL: conint(ge=0, le=1)
    loan_intent_VENTURE: conint(ge=0, le=1)
