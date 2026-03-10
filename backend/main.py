from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("loan_model.pkl")

print("Model classes:", model.classes_)
print("Classes:", model.classes_)

class LoanRequest(BaseModel):
    no_of_dependents: int
    education: int
    self_employed: int
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.post("/predict")
def predict(data: LoanRequest):

    features = pd.DataFrame([{
    "no_of_dependents": data.no_of_dependents,
    "education": data.education,
    "self_employed": data.self_employed,
    "income_annum": data.income_annum,
    "loan_amount": data.loan_amount,
    "loan_term": data.loan_term,
    "cibil_score": data.cibil_score,
    "residential_assets_value": data.residential_assets_value,
    "commercial_assets_value": data.commercial_assets_value,
    "luxury_assets_value": data.luxury_assets_value,
    "bank_asset_value": data.bank_asset_value
}])
    print("Example proba:", model.predict_proba(features))

    # print("Features sent to model:", features)

    # prediction = model.predict(features)

    # probability = model.predict_proba(features)[0][1]

    # status = "Approved" if prediction[0] == 1 else "Rejected"

    # return {
    # "status": status,
    # "approval_probability": round(probability * 100, 2)
    # }
    prediction = model.predict(features)

    probs = model.predict_proba(features)[0]

    approval_probability = probs[0] * 100

    status = "Approved" if prediction[0] == 0 else "Rejected"

    return {
        "status": status,
        "approval_probability": round(approval_probability, 2)
    }