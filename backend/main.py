from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
model = joblib.load("rf_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

print("Model loaded ✅")

class LoanRequest(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
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

    try:
        # =========================
        # RULE-BASED CHECKS (🔥 INDUSTRY LOGIC)
        # =========================
        if data.cibil_score < 600:
            return {
                "status": "Rejected",
                "reason": "Low CIBIL score",
                "approval_probability": 0,
                "rejection_probability": 100
            }

        if data.loan_amount > 4 * data.income_annum:
            return {
                "status": "Rejected",
                "reason": "Loan too high vs income",
                "approval_probability": 0,
                "rejection_probability": 100
            }

        # =========================
        # Convert input
        # =========================
        features = pd.DataFrame([data.dict()])

        # Encode
        for col, encoder in encoders.items():
            if col in features.columns:
                features[col] = encoder.transform(features[col])

        # =========================
        # Derived Features (🔥 SAME AS TRAINING)
        # =========================
        features["loan_to_income_ratio"] = (
            features["loan_amount"] / features["income_annum"]
        )

        features["total_assets"] = (
            features["residential_assets_value"] +
            features["commercial_assets_value"] +
            features["luxury_assets_value"] +
            features["bank_asset_value"]
        )

        features["assets_to_loan_ratio"] = (
            features["total_assets"] / features["loan_amount"]
        )

        # Ensure correct order
        features = features[feature_order]

        # =========================
        # Prediction
        # =========================
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]

        classes = model.classes_
        prob_map = dict(zip(classes, probs))

        APPROVED_CLASS = 0
        REJECTED_CLASS = 1

        return {
            "status": "Approved" if prediction == APPROVED_CLASS else "Rejected",
            "approval_probability": round(prob_map[APPROVED_CLASS] * 100, 2),
            "rejection_probability": round(prob_map[REJECTED_CLASS] * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}