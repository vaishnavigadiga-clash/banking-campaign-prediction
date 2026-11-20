
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "notebooks"
    / "saved_model"
    / "banking_campaign_model.pkl"
)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
preprocessor = artifact["preprocessor"]

app = FastAPI(
    title="Banking Campaign Prediction API",
    description="Predicts the probability of a customer subscribing to a term deposit.",
    version="1.0"
)


class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float



def prepare_customer_data(customer_data):
    data = pd.DataFrame(customer_data)

    data = data.rename(columns={
        "emp_var_rate": "emp.var.rate",
        "cons_price_idx": "cons.price.idx",
        "cons_conf_idx": "cons.conf.idx",
        "nr_employed": "nr.employed"
    })

    data["previously_contacted"] = np.where(
        data["pdays"] == 999, 0, 1
    )

    data["previous_campaign_contacted"] = np.where(
        data["previous"] > 0, 1, 0
    )

    data["age_group"] = pd.cut(
        data["age"],
        bins=artifact["age_bins"],
        labels=artifact["age_labels"]
    )

    data = data[artifact["feature_columns"]]

    return data



def get_predictions(customers):
    prepared_data = prepare_customer_data(customers)

    processed_data = preprocessor.transform(prepared_data)

    probabilities = model.predict_proba(processed_data)[:, 1]

    predictions = (
        probabilities >= artifact["threshold"]
    ).astype(int)

    return probabilities, predictions


@app.get("/")
def home():
    return {
        "message": "Banking Campaign Prediction API is running!",
        "status": "Success"
    }


@app.post("/predict")
def predict(customer: Customer):
    customer_data = [customer.model_dump()]

    probabilities, predictions = get_predictions(customer_data)

    return {
        "subscription_probability": round(float(probabilities[0]), 4),
        "prediction": "Yes" if predictions[0] == 1 else "No",
        "threshold": artifact["threshold"],
        "model": artifact["model_name"]
    }


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]):
    customer_data = [
        customer.model_dump() for customer in customers
    ]

    probabilities, predictions = get_predictions(customer_data)

    results = []

    for i in range(len(customers)):
        results.append({
            "customer_number": i + 1,
            "subscription_probability": round(
                float(probabilities[i]), 4
            ),
            "prediction": "Yes" if predictions[i] == 1 else "No"
        })

    results.sort(
        key=lambda x: x["subscription_probability"],
        reverse=True
    )

    for rank, result in enumerate(results, start=1):
        result["priority"] = rank

    return {
        "total_customers": len(results),
        "results": results
    }
