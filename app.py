from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# --- App Setup ---
app = FastAPI(
    title="Prediction API",
    description="Send feature values via POST and get predictions from the trained model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "best_svm_pipeline.pkl")
features_path = os.path.join(base_dir, "feature_columns.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

feature_columns_list = list(feature_columns)

class PredictionRequest(BaseModel):
    features: dict | list


@app.get("/")
def home():
    return {
        "message": "Prediction API",
        "endpoints": {
            "POST /predict": "Send features and get a prediction",
            "GET /features": "Get the list of expected feature names",
        },
    }


@app.get("/features")
def get_features():
    return {
        "feature_columns": feature_columns_list,
        "count": len(feature_columns_list),
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        if isinstance(request.features, dict):
            missing = [col for col in feature_columns_list if col not in request.features]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing features: {missing}. Use GET /features to see all required feature names.",
                )
            df = pd.DataFrame([request.features], columns=feature_columns_list)

        elif isinstance(request.features, list):
            if len(request.features) != len(feature_columns_list):
                raise HTTPException(
                    status_code=422,
                    detail=f"Expected {len(feature_columns_list)} features, got {len(request.features)}. "
                           f"Use GET /features to see all required feature names.",
                )
            df = pd.DataFrame([request.features], columns=feature_columns_list)

        else:
            raise HTTPException(
                status_code=422,
                detail="'features' must be a dict or a list.",
            )

        prediction = model.predict(df)
        pred_value = int(prediction[0])
        label = "Heart Disease" if pred_value == 1 else "No Heart Disease"

        result = {
            "prediction": pred_value,
            "label": label,
        }

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)[0]
                confidence = round(float(max(proba)) * 100, 2)
                result["confidence"] = f"{confidence}%"
                result["probabilities"] = {
                    "no_heart_disease": round(float(proba[0]) * 100, 2),
                    "heart_disease": round(float(proba[1]) * 100, 2),
                }
            except Exception:
                pass

        if "confidence" not in result and hasattr(model, "decision_function"):
            try:
                score = float(model.decision_function(df)[0])
                prob_positive = 1 / (1 + np.exp(-score))
                prob_negative = 1 - prob_positive
                confidence = round(max(prob_positive, prob_negative) * 100, 2)
                result["confidence"] = f"{confidence}%"
                result["probabilities"] = {
                    "no_heart_disease": round(prob_negative * 100, 2),
                    "heart_disease": round(prob_positive * 100, 2),
                }
            except Exception:
                pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
