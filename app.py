from fastapi import FastAPI
import joblib
import numpy as np

import os

app = FastAPI()

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")

model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "Student CGPA API 🚀"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    pred = model.predict(features)
    return {"cgpa": pred.tolist()}

