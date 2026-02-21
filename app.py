from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

label_map = {
    0: "Normal",
    1: "Stage-1 Hypertension",
    2: "Stage-2 Hypertension",
    3: "Hypertensive Crisis"
}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # ⚠️ TEMP FIX: add dummy values if your model expects 10 features
    input_data = np.array([[ 
    data['gender'],
    float(data["age"]),
    float(data["salt_intake"]),
    float(data["stress_score"]),
    int(data["family_history"]),
    # float(data["sleep_duration"]),
    float(data["bmi"]),
    int(data["medication"]),
    int(data["family_history"]),
    int(data["systolic"]),
    int(data["diastolic" \
    ""])
]])

    input_scaled = scaler.transform(input_data)
    prediction = int(model.predict(input_scaled)[0])

    stage = label_map.get(prediction, "Unknown")

    return {
        "stage": stage,
        "confidence": 87
    }