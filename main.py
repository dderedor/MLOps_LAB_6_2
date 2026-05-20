from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "lr_cars.pkl")
model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Модель не загружена: {e}")

class CarFeatures(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "Car price prediction API"}

@app.post("/predict")
def predict(data: CarFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    try:
        input_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
