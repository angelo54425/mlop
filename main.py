from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from src.model import create_model
from src.preprocessing import preprocessing
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hypertension.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Input features expected for prediction
class PredictionInput(BaseModel):
    Age: float
    Weight: float
    BP_History: int
    Medication: int
    Family_History: int
    Exercise_Level: int
    Smoking_Status: int
    Cholesterol_Level: int
    Salt_Intake_Level: int
    Alcohol_Consumption: int


def load_trained_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@app.get("/")
def home():
    return {"message": "Hypertension Prediction API"}

@app.post("/predict/")
async def predict(input_data: PredictionInput):
    try:
        # Convert input to array
        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)

        features = np.array([
    input_data.Age,
    input_data.Weight,
    input_data.BP_History,
    input_data.Medication,
    input_data.Family_History,
    input_data.Exercise_Level,
    input_data.Smoking_Status,
    input_data.Cholesterol_Level,
    input_data.Salt_Intake_Level,
    input_data.Alcohol_Consumption
]).reshape(1, -1)


        # Load model and scaler
        model, scaler = load_trained_model()

        # Scale and predict
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0][0]
        label = "Hypertensive" if pred >= 0.5 else "Non-Hypertensive"

        return {"prediction": label, "probability": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/retrain/")
async def retrain(file: UploadFile = File(...)):
    try:
        temp_path = os.path.join(BASE_DIR, "temp.csv")
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        # Preprocess new dataset
        trainX, testX, trainY, testY, scaler = preprocessing(temp_path)

        # Create and train new model
        input_shape = trainX.shape[1]
        model, callbacks = create_model(input_shape)
        model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, callbacks=callbacks, verbose=1)

        # Evaluate
        preds = (model.predict(testX) >= 0.5).astype(int)
        acc = accuracy_score(testY, preds)

        # Save
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        os.remove(temp_path)

        return {"status": "Model retrained", "accuracy": acc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")

@app.post("/fine_tune/")
async def fine_tune(file: UploadFile = File(...), epochs: int = 5):
    try:
        temp_path = os.path.join(BASE_DIR, "temp.csv")
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        trainX, testX, trainY, testY, _ = preprocessing(temp_path)

        model, _ = load_trained_model()

        model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, verbose=1)

        preds = (model.predict(testX) >= 0.5).astype(int)
        acc = accuracy_score(testY, preds)

        # Save updated model only
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        os.remove(temp_path)

        return {"status": "Model fine-tuned", "accuracy": acc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fine-tuning error: {str(e)}")
