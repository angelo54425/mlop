import pickle
import numpy as np

def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def preprocess_input(input_data, scaler_path):
    scaler = pickle.load(open(scaler_path, "rb"))
    scaled_data = scaler.transform([input_data])
    return scaled_data

def predict(input_data, model_path, scaler_path):
    input_scaled = preprocess_input(input_data, scaler_path)
    model = load_model(model_path)
    prediction = model.predict(input_scaled)[0][0]
    label = "Hypertensive" if prediction >= 0.5 else "Non-Hypertensive"
    return label
