# --- app/main.py ---
from fastapi import FastAPI
import tensorflow as tf
import numpy as np

# Initialize the FastAPI application
app = FastAPI()

# Global variable to hold the loaded model
model = None

# Path to the model file, assumed to be mounted/copied inside the container
MODEL_PATH = "models/model.h5"


# Function to load the model before the application starts
@app.on_event("startup")
def load_model():
    # Using the global keyword to modify the global 'model' variable
    global model
    try:
        # Load the model using Keras/TensorFlow utility
        model = tf.keras.models.load_model(MODEL_PATH)
        print("INFO: TensorFlow model loaded successfully.")
    except Exception as e:
        # If the model file is not found (normal for the first build) or corrupted
        print(f"WARNING: Could not load model from {MODEL_PATH}. Prediction endpoint will fail. Error: {e}")
        # In a real MLOps environment, the service might exit here, 
        # but for dev/test, we let it run.


# Health check endpoint for Kubernetes Liveness and Readiness Probes
@app.get("/health")
def health_check():
    # Checks if the model is loaded before declaring the service healthy
    if model:
        return {"status": "SUCCESS", "model_loaded": True}
    else:
        # If the model failed to load, return an error status
        return {"status": "ERROR", "model_loaded": False}


# Prediction endpoint (to be completed later)
@app.post("/predict")
def predict_data(data: dict):
    # This function will handle data preprocessing, prediction, 
    # and call the MLSolid gRPC service for logging/storage.
    if model is None:
        return {"status": "ERROR", "message": "Model not loaded. Cannot predict."}
    
    # Placeholder for the prediction logic
    return {"status": "SUCCESS", "prediction_result": "Pending implementation"}
