# --- app/main.py ---
from fastapi import FastAPI
from tensorflow.keras.models import load_model
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

app = FastAPI(title="ML Prediction Service")
model = None
MODEL_LOCAL_PATH = "/app/models/model.h5"

# --- Configuration for MinIO ---
# Environment variables will be injected via Kubernetes Deployment
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio-service:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ml-models")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "model.h5")

def download_model_from_s3():
    """Downloads the model file from the MinIO S3 storage."""
    global model
    
    # Configure Boto3 client to connect to MinIO
    s3_config = Config(
        signature_version='s3v4',
        retries={
            'max_attempts': 5,
            'mode': 'standard'
        }
    )
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=s3_config,
    )

    # Ensure the local directory exists before download
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    
    try:
        print(f"INFO: Attempting to download {S3_MODEL_KEY} from bucket {S3_BUCKET_NAME} at {S3_ENDPOINT_URL}...")
        
        # Download the file
        s3_client.download_file(
            S3_BUCKET_NAME, 
            S3_MODEL_KEY, 
            MODEL_LOCAL_PATH
        )
        print(f"INFO: Model downloaded successfully to {MODEL_LOCAL_PATH}.")
        
        # Load the model using TensorFlow/Keras
        model = load_model(MODEL_LOCAL_PATH, compile=False)
        print("INFO: Model successfully loaded into memory.")
        
    except ClientError as e:
        print(f"WARNING: Could not load model from S3. Prediction endpoint will fail. Error: {e}")
        model = None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during model loading: {e}")
        model = None


@app.on_event("startup")
async def startup_event():
    """Execute model download and loading on application startup."""
    download_model_from_s3()

@app.get("/health", tags=["Monitoring"])
def get_health():
    """Health check endpoint, also indicates if the model is loaded."""
    if model is None:
        return {"status": "SUCCESS", "model_loaded": False}
    return {"status": "SUCCESS", "model_loaded": True}

@app.post("/predict", tags=["Prediction"])
def predict(data: list[float]):
    """Prediction endpoint: expects a list of 10 floats."""
    if model is None:
        return {"status": "ERROR", "message": "Model not loaded. Check startup logs."}
    
    # Simple prediction logic (assuming the input is a list of 10 features)
    # The dummy model expects an array of shape (1, 10)
    import numpy as np
    input_array = np.array([data])
    
    prediction = model.predict(input_array).tolist()
    
    return {"status": "SUCCESS", "prediction": prediction[0][0]}

# To run locally with uvicorn:
# uvicorn app.main:app --reload
