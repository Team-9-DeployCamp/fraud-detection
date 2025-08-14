from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
from typing import List, Dict
from util import load_config, print_debug

# Load config
cfg = load_config()

# Setup MLflow
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

app = FastAPI(
    title="Fraud Detection API with MLflow",
    description="API for predicting credit card fraud using MLflow models",
    version="1.0.0"
)

class TransactionFeatures(BaseModel):
    amount: float
    merchant_type_electronics: int = 0
    merchant_type_groceries: int = 0
    merchant_type_others: int = 0
    merchant_type_travel: int = 0
    device_type_mobile: int = 0
    device_type_tablet: int = 0

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    model_info: Dict

# Global model variable
model = None
model_info = {}

def load_best_model():
    """Load the best model from MLflow experiment"""
    global model, model_info
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
        
        # Search for best run based on F1 score
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.valid_f1 DESC"],
            max_results=1
        )
        
        if len(runs) == 0:
            raise Exception("No runs found in MLflow experiment")
        
        best_run = runs.iloc[0]
        run_id = best_run['run_id']
        
        # Load model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Store model info
        model_info = {
            "run_id": run_id,
            "model_variant": best_run.get('tags.model_variant', 'unknown'),
            "f1_score": float(best_run['metrics.valid_f1']),
            "accuracy": float(best_run['metrics.valid_accuracy']),
            "model_uri": model_uri
        }
        
        print_debug(f"✓ Model loaded: {model_info['model_variant']}")
        print_debug(f"✓ F1-Score: {model_info['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        print_debug(f"✗ Error loading model from MLflow: {e}")
        return False

def load_specific_model(run_id: str):
    """Load specific model by run_id"""
    global model, model_info
    
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get run info
        client = MlflowClient(cfg["mlflow"]["tracking_uri"])
        run = client.get_run(run_id)
        
        model_info = {
            "run_id": run_id,
            "model_variant": run.data.tags.get('model_variant', 'unknown'),
            "f1_score": float(run.data.metrics.get('valid_f1', 0.0)),
            "accuracy": float(run.data.metrics.get('valid_accuracy', 0.0)),
            "model_uri": model_uri
        }
        
        print_debug(f"✓ Specific model loaded: {model_info['model_variant']}")
        return True
        
    except Exception as e:
        print_debug(f"✗ Error loading specific model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_best_model()
    if not success:
        print_debug("⚠ Warning: Failed to load model from MLflow")

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API with MLflow",
        "model_loaded": model is not None,
        "model_info": model_info if model else None
    }

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy", 
        "model_loaded": True,
        "model_info": model_info
    }

@app.get("/models")
def list_available_models():
    """List all available models from MLflow"""
    try:
        experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.valid_f1 DESC"]
        )
        
        models = []
        for _, run in runs.iterrows():
            models.append({
                "run_id": run['run_id'],
                "model_variant": run.get('tags.model_variant', 'unknown'),
                "f1_score": float(run.get('metrics.valid_f1', 0.0)),
                "accuracy": float(run.get('metrics.valid_accuracy', 0.0)),
                "start_time": run['start_time']
            })
        
        return {"models": models}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/switch_model/{run_id}")
def switch_model(run_id: str):
    """Switch to a specific model"""
    success = load_specific_model(run_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model with run_id {run_id} not found")
    
    return {
        "message": f"Successfully switched to model {run_id}",
        "model_info": model_info
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(features: TransactionFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame (sesuai format training)
        input_data = pd.DataFrame([features.dict()])
        
        # Ensure column order matches training data
        expected_columns = cfg["features"]  # dari config.yaml
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of fraud
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            model_info=model_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
def predict_batch(transactions: List[TransactionFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([t.dict() for t in transactions])
        
        # Ensure column order
        expected_columns = cfg["features"]
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk_level = "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High"
            results.append({
                "transaction_id": i,
                "prediction": int(pred),
                "probability": float(prob),
                "risk_level": risk_level
            })
        
        return {
            "results": results,
            "model_info": model_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)