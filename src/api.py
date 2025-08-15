from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import List, Dict
import os

# Now import from src
from src.util import load_config, print_debug, pickle_load
from src.inference import FraudDetectionPredictor
import uvicorn

# Load config
cfg = load_config()

# Setup MLflow
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

app = FastAPI(
    title="Fraud Detection API with MLflow",
    description="API for predicting credit card fraud using MLflow models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")

class TransactionInput(BaseModel):
    amount: float
    merchant_type: str
    device_type: str

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    transaction_details: Dict

class MLflowFraudPredictor:
    """MLflow-based Fraud Detection Predictor that uses inference.py"""
    
    def __init__(self):
        self.config = load_config()
        self.model = None
        self.model_info = {}
        
        # Initialize inference predictor for preprocessing and encoding
        try:
            self.inference_predictor = FraudDetectionPredictor()
            print_debug("✓ Inference predictor initialized successfully")
        except Exception as e:
            print_debug(f"✗ Error initializing inference predictor: {e}")
            raise
    
    def load_best_model(self):
        """Load the best model from MLflow"""
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(self.config["mlflow"]["experiment_name"])
            
            if not experiment:
                print_debug("⚠ Experiment not found, using model from inference.py")
                # Use model from inference predictor
                self.model = self.inference_predictor.model
                self.model_info = {
                    "model_source": "inference_predictor",
                    "model_type": type(self.model).__name__
                }
                print_debug("✓ Model loaded from inference predictor")
                return True
            
            # Search for best run based on F1 score
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                order_by=["metrics.valid_f1 DESC"],
                max_results=1
            )
            
            if len(runs) == 0:
                print_debug("No runs found, using model from inference.py")
                self.model = self.inference_predictor.model
                self.model_info = {"model_source": "inference_fallback"}
                return True
            
            best_run = runs.iloc[0]
            run_id = best_run['run_id']
            
            # Load model from MLflow
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Store model info
            self.model_info = {
                "run_id": run_id,
                "model_variant": best_run.get('tags.model_variant', 'unknown'),
                "f1_score": float(best_run['metrics.valid_f1']),
                "accuracy": float(best_run['metrics.valid_accuracy']),
                "model_uri": model_uri,
                "model_source": "mlflow"
            }
            
            print_debug(f"✓ Model loaded from MLflow: {self.model_info['model_variant']}")
            print_debug(f"✓ F1-Score: {self.model_info['f1_score']:.4f}")
            
            return True
            
        except Exception as e:
            print_debug(f"✗ Error loading model from MLflow: {e}")
            print_debug("Trying fallback to inference predictor...")
            try:
                self.model = self.inference_predictor.model
                self.model_info = {
                    "model_source": "inference_fallback",
                    "model_type": type(self.model).__name__,
                    "error": str(e)
                }
                print_debug("✓ Model loaded from inference fallback")
                return True
            except Exception as inference_error:
                print_debug(f"✗ Inference fallback also failed: {inference_error}")
                return False

    
    def predict_single(self, amount, merchant_type, device_type):
        """Predict fraud for a single transaction using inference.py"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            print_debug(f"Input transaction: amount={amount}, merchant={merchant_type}, device={device_type}")
            
            # Use inference predictor for preprocessing and prediction
            # But replace the model with MLflow model if available
            original_model = self.inference_predictor.model
            self.inference_predictor.model = self.model
            
            # Use inference predictor's predict_single method
            result = self.inference_predictor.predict_single(amount, merchant_type, device_type)
            
            # Restore original model
            self.inference_predictor.model = original_model
            
            print_debug(f"Final result: {result}")
            return result
            
        except Exception as e:
            print_debug(f"✗ Error in predict_single: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            base_info = {
                'model_type': type(self.model).__name__ if self.model else 'Not loaded',
                'merchant_categories': list(self.inference_predictor.merchant_encoder.categories_[0]),
                'device_categories': list(self.inference_predictor.device_encoder.categories_[0]),
            }
            base_info.update(self.model_info)
            return base_info
        except Exception as e:
            return {
                'error': str(e),
                'model_type': 'Error loading info'
            }

# Initialize global predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    global predictor
    try:
        print_debug("🚀 Starting Fraud Detection API...")
        print_debug("✓ API startup completed successfully")
        # Initialize predictor lazily on first request to avoid blocking startup
    except Exception as e:
        print_debug(f"⚠ Warning: Failed during startup: {e}")

def get_predictor():
    """Get or initialize predictor lazily"""
    global predictor
    if predictor is None:
        try:
            print_debug("🔄 Initializing predictor...")
            predictor = MLflowFraudPredictor()
            success = predictor.load_best_model()
            if not success:
                print_debug("⚠ Warning: Failed to load any model")
            else:
                print_debug("✓ Predictor initialized successfully")
        except Exception as e:
            print_debug(f"⚠ Warning: Failed to initialize predictor: {e}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    return predictor

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API with MLflow",
        "status": "ready" if predictor and predictor.model else "model not loaded",
        "model_info": predictor.get_model_info() if predictor else None,
        "usage": {
            "prediction_endpoint": "/predict",
            "models_endpoint": "/models",
            "health_endpoint": "/health",
            "frontend_url": "/app",
            "example_request": {
                "amount": 100.50,
                "merchant_type": "electronics",
                "device_type": "mobile"
            }
        }
    }

@app.get("/app")
def frontend():
    """Serve the frontend application"""
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    frontend_file = os.path.join(frontend_dir, "index.html")
    
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")

@app.get("/health")
def health_check():
    try:
        # For health check, don't initialize the full predictor, just check basic connectivity
        return {
            "status": "healthy",
            "api_ready": True,
            "mlflow_uri": cfg["mlflow"]["tracking_uri"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/models")
def list_available_models():
    """List all available models from MLflow"""
    try:
        experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
        
        if not experiment:
            return {"models": [], "message": "No MLflow experiment found"}
        
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
        return {"models": [], "error": str(e)}

@app.options("/predict")
def predict_options():
    """Handle CORS preflight for prediction endpoint"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
        }
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionInput):
    """Predict fraud for a single transaction"""
    try:
        # Get or initialize predictor lazily
        current_predictor = get_predictor()
        
        result = current_predictor.predict_single(
            amount=transaction.amount,
            merchant_type=transaction.merchant_type,
            device_type=transaction.device_type
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        print_debug(f"API Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
def predict_batch(transactions: List[TransactionInput]):
    """Predict fraud for multiple transactions"""
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for i, transaction in enumerate(transactions):
            result = predictor.predict_single(
                amount=transaction.amount,
                merchant_type=transaction.merchant_type,
                device_type=transaction.device_type
            )
            result['transaction_id'] = i
            results.append(result)
        
        return {
            "results": results,
            "model_info": predictor.get_model_info(),
            "total_transactions": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)