from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Setup MLflow - use environment variable for Docker containers
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
print_debug(f"🔧 Setting MLflow URI: {mlflow_uri}")
mlflow.set_tracking_uri(mlflow_uri)

app = FastAPI(
    title="Fraud Detection API with MLflow",
    description="API for predicting credit card fraud using MLflow models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
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
        """Load the best model from MLflow with robust fallback"""
        try:
            # Get experiment
            experiment_name = self.config["mlflow"]["experiment_name"]
            print_debug(f"🔍 Looking for experiment: {experiment_name}")
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if not experiment:
                print_debug("⚠ Experiment not found, using model from inference.py")
                self.model = self.inference_predictor.model
                self.model_info = {
                    "model_source": "inference_predictor",
                    "model_type": type(self.model).__name__
                }
                print_debug("✓ Model loaded from inference predictor")
                return True
            
            print_debug(f"✓ Found experiment ID: {experiment.experiment_id}")
            
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
            print_debug(f"🔍 Best run ID: {run_id}")
            
            # Try multiple approaches to load the model
            model_loaded = False
            
            # Method 1: Standard MLflow URI
            if not model_loaded:
                try:
                    model_uri = f"runs:/{run_id}/model"
                    print_debug(f"🔍 Trying standard URI: {model_uri}")
                    self.model = mlflow.sklearn.load_model(model_uri)
                    print_debug("✓ Model loaded from standard MLflow URI")
                    model_loaded = True
                except Exception as e:
                    print_debug(f"✗ Standard URI failed: {e}")
            
            # Method 2: Direct file path (for Docker containers)
            if not model_loaded:
                try:
                    # Try absolute path in container
                    artifact_path = f"/app/mlruns/{experiment.experiment_id}/{run_id}/artifacts/model"
                    print_debug(f"🔍 Trying container path: {artifact_path}")
                    if os.path.exists(artifact_path):
                        self.model = mlflow.sklearn.load_model(artifact_path)
                        print_debug("✓ Model loaded from container path")
                        model_loaded = True
                    else:
                        print_debug(f"✗ Container path not found: {artifact_path}")
                except Exception as e:
                    print_debug(f"✗ Container path failed: {e}")
            
            # Method 3: Relative path
            if not model_loaded:
                try:
                    artifact_path = f"./mlruns/{experiment.experiment_id}/{run_id}/artifacts/model"
                    print_debug(f"🔍 Trying relative path: {artifact_path}")
                    if os.path.exists(artifact_path):
                        self.model = mlflow.sklearn.load_model(artifact_path)
                        print_debug("✓ Model loaded from relative path")
                        model_loaded = True
                    else:
                        print_debug(f"✗ Relative path not found: {artifact_path}")
                except Exception as e:
                    print_debug(f"✗ Relative path failed: {e}")
            
            if model_loaded:
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
            else:
                print_debug("⚠ All MLflow methods failed, using inference fallback")
                raise Exception("All MLflow loading methods failed")
                
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
            "mlflow_uri": cfg["mlflow"]["tracking_uri"],
            "mlflow_env": os.getenv("MLFLOW_TRACKING_URI"),
            "current_mlflow_uri": mlflow.get_tracking_uri()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/debug/mlflow")
def debug_mlflow():
    """Debug endpoint to test MLflow connection and paths"""
    try:
        result = {
            "tracking_uri_config": cfg["mlflow"]["tracking_uri"],
            "tracking_uri_env": os.getenv("MLFLOW_TRACKING_URI"),
            "tracking_uri_current": mlflow.get_tracking_uri(),
            "experiment_name": cfg["mlflow"]["experiment_name"]
        }
        
        # Check if mlruns exists
        mlruns_paths = [
            "/app/mlruns",
            "./mlruns",
            "mlruns"
        ]
        
        for path in mlruns_paths:
            result[f"path_exists_{path.replace('/', '_').replace('.', 'dot')}"] = os.path.exists(path)
        
        # Try to get experiment
        try:
            experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
            if experiment:
                result["experiment_found"] = True
                result["experiment_id"] = experiment.experiment_id
                
                # List available runs
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=3
                )
                result["runs_count"] = len(runs)
                
                if len(runs) > 0:
                    result["sample_run_id"] = runs.iloc[0]["run_id"]
                    
                    # Check model paths
                    run_id = runs.iloc[0]["run_id"]
                    model_paths = [
                        f"/app/mlruns/{experiment.experiment_id}/{run_id}/artifacts/model",
                        f"./mlruns/{experiment.experiment_id}/{run_id}/artifacts/model",
                        f"mlruns/{experiment.experiment_id}/{run_id}/artifacts/model"
                    ]
                    
                    for path in model_paths:
                        result[f"model_path_exists_{path.replace('/', '_').replace('.', 'dot')}"] = os.path.exists(path)
            else:
                result["experiment_found"] = False
        except Exception as exp_error:
            result["experiment_error"] = str(exp_error)
        
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
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