from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import sys
import os
from .inference import FraudDetectionPredictor
from . import util as utils
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions using machine learning",
    version="1.0.0"
)

# Initialize predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor when the API starts"""
    global predictor
    try:
        predictor = FraudDetectionPredictor()
        print("✅ Fraud Detection API started successfully!")
        print("📊 Model loaded and ready for predictions")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        predictor = None

# Pydantic models for request/response
class TransactionData(BaseModel):
    amount: float = Field(..., description="Transaction amount", example=150.50)
    merchant_type: str = Field(..., description="Type of merchant", example="electronics")
    device_type: str = Field(..., description="Type of device used", example="mobile")

class BatchTransactionData(BaseModel):
    transactions: List[TransactionData] = Field(..., description="List of transactions to analyze")

class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="Whether the transaction is predicted as fraud")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    transaction_details: dict = Field(..., description="Original transaction details")

class BatchPredictionResponse(BaseModel):
    predictions: List[dict] = Field(..., description="List of prediction results")
    summary: dict = Field(..., description="Summary statistics")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_info: Optional[dict] = Field(None, description="Model information")

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection API is running! 🚀",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict_batch"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionData):
    """
    Predict fraud for a single transaction
    
    - **amount**: Transaction amount in currency units
    - **merchant_type**: Category of the merchant (e.g., electronics, groceries)
    - **device_type**: Device used for transaction (e.g., mobile, desktop, tablet)
    
    Returns fraud prediction with probability and risk level.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and restart the service."
        )
    
    try:
        result = predictor.predict_single(
            amount=transaction.amount,
            merchant_type=transaction.merchant_type,
            device_type=transaction.device_type
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(batch_data: BatchTransactionData):
    """
    Predict fraud for multiple transactions in batch
    
    Send multiple transactions at once for efficient processing.
    Returns predictions for all transactions with summary statistics.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and restart the service."
        )
    
    try:
        # Convert to DataFrame
        transactions_dict = [t.dict() for t in batch_data.transactions]
        data = pd.DataFrame(transactions_dict)
        
        # Make predictions
        predictions, probabilities = predictor.predict(data)
        
        # Format results
        results = []
        fraud_count = 0
        high_risk_count = 0
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Determine risk level
            if prob > 0.7:
                risk_level = 'High'
                high_risk_count += 1
            elif prob > 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            if pred:
                fraud_count += 1
            
            results.append({
                'transaction_id': i,
                'is_fraud': bool(pred),
                'fraud_probability': float(prob),
                'risk_level': risk_level,
                'transaction_details': transactions_dict[i]
            })
        
        # Calculate summary statistics
        total_transactions = len(results)
        summary = {
            'total_transactions': total_transactions,
            'fraud_detected': fraud_count,
            'fraud_percentage': round((fraud_count / total_transactions) * 100, 2),
            'high_risk_transactions': high_risk_count,
            'avg_fraud_probability': round(float(probabilities.mean()), 3),
            'max_fraud_probability': round(float(probabilities.max()), 3)
        }
        
        return BatchPredictionResponse(predictions=results, summary=summary)
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch prediction error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns the current status of the API and model information.
    """
    model_info = None
    if predictor is not None:
        try:
            model_info = predictor.get_model_info()
        except:
            model_info = None
    
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_info=model_info
    )

@app.get("/model/info")
async def get_model_info():
    """Get detailed information about the loaded model"""
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        return predictor.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting model info: {str(e)}"
        )

# Main function to run the API
def run_api():
    """Run the FastAPI server"""
    try:
        config = utils.load_config()
        host = config.get('api_host', '0.0.0.0')
        port = config.get('api_port', 8000)
        
        print(f"🚀 Starting Fraud Detection API server...")
        print(f"📍 Host: {host}")
        print(f"🔌 Port: {port}")
        print(f"📖 Documentation: http://{host}:{port}/docs")
        print(f"🔍 Interactive API: http://{host}:{port}/redoc")
        
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

if __name__ == "__main__":
    run_api()
