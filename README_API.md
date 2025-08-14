# Fraud Detection Project

## 🔍 Overview
This project implements a machine learning-based fraud detection system with a REST API for real-time predictions.

## 📁 Project Structure
```
fraud_detection/
├── src/
│   ├── train_model.py      # Model training script
│   ├── inference.py        # Prediction logic
│   ├── api.py             # FastAPI REST API
│   └── util.py            # Utility functions
├── models/                 # Trained models (created after training)
├── dataset/               # Data files
├── config/
│   └── config.yaml        # Configuration file
├── notebooks/             # Jupyter notebooks for EDA
├── run_pipeline.py        # Main pipeline script
├── test_api.py           # API testing script
└── requirements_api.txt   # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python run_pipeline.py install
```

### 2. Train Model
```bash
python run_pipeline.py train
```

### 3. Start API Server
```bash
python run_pipeline.py api
```

### 4. Run Everything (Install + Train + API)
```bash
python run_pipeline.py full
```

## 📡 API Endpoints

### Health Check
```bash
GET http://127.0.0.1:8000/health
```

### Single Prediction
```bash
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
    "amount": 150.50,
    "merchant_type": "electronics",
    "device_type": "mobile"
}
```

### Batch Prediction
```bash
POST http://127.0.0.1:8000/predict_batch
Content-Type: application/json

{
    "transactions": [
        {"amount": 50.25, "merchant_type": "groceries", "device_type": "mobile"},
        {"amount": 1000.00, "merchant_type": "electronics", "device_type": "desktop"}
    ]
}
```

## 🧪 Testing

### Test API (run this in a separate terminal while API is running)
```bash
python test_api.py
```

### Interactive API Documentation
Once the API is running, visit:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## 🔧 Configuration

Edit `config/config.yaml` to modify:
- Data paths
- Model parameters
- API settings (host, port)

## 📊 Model Information

The system trains baseline models:
- **Logistic Regression**: Fast, interpretable
- **Random Forest**: More complex, potentially better performance

The best performing model (based on ROC-AUC) is automatically selected and saved.

## 🛡️ Fraud Detection Features

- **Real-time predictions**: Single transaction analysis
- **Batch processing**: Multiple transactions at once
- **Risk levels**: Low, Medium, High risk classification
- **Probability scores**: Fraud likelihood (0-1)
- **Model info**: Get details about the loaded model

## 📈 Response Format

```json
{
    "is_fraud": false,
    "fraud_probability": 0.234,
    "risk_level": "Low",
    "transaction_details": {
        "amount": 150.50,
        "merchant_type": "electronics",
        "device_type": "mobile"
    }
}
```

## 🔄 Pipeline Commands

| Command | Description |
|---------|-------------|
| `python run_pipeline.py install` | Install required packages |
| `python run_pipeline.py train` | Train the fraud detection model |
| `python run_pipeline.py test` | Test the inference module |
| `python run_pipeline.py api` | Start the API server |
| `python run_pipeline.py full` | Run complete pipeline |

## 🚨 Troubleshooting

### Model Not Found Error
- Make sure to run training first: `python run_pipeline.py train`

### API Connection Error
- Check if API is running: `python run_pipeline.py api`
- Verify port 8000 is not in use

### Package Import Errors
- Install dependencies: `python run_pipeline.py install`
- Or manually: `pip install -r requirements_api.txt`

## 📝 Development Notes

- The model handles unknown categories gracefully
- All categorical features are automatically encoded
- The API includes comprehensive error handling
- Logging and debugging information is available

## 🎯 Next Steps

1. **Model Improvement**: 
   - Feature engineering
   - Hyperparameter tuning
   - Advanced algorithms (XGBoost, Neural Networks)

2. **Production Deployment**:
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)
   - Load balancing and scaling

3. **Monitoring**:
   - Model performance tracking
   - Data drift detection
   - Real-time alerts
