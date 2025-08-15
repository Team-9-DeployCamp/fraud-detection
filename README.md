# Fraud Detection System

A complete machine learning system for credit card fraud detection using MLflow for model tracking and FastAPI for serving predictions.

## 🏗️ Project Structure

```
fraud_detection/
├── 📁 src/                     # Source code
│   ├── api.py                  # FastAPI application
│   ├── inference.py            # Model inference logic
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Model training
│   └── util.py                 # Utility functions
├── 📁 deployment/              # Docker deployment files
│   ├── docker-compose.yml     # Multi-service orchestration
│   ├── Dockerfile.api          # API service container
│   ├── Dockerfile.mlflow       # MLflow server container
│   ├── Dockerfile              # Standalone API container
│   └── .dockerignore           # Docker ignore rules
├── 📁 scripts/                 # Management and utility scripts
│   ├── manage.ps1              # PowerShell management script
│   ├── manage.sh               # Bash management script
│   ├── run_pipeline.py         # ML pipeline runner
│   ├── test_api.py             # API testing script
│   └── test_docker_services.py # Docker services test
├── 📁 configs/                 # Configuration files
│   ├── config/
│   │   ├── config.yaml         # Main configuration
│   │   └── config.docker.yaml  # Docker-specific config
│   ├── dt_tuned_params.json    # Decision Tree hyperparameters
│   ├── first_tuned_params.json # Initial tuning results
│   └── xgb_tuned_params.json   # XGBoost hyperparameters
├── 📁 docs/                    # Documentation
│   ├── README.md               # Basic project info
│   ├── README_API.md           # API documentation
│   └── README_DOCKER.md        # Docker deployment guide
├── 📁 dataset/                 # Data files
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed data
├── 📁 models/                  # Trained model files
├── 📁 notebooks/               # Jupyter notebooks for analysis
├── 📁 mlruns/                  # MLflow experiment tracking
├── mlflow.db                   # MLflow metadata database
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Docker Desktop (for containerized deployment)
- Python 3.10+ (for local development)

### Using Docker (Recommended)

1. **Navigate to project directory:**
   ```bash
   cd fraud_detection
   ```

2. **Start services using PowerShell:**
   ```powershell
   cd scripts
   .\manage.ps1 -Command build
   .\manage.ps1 -Command start
   ```

   **Or using Bash:**
   ```bash
   cd scripts
   chmod +x manage.sh
   ./manage.sh build
   ./manage.sh start
   ```

3. **Access services:**
   - **MLflow UI**: http://localhost:5000
   - **API Documentation**: http://localhost:8000/docs
   - **API Health**: http://localhost:8000/health

4. **Test the system:**
   ```bash
   cd scripts
   python test_docker_services.py
   ```

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLflow server:**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```

3. **Start API server:**
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🎯 Key Features

- **ML Model Management**: MLflow for experiment tracking and model registry
- **Real-time API**: FastAPI for serving fraud detection predictions
- **Containerized Deployment**: Docker Compose for easy deployment
- **Model Training Pipeline**: Automated training with hyperparameter tuning
- **Data Preprocessing**: Complete preprocessing pipeline with encoding and scaling
- **Health Monitoring**: Health checks and service monitoring

## 📊 API Usage

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 1500.50,
       "merchant_type": "grocery",
       "device_type": "mobile"
     }'
```

### Response
```json
{
  "is_fraud": true,
  "fraud_probability": 0.76,
  "risk_level": "High",
  "transaction_details": {
    "amount": 1500.5,
    "merchant_type": "grocery",
    "device_type": "mobile"
  }
}
```

## 🛠️ Management Commands

### PowerShell (Windows)
```powershell
# Start all services
.\scripts\manage.ps1 -Command start

# Stop all services
.\scripts\manage.ps1 -Command stop

# View logs
.\scripts\manage.ps1 -Command logs

# Check status
.\scripts\manage.ps1 -Command status
```

### Bash (Linux/Mac)
```bash
# Start all services
./scripts/manage.sh start

# Stop all services
./scripts/manage.sh stop

# View logs
./scripts/manage.sh logs

# Check status
./scripts/manage.sh status
```

## 📚 Documentation

- **[API Documentation](docs/README_API.md)**: Detailed API reference
- **[Docker Deployment](docs/README_DOCKER.md)**: Complete Docker setup guide
- **[Basic Setup](docs/README.md)**: Basic project information

## 🔧 Development

For development and experimentation, see the Jupyter notebooks in the `notebooks/` directory:
- `EDA.ipynb`: Exploratory Data Analysis
- `data_preparation.ipynb`: Data preparation steps
- `data_preprocessing.ipynb`: Preprocessing pipeline
- `modelling.ipynb`: Model training and evaluation
- `analisis_dan_eksperimen.ipynb`: Analysis and experiments

## 🏷️ Model Information

The system supports multiple ML algorithms:
- Decision Tree
- XGBoost
- CatBoost
- with SMOTE for handling imbalanced data

Models are automatically tracked in MLflow with performance metrics, and the best performing model is served via the API.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes using the provided test scripts
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
