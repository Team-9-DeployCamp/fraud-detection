# Fraud Detection Docker Deployment

This document explains how to deploy the Fraud Detection system using Docker Compose with two services: MLflow tracking server and the Fraud Detection API.

## Architecture

The system consists of two main services:

1. **MLflow Server** (`mlflow-server`): 
   - Runs on port 5000
   - Manages ML models and experiments
   - Provides model registry and tracking

2. **Fraud Detection API** (`fraud-api`):
   - Runs on port 8000
   - FastAPI application that serves predictions
   - Connects to MLflow server to access the best model

## Prerequisites

- Docker
- Docker Compose
- Git

## Quick Start

### 1. Clone and Navigate to Project
```bash
git clone <repository-url>
cd fraud_detection
```

### 2. Build and Start Services

#### Using PowerShell (Windows):
```powershell
.\manage.ps1 -Command build
.\manage.ps1 -Command start
```

#### Using Bash (Linux/Mac):
```bash
chmod +x manage.sh
./manage.sh build
./manage.sh start
```

#### Using Docker Compose directly:
```bash
docker-compose build
docker-compose up -d
```

### 3. Access Services

- **MLflow UI**: http://localhost:5000
- **Fraud Detection API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Service Management

### Using Management Scripts

#### PowerShell Commands:
```powershell
# Start services
.\manage.ps1 -Command start

# Stop services
.\manage.ps1 -Command stop

# Restart services
.\manage.ps1 -Command restart

# View logs
.\manage.ps1 -Command logs

# Check status
.\manage.ps1 -Command status

# Rebuild images
.\manage.ps1 -Command build
```

#### Bash Commands:
```bash
# Start services
./manage.sh start

# Stop services
./manage.sh stop

# Restart services
./manage.sh restart

# View logs
./manage.sh logs

# Check status
./manage.sh status

# Rebuild images
./manage.sh build
```

### Using Docker Compose Directly

```bash
# Start services in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Rebuild and start
docker-compose up --build -d
```

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 1500.50,
       "merchant_type": "grocery",
       "device_type": "mobile"
     }'
```

## Configuration

### Environment Variables

The services use the following environment variables:

**MLflow Server:**
- `MLFLOW_BACKEND_STORE_URI`: SQLite database URL
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: Artifact storage path

**Fraud API:**
- `MLFLOW_TRACKING_URI`: MLflow server URL (http://mlflow-server:5000)

### Configuration Files

- `config/config.docker.yaml`: Docker-specific configuration
- `docker-compose.yml`: Service definitions
- `Dockerfile.mlflow`: MLflow server image
- `Dockerfile.api`: API service image

## Data Persistence

The following directories are mounted as volumes to persist data:

- `./mlruns`: MLflow artifacts and experiments
- `./mlflow.db`: MLflow metadata database
- `./models`: Trained model files
- `./config`: Configuration files
- `./dataset`: Dataset files

## Troubleshooting

### Service Health Checks

Both services include health checks that can be monitored:

```bash
# Check service health
docker-compose ps

# View detailed logs
docker-compose logs mlflow-server
docker-compose logs fraud-api
```

### Common Issues

1. **Port Conflicts**: If ports 5000 or 8000 are already in use, modify the ports in `docker-compose.yml`

2. **MLflow Connection Issues**: Ensure the MLflow server is healthy before the API starts. The API service waits for MLflow health check.

3. **Model Loading Issues**: Check that the model files exist in the `models` directory and are accessible.

### Debugging

```bash
# Enter service container for debugging
docker-compose exec mlflow-server bash
docker-compose exec fraud-api bash

# View real-time logs
docker-compose logs -f --tail=100
```

## Development

For development, you can modify the services to enable auto-reload:

1. Uncomment the `--reload` flag in the API service command
2. Mount source code as volumes for live updates

## Production Considerations

For production deployment:

1. Use environment-specific configuration files
2. Set up proper logging and monitoring
3. Use external databases instead of SQLite
4. Implement proper security measures
5. Set up load balancing if needed
6. Use secrets management for sensitive data

## API Endpoints

Once running, visit http://localhost:8000/docs for interactive API documentation with the following main endpoints:

- `GET /health`: Health check
- `POST /predict`: Make fraud predictions
- `GET /model/info`: Get current model information
- `GET /metrics`: Get API metrics (if implemented)

## MLflow UI

Access the MLflow UI at http://localhost:5000 to:

- View experiments and runs
- Compare model performance
- Manage model registry
- Track model versions
