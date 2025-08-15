# Fraud Detection Docker Compose Management Script (PowerShell)

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "restart", "logs", "status", "build")]
    [string]$Command
)

function Show-Usage {
    Write-Host "Usage: .\manage.ps1 -Command <command>"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  start   - Start all services"
    Write-Host "  stop    - Stop all services"
    Write-Host "  restart - Restart all services"
    Write-Host "  logs    - Show logs for all services"
    Write-Host "  status  - Show status of all services"
    Write-Host "  build   - Build all Docker images"
    Write-Host ""
    Write-Host "Example: .\manage.ps1 -Command start"
}

function Start-Services {
    Write-Host "Starting Fraud Detection services..." -ForegroundColor Green
    Set-Location -Path "..\deployment"
    docker-compose up -d
    Set-Location -Path "..\scripts"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Services started successfully!" -ForegroundColor Green
        Write-Host "MLflow UI: http://localhost:5000" -ForegroundColor Yellow
        Write-Host "Fraud Detection API: http://localhost:8000" -ForegroundColor Yellow
        Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Yellow
    } else {
        Write-Host "Failed to start services!" -ForegroundColor Red
    }
}

function Stop-Services {
    Write-Host "Stopping Fraud Detection services..." -ForegroundColor Yellow
    Set-Location -Path "..\deployment"
    docker-compose down
    Set-Location -Path "..\scripts"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Services stopped successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to stop services!" -ForegroundColor Red
    }
}

function Restart-Services {
    Write-Host "Restarting Fraud Detection services..." -ForegroundColor Yellow
    Set-Location -Path "..\deployment"
    docker-compose restart
    Set-Location -Path "..\scripts"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Services restarted successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to restart services!" -ForegroundColor Red
    }
}

function Show-Logs {
    Write-Host "Showing logs for all services..." -ForegroundColor Blue
    Set-Location -Path "..\deployment"
    docker-compose logs -f
    Set-Location -Path "..\scripts"
}

function Show-Status {
    Write-Host "Service status:" -ForegroundColor Blue
    Set-Location -Path "..\deployment"
    docker-compose ps
    Set-Location -Path "..\scripts"
}

function Build-Images {
    Write-Host "Building Docker images..." -ForegroundColor Blue
    Set-Location -Path "..\deployment"
    docker-compose build
    Set-Location -Path "..\scripts"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Images built successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to build images!" -ForegroundColor Red
    }
}

# Main script logic
switch ($Command) {
    "start" { Start-Services }
    "stop" { Stop-Services }
    "restart" { Restart-Services }
    "logs" { Show-Logs }
    "status" { Show-Status }
    "build" { Build-Images }
    default { Show-Usage }
}
