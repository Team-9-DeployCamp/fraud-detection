#!/bin/bash

# Fraud Detection Docker Compose Management Script

# Function to display usage
usage() {
    echo "Usage: $0 {start|stop|restart|logs|status|build}"
    echo ""
    echo "Commands:"
    echo "  start   - Start all services"
    echo "  stop    - Stop all services"
    echo "  restart - Restart all services"
    echo "  logs    - Show logs for all services"
    echo "  status  - Show status of all services"
    echo "  build   - Build all Docker images"
    echo ""
    exit 1
}

# Function to start services
start_services() {
    echo "Starting Fraud Detection services..."
    cd ../deployment
    docker-compose up -d
    cd ../scripts
    echo "Services started. MLflow UI: http://localhost:5000, API: http://localhost:8000"
}

# Function to stop services
stop_services() {
    echo "Stopping Fraud Detection services..."
    cd ../deployment
    docker-compose down
    cd ../scripts
    echo "Services stopped."
}

# Function to restart services
restart_services() {
    echo "Restarting Fraud Detection services..."
    cd ../deployment
    docker-compose restart
    cd ../scripts
    echo "Services restarted."
}

# Function to show logs
show_logs() {
    echo "Showing logs for all services..."
    cd ../deployment
    docker-compose logs -f
    cd ../scripts
}

# Function to show status
show_status() {
    echo "Service status:"
    cd ../deployment
    docker-compose ps
    cd ../scripts
}

# Function to build images
build_images() {
    echo "Building Docker images..."
    cd ../deployment
    docker-compose build
    cd ../scripts
    echo "Images built successfully."
}

# Main script logic
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    build)
        build_images
        ;;
    *)
        usage
        ;;
esac
