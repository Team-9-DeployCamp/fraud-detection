#!/usr/bin/env python3
"""
Test script to verify that the Fraud Detection Docker services are working correctly.
"""

import requests
import time
import json
import sys

def test_mlflow_server(base_url="http://localhost:5000"):
    """Test MLflow server connectivity."""
    print("Testing MLflow server...")
    try:
        # Test basic connectivity
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ MLflow server is healthy")
            return True
        else:
            print(f"❌ MLflow server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ MLflow server is not accessible: {e}")
        return False

def test_fraud_api(base_url="http://localhost:8000"):
    """Test Fraud Detection API."""
    print("\nTesting Fraud Detection API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API health check passed")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API is not accessible: {e}")
        return False
    
    # Test prediction endpoint
    test_data = {
        "amount": 1500.50,
        "merchant_type": "grocery",
        "device_type": "mobile"
    }
    
    try:
        print("Testing prediction endpoint...")
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction endpoint working")
            print(f"   Sample prediction: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction request failed: {e}")
        return False

def wait_for_services(max_wait=120):
    """Wait for services to be ready."""
    print(f"Waiting for services to be ready (max {max_wait} seconds)...")
    
    for i in range(max_wait):
        try:
            # Check MLflow
            mlflow_response = requests.get("http://localhost:5000/health", timeout=5)
            # Check API
            api_response = requests.get("http://localhost:8000/health", timeout=5)
            
            if mlflow_response.status_code == 200 and api_response.status_code == 200:
                print(f"✅ Services are ready after {i+1} seconds")
                return True
                
        except requests.exceptions.RequestException:
            pass
        
        if i % 10 == 0 and i > 0:
            print(f"   Still waiting... ({i}/{max_wait} seconds)")
        
        time.sleep(1)
    
    print(f"❌ Services not ready after {max_wait} seconds")
    return False

def main():
    """Main test function."""
    print("🚀 Starting Fraud Detection Services Test")
    print("=" * 50)
    
    # Wait for services to be ready
    if not wait_for_services():
        print("\n❌ Services are not ready. Please check Docker containers.")
        sys.exit(1)
    
    # Test MLflow server
    mlflow_ok = test_mlflow_server()
    
    # Test Fraud API
    api_ok = test_fraud_api()
    
    print("\n" + "=" * 50)
    if mlflow_ok and api_ok:
        print("🎉 All tests passed! Fraud Detection system is working correctly.")
        print("\n📋 Service URLs:")
        print("   MLflow UI: http://localhost:5000")
        print("   API Documentation: http://localhost:8000/docs")
        print("   API Health: http://localhost:8000/health")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the service logs.")
        print("\nTo check logs:")
        print("   docker-compose logs -f")
        sys.exit(1)

if __name__ == "__main__":
    main()
