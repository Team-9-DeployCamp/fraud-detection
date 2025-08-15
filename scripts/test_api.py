import requests
import json
import time

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API health...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_single_prediction():
    """Test single fraud prediction"""
    print("\n🧪 Testing single prediction...")
    
    url = "http://127.0.0.1:8000/predict"
    test_data = {
        "amount": 150.50,
        "merchant_type": "electronics", 
        "device_type": "mobile"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction successful!")
            print(f"📊 Transaction Amount: ${test_data['amount']}")
            print(f"🏪 Merchant Type: {test_data['merchant_type']}")
            print(f"📱 Device Type: {test_data['device_type']}")
            print(f"🚨 Fraud Detected: {result['is_fraud']}")
            print(f"📈 Fraud Probability: {result['fraud_probability']:.3f}")
            print(f"⚠️  Risk Level: {result['risk_level']}")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_batch_prediction():
    """Test batch fraud prediction"""
    print("\n🧪 Testing batch prediction...")
    
    url = "http://127.0.0.1:8000/predict_batch"
    test_data = {
        "transactions": [
            {"amount": 50.25, "merchant_type": "groceries", "device_type": "mobile"},
            {"amount": 150.50, "merchant_type": "electronics", "device_type": "desktop"},
            {"amount": 1000.00, "merchant_type": "others", "device_type": "tablet"},
            {"amount": 25.99, "merchant_type": "groceries", "device_type": "mobile"},
            {"amount": 500.75, "merchant_type": "electronics", "device_type": "mobile"}
        ]
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            
            # Print summary
            summary = result['summary']
            print(f"\n📊 Batch Summary:")
            print(f"   Total Transactions: {summary['total_transactions']}")
            print(f"   Fraud Detected: {summary['fraud_detected']}")
            print(f"   Fraud Percentage: {summary['fraud_percentage']}%")
            print(f"   High Risk Transactions: {summary['high_risk_transactions']}")
            print(f"   Average Fraud Probability: {summary['avg_fraud_probability']}")
            
            # Print individual results
            print(f"\n📋 Individual Results:")
            for i, pred in enumerate(result['predictions']):
                details = pred['transaction_details']
                print(f"   Transaction {i+1}: ${details['amount']} | "
                      f"{details['merchant_type']} | {details['device_type']} | "
                      f"Fraud: {pred['is_fraud']} | Risk: {pred['risk_level']} | "
                      f"Prob: {pred['fraud_probability']:.3f}")
            
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing model info...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/model/info", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Model info retrieved successfully!")
            print(f"🤖 Model Type: {result.get('model_type', 'Unknown')}")
            print(f"🏪 Merchant Categories: {len(result.get('merchant_categories', []))} types")
            print(f"📱 Device Categories: {len(result.get('device_categories', []))} types")
            print(f"🔢 Feature Count: {result.get('feature_count', 'Unknown')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def run_comprehensive_test():
    """Run all API tests"""
    print("🚀 Starting Comprehensive API Test")
    print("=" * 50)
    
    # Wait a moment for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_api_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print(f"\n{'='*50}")
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the API server logs.")
    
    print("=" * 50)

if __name__ == "__main__":
    run_comprehensive_test()
