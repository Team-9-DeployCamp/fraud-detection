import pandas as pd
import joblib
import numpy as np
import sys
import os
from . import util as utils

class FraudDetectionPredictor:
    """Fraud Detection Model Predictor Class"""
    
    def __init__(self):
        """Initialize the predictor with trained model and encoders"""
        self.config = utils.load_config()
        
        # Load trained model and encoders
        try:
            utils.print_debug(f"Loading model from: {self.config['model_path']}")
            self.model = utils.pickle_load(self.config['model_path'])
            utils.print_debug("✓ Model loaded successfully")
            
            utils.print_debug(f"Loading merchant encoder from: {self.config['merchant_encoder_path']}")
            self.merchant_encoder = utils.pickle_load(self.config['merchant_encoder_path'])
            utils.print_debug("✓ Merchant encoder loaded successfully")
            
            utils.print_debug(f"Loading device encoder from: {self.config['device_encoder_path']}")
            self.device_encoder = utils.pickle_load(self.config['device_encoder_path'])
            utils.print_debug("✓ Device encoder loaded successfully")
            
            print("Model and encoders loaded successfully!")
        except Exception as e:
            utils.print_debug(f"Error details: {e}")
            utils.print_debug(f"Working directory: {os.getcwd()}")
            utils.print_debug(f"Config: {self.config}")
            raise RuntimeError(f"Error loading model or encoders: {e}")
    
    def preprocess_data(self, data):
        """
        Preprocess input data for prediction
        
        Args:
            data (pd.DataFrame): Input data with columns ['amount', 'merchant_type', 'device_type']
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        data_processed = data.copy()
        
        # Ensure required columns exist
        required_columns = ['amount', 'merchant_type', 'device_type']
        for col in required_columns:
            if col not in data_processed.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert to string and handle unknown categories
        data_processed['merchant_type'] = data_processed['merchant_type'].astype(str)
        data_processed['device_type'] = data_processed['device_type'].astype(str)
        
        # Handle unknown categories by replacing with most frequent category
        known_merchants = set(self.merchant_encoder.classes_)
        known_devices = set(self.device_encoder.classes_)
        
        # Replace unknown categories with most frequent one (first in classes_)
        data_processed['merchant_type'] = data_processed['merchant_type'].apply(
            lambda x: x if x in known_merchants else self.merchant_encoder.classes_[0]
        )
        data_processed['device_type'] = data_processed['device_type'].apply(
            lambda x: x if x in known_devices else self.device_encoder.classes_[0]
        )
        
        # Encode categorical features
        data_processed['merchant_type'] = self.merchant_encoder.transform(data_processed['merchant_type'])
        data_processed['device_type'] = self.device_encoder.transform(data_processed['device_type'])
        
        return data_processed
    
    def predict(self, data):
        """
        Make predictions on batch data
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            tuple: (predictions, probabilities)
        """
        # Preprocess data
        data_processed = self.preprocess_data(data)
        
        # Make predictions
        predictions = self.model.predict(data_processed)
        probabilities = self.model.predict_proba(data_processed)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(self, amount, merchant_type, device_type):
        """
        Predict fraud for a single transaction
        
        Args:
            amount (float): Transaction amount
            merchant_type (str): Type of merchant
            device_type (str): Type of device used
            
        Returns:
            dict: Prediction results with fraud status, probability, and risk level
        """
        # Create DataFrame for single prediction
        data = pd.DataFrame({
            'amount': [float(amount)],
            'merchant_type': [str(merchant_type)],
            'device_type': [str(device_type)]
        })
        
        # Make prediction
        prediction, probability = self.predict(data)
        
        # Determine risk level based on probability
        if probability[0] > 0.7:
            risk_level = 'High'
        elif probability[0] > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'is_fraud': bool(prediction[0]),
            'fraud_probability': float(probability[0]),
            'risk_level': risk_level,
            'transaction_details': {
                'amount': float(amount),
                'merchant_type': str(merchant_type),
                'device_type': str(device_type)
            }
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': type(self.model).__name__,
            'merchant_categories': list(self.merchant_encoder.classes_),
            'device_categories': list(self.device_encoder.classes_),
            'feature_count': len(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else 'Unknown'
        }

# Test function
def test_predictor():
    """Test the predictor with sample data"""
    try:
        predictor = FraudDetectionPredictor()
        
        # Test single prediction
        result = predictor.predict_single(
            amount=150.50,
            merchant_type='electronics',
            device_type='mobile'
        )
        
        print("Test prediction result:")
        print(result)
        
        # Test batch prediction
        test_data = pd.DataFrame({
            'amount': [100.0, 500.0, 1000.0],
            'merchant_type': ['groceries', 'electronics', 'others'],
            'device_type': ['mobile', 'desktop', 'tablet']
        })
        
        predictions, probabilities = predictor.predict(test_data)
        print("\nBatch prediction results:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Transaction {i+1}: Fraud={pred}, Probability={prob:.3f}")
            
        # Print model info
        print("\nModel information:")
        print(predictor.get_model_info())
        
    except Exception as e:
        print(f"Error testing predictor: {e}")

if __name__ == "__main__":
    test_predictor()
