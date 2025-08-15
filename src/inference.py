import pandas as pd
import numpy as np
from src.util import load_config, pickle_load, print_debug

class FraudDetectionPredictor:
    """Fraud Detection Model Predictor Class"""
    
    def __init__(self):
        """Initialize the predictor with trained model and encoders"""
        self.config = load_config()
        
        # Load trained model and encoders
        try:
            # Load model
            self.model = pickle_load(self.config.get('production_model_path', 'models/dt_model_selected.pkl'))
            print_debug("✓ Model loaded successfully")
            
            # Load encoders (same as used in preprocessing and training)
            self.merchant_encoder = pickle_load(self.config['merchant_encoder_path'])
            self.device_encoder = pickle_load(self.config['device_encoder_path'])
            print_debug("✓ Encoders loaded successfully")
            
            # Get encoder info
            print_debug(f"Merchant categories: {self.merchant_encoder.categories_[0]}")
            print_debug(f"Device categories: {self.device_encoder.categories_[0]}")
            
        except Exception as e:
            print_debug(f"✗ Error loading model or encoders: {e}")
            raise
    
    def apply_ohe_encoding(self, X_data):
        """
        Apply one-hot encoding using pre-fitted encoders
        Same function as used in train.py
        """
        X_processed = X_data.copy()
        
        # Transform merchant_type
        merchant_encoded = self.merchant_encoder.transform(X_data[['merchant_type']])
        merchant_feature_names = self.merchant_encoder.get_feature_names_out(['merchant_type'])
        merchant_df = pd.DataFrame(
            merchant_encoded, 
            columns=merchant_feature_names,
            index=X_data.index
        )
        
        # Transform device_type
        device_encoded = self.device_encoder.transform(X_data[['device_type']])
        device_feature_names = self.device_encoder.get_feature_names_out(['device_type'])
        device_df = pd.DataFrame(
            device_encoded, 
            columns=device_feature_names,
            index=X_data.index
        )
        
        # Remove original categorical columns
        X_processed = X_processed.drop(columns=['merchant_type', 'device_type'])
        
        # Add encoded features
        X_processed = pd.concat([X_processed, merchant_df, device_df], axis=1)
        
        return X_processed
    
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
        known_merchants = set(self.merchant_encoder.categories_[0])
        known_devices = set(self.device_encoder.categories_[0])
        
        # Replace unknown categories with most frequent one (first in classes_)
        merchant_mask = ~data_processed['merchant_type'].isin(known_merchants)
        device_mask = ~data_processed['device_type'].isin(known_devices)
        
        if merchant_mask.any():
            default_merchant = self.merchant_encoder.categories_[0][0]
            data_processed.loc[merchant_mask, 'merchant_type'] = default_merchant
            print_debug(f"⚠ Unknown merchant types replaced with: {default_merchant}")
        
        if device_mask.any():
            default_device = self.device_encoder.categories_[0][0]
            data_processed.loc[device_mask, 'device_type'] = default_device
            print_debug(f"⚠ Unknown device types replaced with: {default_device}")
        
        # Apply one-hot encoding
        data_encoded = self.apply_ohe_encoding(data_processed)
        
        print_debug(f"Preprocessed data shape: {data_encoded.shape}")
        print_debug(f"Preprocessed features: {data_encoded.columns.tolist()}")
        
        return data_encoded
    
    def predict(self, data):
        """
        Predict fraud for input data
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Prediction results
        """
        try:
            # Preprocess data
            data_processed = self.preprocess_data(data)
            
            # Make predictions
            predictions = self.model.predict(data_processed)
            probabilities = self.model.predict_proba(data_processed)[:, 1]
            
            # Create results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                # Determine risk level
                if prob > 0.7:
                    risk_level = 'High'
                elif prob > 0.3:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                
                result = {
                    'is_fraud': bool(pred),
                    'fraud_probability': float(prob),
                    'risk_level': risk_level,
                    'transaction_details': {
                        'amount': float(data.iloc[i]['amount']),
                        'merchant_type': str(data.iloc[i]['merchant_type']),
                        'device_type': str(data.iloc[i]['device_type'])
                    }
                }
                results.append(result)
            
            return results if len(results) > 1 else results[0]
            
        except Exception as e:
            print_debug(f"✗ Prediction error: {e}")
            raise
    
    def predict_single(self, amount, merchant_type, device_type):
        """
        Predict fraud for a single transaction
        
        Args:
            amount (float): Transaction amount
            merchant_type (str): Merchant type
            device_type (str): Device type
            
        Returns:
            dict: Prediction result
        """
        # Create DataFrame from input
        data = pd.DataFrame({
            'amount': [float(amount)],
            'merchant_type': [str(merchant_type)],
            'device_type': [str(device_type)]
        })
        
        # Use predict method
        result = self.predict(data)
        
        return result
    
    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            model_type = type(self.model).__name__
            
            # Try to get model parameters
            model_params = {}
            if hasattr(self.model, 'get_params'):
                model_params = self.model.get_params()
            
            # Get supported categories
            merchant_categories = self.merchant_encoder.categories_[0].tolist()
            device_categories = self.device_encoder.categories_[0].tolist()
            
            # Get feature names (after encoding)
            sample_data = pd.DataFrame({
                'amount': [100.0],
                'merchant_type': [merchant_categories[0]],
                'device_type': [device_categories[0]]
            })
            processed_sample = self.apply_ohe_encoding(sample_data)
            feature_names = processed_sample.columns.tolist()
            
            info = {
                'model_type': model_type,
                'model_parameters': model_params,
                'supported_merchant_types': merchant_categories,
                'supported_device_types': device_categories,
                'expected_features': feature_names,
                'encoding_type': 'OneHotEncoder'
            }
            
            return info
            
        except Exception as e:
            print_debug(f"✗ Error getting model info: {e}")
            return {
                'model_type': 'Unknown',
                'error': str(e)
            }

# Test function
def test_predictor():
    """Test the predictor with sample data"""
    try:
        print_debug("=== TESTING FRAUD DETECTION PREDICTOR ===")
        
        # Initialize predictor
        predictor = FraudDetectionPredictor()
        
        # Test single prediction
        print_debug("\n--- Single Prediction Test ---")
        result = predictor.predict_single(
            amount=150.50,
            merchant_type="electronics",
            device_type="mobile"
        )
        print_debug(f"Single prediction result: {result}")
        
        # Test batch prediction
        print_debug("\n--- Batch Prediction Test ---")
        test_data = pd.DataFrame({
            'amount': [100.0, 500.0, 25.0],
            'merchant_type': ['electronics', 'groceries', 'travel'],
            'device_type': ['mobile', 'tablet', 'mobile']
        })
        
        batch_results = predictor.predict(test_data)
        print_debug(f"Batch prediction results: {batch_results}")
        
        # Test model info
        print_debug("\n--- Model Info Test ---")
        model_info = predictor.get_model_info()
        print_debug(f"Model info: {model_info}")
        
        # Test unknown categories
        print_debug("\n--- Unknown Categories Test ---")
        unknown_result = predictor.predict_single(
            amount=200.0,
            merchant_type="unknown_merchant",  # Unknown category
            device_type="unknown_device"       # Unknown category
        )
        print_debug(f"Unknown categories result: {unknown_result}")
        
        print_debug("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print_debug(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_predictor()