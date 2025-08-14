import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import sys
import os
from . import util as utils
from sklearn.preprocessing import LabelEncoder

def train_baseline_model():
    """Train baseline fraud detection models and save the best one"""
    print("Starting fraud detection model training...")
    
    # Load config dan data
    config = utils.load_config()
    
    # Load training data
    print("Loading training data...")
    X_train = utils.pickle_load(config['train_set_path'][0])
    y_train = utils.pickle_load(config['train_set_path'][1])
    X_valid = utils.pickle_load(config['valid_set_path'][0])
    y_valid = utils.pickle_load(config['valid_set_path'][1])
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_valid.shape}")
    print(f"Training labels distribution: {y_train.value_counts().to_dict()}")
    
    # Initialize label encoders for categorical features
    le_merchant = LabelEncoder()
    le_device = LabelEncoder()
    
    # Fit encoders on training data and transform
    print("Preprocessing categorical features...")
    X_train_processed = X_train.copy()
    X_train_processed['merchant_type'] = le_merchant.fit_transform(X_train['merchant_type'].astype(str))
    X_train_processed['device_type'] = le_device.fit_transform(X_train['device_type'].astype(str))
    
    # Transform validation data
    X_valid_processed = X_valid.copy()
    X_valid_processed['merchant_type'] = le_merchant.transform(X_valid['merchant_type'].astype(str))
    X_valid_processed['device_type'] = le_device.transform(X_valid['device_type'].astype(str))
    
    print("Available features:", X_train_processed.columns.tolist())
    
    # Define baseline models
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            max_depth=10
        )
    }
    
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_processed, y_train)
        
        # Validation predictions
        val_pred = model.predict(X_valid_processed)
        val_prob = model.predict_proba(X_valid_processed)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_valid, val_prob)
        
        print(f"\n{name} Results:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_valid, val_pred))
        
        trained_models[name] = {
            'model': model,
            'roc_auc': roc_auc
        }
    
    # Select best model based on ROC-AUC score
    best_model_name = max(trained_models.keys(), 
                         key=lambda k: trained_models[k]['roc_auc'])
    best_model = trained_models[best_model_name]['model']
    best_score = trained_models[best_model_name]['roc_auc']
    
    print(f"\n{'='*50}")
    print(f"Best model: {best_model_name}")
    print(f"Best ROC-AUC score: {best_score:.4f}")
    print(f"{'='*50}")
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Save model and encoders
    print("\nSaving model and encoders...")
    joblib.dump(best_model, config['model_path'])
    joblib.dump(le_merchant, config['merchant_encoder_path'])
    joblib.dump(le_device, config['device_encoder_path'])
    
    print("Model training completed successfully!")
    print(f"Model saved to: {config['model_path']}")
    print(f"Merchant encoder saved to: {config['merchant_encoder_path']}")
    print(f"Device encoder saved to: {config['device_encoder_path']}")
    
    return best_model, le_merchant, le_device

if __name__ == "__main__":
    train_baseline_model()
