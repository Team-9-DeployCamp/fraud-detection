import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from src.util import load_config, pickle_load, print_debug, pickle_dump
import os

def load_data_and_encoders(config):
    """Load raw data splits and encoders from preprocessing"""
    print_debug("Loading raw data and encoders...")
    
    # Load raw data splits (sebelum one-hot encoding)
    X_train = pickle_load(config['train_set_path'][0])
    y_train = pickle_load(config['train_set_path'][1])
    X_valid = pickle_load(config['valid_set_path'][0])
    y_valid = pickle_load(config['valid_set_path'][1])
    X_test = pickle_load(config['test_set_path'][0])
    y_test = pickle_load(config['test_set_path'][1])
    
    print_debug(f"Raw train shape: {X_train.shape}")
    print_debug(f"Raw valid shape: {X_valid.shape}")
    print_debug(f"Raw test shape: {X_test.shape}")
    
    # Load encoders
    merchant_encoder = pickle_load(config['merchant_encoder_path'])
    device_encoder = pickle_load(config['device_encoder_path'])
    
    print_debug(f"Merchant categories: {merchant_encoder.categories_[0]}")
    print_debug(f"Device categories: {device_encoder.categories_[0]}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, merchant_encoder, device_encoder

def apply_ohe_encoding(X_data, merchant_encoder, device_encoder):
    """Apply one-hot encoding to data using pre-fitted encoders"""
    X_processed = X_data.copy()
    
    # Transform merchant_type
    merchant_encoded = merchant_encoder.transform(X_data[['merchant_type']])
    merchant_feature_names = merchant_encoder.get_feature_names_out(['merchant_type'])
    merchant_df = pd.DataFrame(
        merchant_encoded, 
        columns=merchant_feature_names,
        index=X_data.index
    )
    
    # Transform device_type
    device_encoded = device_encoder.transform(X_data[['device_type']])
    device_feature_names = device_encoder.get_feature_names_out(['device_type'])
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

def train_tree_model():
    cfg = load_config()
    
    # Setup MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    
    # Load raw data and encoders
    X_train_raw, y_train, X_valid_raw, y_valid, X_test_raw, y_test, merchant_encoder, device_encoder = load_data_and_encoders(cfg)
    
    # Apply one-hot encoding using pre-fitted encoders
    print_debug("Applying one-hot encoding...")
    X_train = apply_ohe_encoding(X_train_raw, merchant_encoder, device_encoder)
    X_valid = apply_ohe_encoding(X_valid_raw, merchant_encoder, device_encoder)
    X_test = apply_ohe_encoding(X_test_raw, merchant_encoder, device_encoder)
    
    print_debug(f"Processed train shape: {X_train.shape}")
    print_debug(f"Processed valid shape: {X_valid.shape}")
    print_debug(f"Processed test shape: {X_test.shape}")
    print_debug(f"Features: {X_train.columns.tolist()}")
    
    # Verify that all datasets have the same columns
    assert list(X_train.columns) == list(X_valid.columns) == list(X_test.columns), "Column mismatch between datasets"
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Load tuned parameters
    with open("dt_tuned_params.json") as f:
        tuned_data = json.load(f)
        tuned_params = tuned_data["best_parameters"]
    
    # Define models
    models = {
        "dt_baseline": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {"random_state": 42},
            "description": "Decision Tree Baseline (default params)"
        },
        "dt_tuned": {
            "model": DecisionTreeClassifier(**tuned_params, random_state=42),
            "params": {**tuned_params, "random_state": 42},
            "description": "Decision Tree with Tuned Hyperparameters"
        },
        "xgboost": {
            "model": XGBClassifier(random_state=42),
            "params": {"random_state": 42},
            "description": "XGBoost Classifier"
        },
        "catboost": {
            "model": CatBoostClassifier(random_state=42, verbose=0),
            "params": {"random_state": 42},
            "description": "CatBoost Classifier"
        }
    }

    results = []
    
    for model_name, model_info in models.items():
        with mlflow.start_run(run_name=model_name):
            print_debug(f"Training {model_name}...")
            
            # Train model
            model = model_info["model"]
            model.fit(X_train, y_train)
            
            # Predictions on validation set
            y_pred_train = model.predict(X_train)
            y_pred_valid = model.predict(X_valid)
            y_proba_valid = model.predict_proba(X_valid)[:, 1]
            
            # Predictions on test set for final evaluation
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics for training set
            train_metrics = {
                "train_accuracy": accuracy_score(y_train, y_pred_train),
                "train_f1": f1_score(y_train, y_pred_train, zero_division=0),
                "train_precision": precision_score(y_train, y_pred_train, zero_division=0),
                "train_recall": recall_score(y_train, y_pred_train, zero_division=0)
            }
            
            # Calculate metrics for validation set
            valid_metrics = {
                "valid_accuracy": accuracy_score(y_valid, y_pred_valid),
                "valid_f1": f1_score(y_valid, y_pred_valid, zero_division=0),
                "valid_precision": precision_score(y_valid, y_pred_valid, zero_division=0),
                "valid_recall": recall_score(y_valid, y_pred_valid, zero_division=0),
                "valid_auc": roc_auc_score(y_valid, y_proba_valid) if len(set(y_valid)) > 1 else 0.0
            }
            
            # Calculate metrics for test set
            test_metrics = {
                "test_accuracy": accuracy_score(y_test, y_pred_test),
                "test_f1": f1_score(y_test, y_pred_test, zero_division=0),
                "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
                "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
                "test_auc": roc_auc_score(y_test, y_proba_test) if len(set(y_test)) > 1 else 0.0
            }
            
            all_metrics = {**train_metrics, **valid_metrics, **test_metrics}
            
            # Log to MLflow
            mlflow.log_params(model_info["params"])
            mlflow.log_metrics(all_metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Log encoders sebagai artifacts
            mlflow.log_artifact(cfg['merchant_encoder_path'], "encoders")
            mlflow.log_artifact(cfg['device_encoder_path'], "encoders")
            
            # Add tags
            mlflow.set_tags({
                "model_type": model_name.split('_')[0] if '_' in model_name else model_name,
                "model_variant": model_name,
                "description": model_info["description"],
                "data_preprocessing": "one_hot_encoding",
                "encoder_type": "OneHotEncoder"
            })
            
            # Store results
            result = {
                "model_name": model_name,
                "description": model_info["description"],
                **all_metrics,
                "run_id": mlflow.active_run().info.run_id
            }
            results.append(result)
            
            print_debug(f"{model_name} Results:")
            print_debug(f"  Valid F1: {valid_metrics['valid_f1']:.4f}, Valid Accuracy: {valid_metrics['valid_accuracy']:.4f}")
            print_debug(f"  Test F1: {test_metrics['test_f1']:.4f}, Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.round(4)
    
    print_debug("\n=== MODELS COMPARISON ===")
    print(comparison_df[['model_name', 'valid_f1', 'valid_accuracy', 'test_f1', 'test_accuracy', 'valid_auc', 'test_auc']])
    
    # Save best model (berdasarkan valid_f1)
    best_model_idx = comparison_df['valid_f1'].idxmax()
    best_model_info = comparison_df.iloc[best_model_idx]
    
    print_debug(f"\n=== BEST MODEL ===")
    print_debug(f"Best model: {best_model_info['model_name']}")
    print_debug(f"Valid F1: {best_model_info['valid_f1']:.4f}")
    print_debug(f"Test F1: {best_model_info['test_f1']:.4f}")
    
    # Re-train best model and save
    best_model_name = best_model_info['model_name']
    best_model_config = models[best_model_name]
    best_model = best_model_config["model"]
    best_model.fit(X_train, y_train)
    
    # Save best model
    best_model_path = f"models/{best_model_name}_best.pkl"
    pickle_dump(best_model, best_model_path)
    print_debug(f"✓ Best model saved to: {best_model_path}")
    
    return results, comparison_df

if __name__ == "__main__":
    results, comparison = train_tree_model()