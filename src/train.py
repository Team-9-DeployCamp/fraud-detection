import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from util import load_config, pickle_load, print_debug

def train_tree_model():
    cfg = load_config()
    
    # Setup MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    
    # Load data
    X_train = pickle_load("dataset/processed/X_train_processed.pkl")
    y_train = pickle_load("dataset/processed/y_train.pkl")
    X_valid = pickle_load("dataset/processed/X_test_processed.pkl")
    y_valid = pickle_load("dataset/processed/y_test.pkl")
    
    print_debug(f"Training data shape: {X_train.shape}")
    print_debug(f"Validation data shape: {X_valid.shape}")
    
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
        "dt_csl": {
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),
            "params": {"class_weight": "balanced", "random_state": 42},
            "description": "Decision Tree with Cost-Sensitive Learning (balanced)"
        }
    }
    
    results = []
    
    for model_name, model_info in models.items():
        with mlflow.start_run(run_name=model_name):
            print_debug(f"Training {model_name}...")
            
            # Train model
            model = model_info["model"]
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_valid = model.predict(X_valid)
            y_proba_valid = model.predict_proba(X_valid)[:, 1]
            
            # Calculate metrics
            train_metrics = {
                "train_accuracy": accuracy_score(y_train, y_pred_train),
                "train_f1": f1_score(y_train, y_pred_train, zero_division=0),
                "train_precision": precision_score(y_train, y_pred_train, zero_division=0),
                "train_recall": recall_score(y_train, y_pred_train, zero_division=0)
            }
            
            valid_metrics = {
                "valid_accuracy": accuracy_score(y_valid, y_pred_valid),
                "valid_f1": f1_score(y_valid, y_pred_valid, zero_division=0),
                "valid_precision": precision_score(y_valid, y_pred_valid, zero_division=0),
                "valid_recall": recall_score(y_valid, y_pred_valid, zero_division=0),
                "valid_auc": roc_auc_score(y_valid, y_proba_valid) if len(set(y_valid)) > 1 else 0.0
            }
            
            all_metrics = {**train_metrics, **valid_metrics}
            
            # Log to MLflow
            mlflow.log_params(model_info["params"])
            mlflow.log_metrics(all_metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Add tags
            mlflow.set_tags({
                "model_type": "decision_tree",
                "model_variant": model_name,
                "description": model_info["description"],
                "data_preprocessing": "standard_scaling"
            })
            
            # Store results
            result = {
                "model_name": model_name,
                "description": model_info["description"],
                **all_metrics,
                "run_id": mlflow.active_run().info.run_id
            }
            results.append(result)
            
            print_debug(f"{model_name} - Valid F1: {valid_metrics['valid_f1']:.4f}, Valid Accuracy: {valid_metrics['valid_accuracy']:.4f}")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.round(4)
    
    print_debug("\n=== BASELINE MODELS COMPARISON ===")
    print(comparison_df[['model_name', 'valid_f1', 'valid_accuracy', 'valid_precision', 'valid_recall', 'valid_auc']])
    
    return results, comparison_df

if __name__ == "__main__":
    results, comparison = train_tree_model()