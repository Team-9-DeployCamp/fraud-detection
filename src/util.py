import yaml
import joblib
from datetime import datetime
import os

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

def load_config() -> dict:
    # Try multiple possible config paths
    possible_paths = [
        '/app/configs/config.yaml',  # Docker path
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'config.yaml'),  # Local path
        'configs/config.yaml'  # Relative path
    ]
    
    for config_path in possible_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)
                    return config
            except Exception:
                continue
    
    # Default config if none found
    return {
        "print_debug": True,
        "mlflow": {
            "tracking_uri": "http://mlflow-server:5000",
            "experiment_name": "fraud_detection"
        }
    }

def pickle_load(file_path: str):
    # Convert relative path to absolute path if needed
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_path = os.path.join(project_root, file_path)
    
    # Load and return pickle file
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    # Convert relative path to absolute path if needed
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_path = os.path.join(project_root, file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Dump data into file
    joblib.dump(data, file_path)

params = load_config()
PRINT_DEBUG = params["print_debug"]

def print_debug(messages: str) -> None:
    # Check wheter user wants to use print or not
    if PRINT_DEBUG == True:
        print(time_stamp(), messages)