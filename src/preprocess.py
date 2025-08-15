import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from . import util as utils

def check_data(input_data, config):
    """Data validation function from notebook"""
    # count input data
    len_input_data = len(input_data)

    # check data types
    assert set(input_data.select_dtypes('float').columns.to_list()) == set(config['float_columns']), 'an error occurs in float columns'
    assert set(input_data.select_dtypes('int').columns.to_list()) == set(config['int_columns']), 'an error occurs in int columns'
    assert set(input_data.select_dtypes('object').columns.to_list()) == set(config['categorical_columns']), 'an error occurs in categorical columns'

    # Check range of data in float_columns
    for col in config['float_columns']:
        range_key = f"range_{col}"       
        if range_key in config:
            min_val, max_val = config[range_key]
            out_of_range_count = (len_input_data - input_data[col].between(min_val, max_val).sum())
            assert out_of_range_count == 0, \
                f"Error in {col} range. {out_of_range_count} values are outside the defined range [{min_val}, {max_val}]."

    # Check range of data in int_columns
    for col in config['int_columns']:
        range_key = f"range_{col}"       
        if range_key in config:
            min_val, max_val = config[range_key]
            out_of_range_count = (len_input_data - input_data[col].between(min_val, max_val).sum())
            assert out_of_range_count == 0, \
                f"Error in {col} range. {out_of_range_count} values are outside the defined range [{min_val}, {max_val}]."

def load_raw_data(config):
    """Load raw data from CSV and perform basic cleaning"""
    try:
        # Load data
        dataset = pd.read_csv(config['data_path'])
        utils.print_debug(f"✓ Raw data loaded: {dataset.shape}")
        
        # Drop transaction_id (same as notebook)
        if 'transaction_id' in dataset.columns:
            dataset = dataset.drop(columns=['transaction_id'], axis=1)
            utils.print_debug("✓ Dropped transaction_id column")
        
        # Data validation
        utils.print_debug("Performing data validation...")
        check_data(dataset, config)
        utils.print_debug("✓ Data validation passed")
        
        return dataset
    except Exception as e:
        utils.print_debug(f"✗ Error loading raw data: {e}")
        raise

def preprocess_data(config):
    """
    Main preprocessing function following notebook pattern:
    - Load and validate raw data
    - Split: 70% train, 15% valid, 15% test
    - Apply One-Hot Encoding
    - Save processed data and encoders
    """
    utils.print_debug("Starting data preprocessing...")
    
    # Load raw data
    dataset = load_raw_data(config)
    
    # Basic data info
    utils.print_debug(f"Dataset shape: {dataset.shape}")
    utils.print_debug(f"Dataset columns: {dataset.columns.tolist()}")
    utils.print_debug(f"Dataset info:")
    utils.print_debug(f"Data types:\n{dataset.dtypes}")
    utils.print_debug(f"Target distribution:\n{dataset[config['label']].value_counts()}")
    
    # Prepare features and target (same as notebook)
    feature_columns = config['float_columns'] + config['categorical_columns']
    X = dataset[feature_columns].copy()
    y = dataset[config['label']].copy()
    
    utils.print_debug(f"Features: {feature_columns}")
    utils.print_debug(f"X shape: {X.shape}")
    utils.print_debug(f"y shape: {y.shape}")
    
    # Data splitting (exactly same as notebook)
    utils.print_debug("Splitting data: 70% train, 15% valid, 15% test...")
    
    # split data train and test
    X_train, X_pretest, y_train, y_pretest = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # split data test into set validation = valid
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_pretest, y_pretest,
        test_size=0.5,
        random_state=42,
        stratify=y_pretest
    )
    
    # Verify splitting
    total_samples = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]
    if total_samples == len(dataset):
        utils.print_debug("✓ Data splitting is correct")
    else:
        utils.print_debug("✗ Error in data splitting")
        raise ValueError("Data splitting error")
    
    utils.print_debug(f"Train set: {X_train.shape}, {y_train.shape}")
    utils.print_debug(f"Valid set: {X_valid.shape}, {y_valid.shape}")
    utils.print_debug(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Save raw splits first (same as notebook)
    utils.print_debug("Saving raw datasets...")
    utils.pickle_dump(dataset, config["dataset_fraud_detection_cleaned_path"])
    utils.pickle_dump(X_train, config['train_set_path'][0])
    utils.pickle_dump(y_train, config['train_set_path'][1])
    utils.pickle_dump(X_valid, config['valid_set_path'][0])
    utils.pickle_dump(y_valid, config['valid_set_path'][1])
    utils.pickle_dump(X_test, config['test_set_path'][0])
    utils.pickle_dump(y_test, config['test_set_path'][1])
    
    # Apply One-Hot Encoding for processed data
    utils.print_debug("Applying One-Hot Encoding...")
    
    # Create and fit encoders on training data
    merchant_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    device_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit encoders
    merchant_encoder.fit(X_train[['merchant_type']])
    device_encoder.fit(X_train[['device_type']])
    
    utils.print_debug(f"Merchant categories: {merchant_encoder.categories_[0]}")
    utils.print_debug(f"Device categories: {device_encoder.categories_[0]}")
    
    # Function to apply one-hot encoding
    def apply_ohe_encoding(X_data, merchant_enc, device_enc):
        """Apply one-hot encoding to data"""
        X_processed = X_data.copy()
        
        # Transform merchant_type
        merchant_encoded = merchant_enc.transform(X_data[['merchant_type']])
        merchant_feature_names = merchant_enc.get_feature_names_out(['merchant_type'])
        merchant_df = pd.DataFrame(
            merchant_encoded, 
            columns=merchant_feature_names,
            index=X_data.index
        )
        
        # Transform device_type
        device_encoded = device_enc.transform(X_data[['device_type']])
        device_feature_names = device_enc.get_feature_names_out(['device_type'])
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
    
    # Apply encoding to all sets
    X_train_processed = apply_ohe_encoding(X_train, merchant_encoder, device_encoder)
    X_valid_processed = apply_ohe_encoding(X_valid, merchant_encoder, device_encoder)
    X_test_processed = apply_ohe_encoding(X_test, merchant_encoder, device_encoder)
    
    utils.print_debug(f"Processed train shape: {X_train_processed.shape}")
    utils.print_debug(f"Processed valid shape: {X_valid_processed.shape}")
    utils.print_debug(f"Processed test shape: {X_test_processed.shape}")
    utils.print_debug(f"Processed features: {X_train_processed.columns.tolist()}")
    
    # Save processed data
    utils.print_debug("Saving processed data...")
    
    # Save processed datasets
    utils.pickle_dump(X_train_processed, 'dataset/processed/X_train_processed.pkl')
    utils.pickle_dump(y_train, 'dataset/processed/y_train_processed.pkl')  # y tidak berubah
    utils.pickle_dump(X_valid_processed, 'dataset/processed/X_valid_processed.pkl')
    utils.pickle_dump(y_valid, 'dataset/processed/y_valid_processed.pkl')
    utils.pickle_dump(X_test_processed, 'dataset/processed/X_test_processed.pkl')
    utils.pickle_dump(y_test, 'dataset/processed/y_test_processed.pkl')
    
    # Save encoders
    utils.print_debug("Saving encoders...")
    utils.pickle_dump(merchant_encoder, config['merchant_encoder_path'])
    utils.pickle_dump(device_encoder, config['device_encoder_path'])
    
    utils.print_debug("✓ Preprocessing completed successfully!")
    
    # Return summary
    summary = {
        'dataset_shape': dataset.shape,
        'train_shape': X_train_processed.shape,
        'valid_shape': X_valid_processed.shape,
        'test_shape': X_test_processed.shape,
        'features': X_train_processed.columns.tolist(),
        'target_distribution': {
            'train': y_train.value_counts().to_dict(),
            'valid': y_valid.value_counts().to_dict(),
            'test': y_test.value_counts().to_dict()
        },
        'merchant_categories': merchant_encoder.categories_[0].tolist(),
        'device_categories': device_encoder.categories_[0].tolist()
    }
    
    return summary

def load_processed_data(config):
    """Load processed data for training"""
    try:
        # Load processed training data
        X_train = utils.pickle_load('dataset/processed/X_train_processed.pkl')
        y_train = utils.pickle_load('dataset/processed/y_train_processed.pkl')
        X_valid = utils.pickle_load('dataset/processed/X_valid_processed.pkl')
        y_valid = utils.pickle_load('dataset/processed/y_valid_processed.pkl')
        X_test = utils.pickle_load('dataset/processed/X_test_processed.pkl')
        y_test = utils.pickle_load('dataset/processed/y_test_processed.pkl')
        
        utils.print_debug("✓ Processed data loaded successfully")
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
        
    except Exception as e:
        utils.print_debug(f"✗ Error loading processed data: {e}")
        raise

def load_encoders(config):
    """Load saved encoders"""
    try:
        merchant_encoder = utils.pickle_load(config['merchant_encoder_path'])
        device_encoder = utils.pickle_load(config['device_encoder_path'])
        
        utils.print_debug("✓ Encoders loaded successfully")
        
        return merchant_encoder, device_encoder
        
    except Exception as e:
        utils.print_debug(f"✗ Error loading encoders: {e}")
        raise

if __name__ == "__main__":
    # Test preprocessing
    config = utils.load_config()
    summary = preprocess_data(config)
    
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Dataset shape: {summary['dataset_shape']}")
    print(f"Train shape: {summary['train_shape']}")
    print(f"Valid shape: {summary['valid_shape']}")
    print(f"Test shape: {summary['test_shape']}")
    print(f"Features: {summary['features']}")
    print(f"Target distribution: {summary['target_distribution']}")
    print(f"Merchant categories: {summary['merchant_categories']}")
    print(f"Device categories: {summary['device_categories']}")