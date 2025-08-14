import subprocess
import sys
import os
import argparse

def run_training():
    """Run model training"""
    print("🔥 Starting fraud detection model training...")
    print("📊 This may take a few minutes depending on your data size...")
    
    try:
        # Change to src directory and run training
        os.chdir(os.path.join(os.path.dirname(__file__), 'src'))
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(os.path.dirname(__file__))
    
    return True

def test_inference():
    """Test the inference module"""
    print("🧪 Testing inference module...")
    
    try:
        os.chdir(os.path.join(os.path.dirname(__file__), 'src'))
        result = subprocess.run([sys.executable, "inference.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Inference test completed successfully!")
            print(result.stdout)
        else:
            print("❌ Inference test failed!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error testing inference: {e}")
        return False
    finally:
        os.chdir(os.path.dirname(__file__))
    
    return True

def run_api():
    """Run FastAPI server"""
    print("🚀 Starting Fraud Detection API server...")
    
    try:
        os.chdir(os.path.join(os.path.dirname(__file__), 'src'))
        subprocess.run([sys.executable, "api.py"])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped by user")
    except Exception as e:
        print(f"❌ Error running API server: {e}")
    finally:
        os.chdir(os.path.dirname(__file__))

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0", 
        "pydantic==2.5.0",
        "scikit-learn>=1.3.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "joblib>=1.2.0",
        "pyyaml>=6.0"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("command", choices=["install", "train", "test", "api", "full"], 
                       help="Command to run")
    
    args = parser.parse_args()
    
    print("🔍 Fraud Detection Pipeline 🔍")
    print("=" * 50)
    
    if args.command == "install":
        install_requirements()
        
    elif args.command == "train":
        if run_training():
            print("\n✅ Training pipeline completed successfully!")
        else:
            print("\n❌ Training pipeline failed!")
            
    elif args.command == "test":
        if test_inference():
            print("\n✅ Testing completed successfully!")
        else:
            print("\n❌ Testing failed!")
            
    elif args.command == "api":
        run_api()
        
    elif args.command == "full":
        print("🔄 Running full pipeline...")
        
        # Step 1: Install requirements
        if not install_requirements():
            print("❌ Failed to install requirements. Stopping.")
            return
        
        # Step 2: Train model
        if not run_training():
            print("❌ Failed to train model. Stopping.")
            return
            
        # Step 3: Test inference
        if not test_inference():
            print("❌ Failed to test inference. Stopping.")
            return
            
        print("\n🎉 Full pipeline completed successfully!")
        print("📝 You can now start the API with: python run_pipeline.py api")

def print_help():
    """Print usage instructions"""
    print("""
🔍 Fraud Detection Pipeline Usage:

1. Install dependencies:
   python run_pipeline.py install

2. Train the model:
   python run_pipeline.py train

3. Test inference:
   python run_pipeline.py test

4. Start API server:
   python run_pipeline.py api

5. Run full pipeline (install + train + test):
   python run_pipeline.py full

📖 API Documentation:
   Once the API is running, visit: http://127.0.0.1:8000/docs
    """)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_help()
    else:
        main()
