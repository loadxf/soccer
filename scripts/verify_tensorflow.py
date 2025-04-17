#!/usr/bin/env python
"""
Verify TensorFlow installation and configuration
"""

import sys
import os
import importlib

def check_package(package_name):
    try:
        module = importlib.import_module(package_name)
        print(f"✅ {package_name} is installed (version: {module.__version__})")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

if __name__ == "__main__":
    print("\n--- TensorFlow Verification ---\n")
    
    # Check TensorFlow installation
    tf_installed = check_package("tensorflow")
    
    if tf_installed:
        # Import tensorflow for further checks
        import tensorflow as tf
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow can access {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("ℹ️ TensorFlow is running on CPU only (no GPU available)")
        
        # Check if TensorFlow can perform a simple operation
        print("\nVerifying TensorFlow can perform computations...")
        try:
            # Create a simple model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            
            model = Sequential([Dense(1, input_shape=(1,))])
            model.compile(optimizer='adam', loss='mse')
            print("✅ Successfully created and compiled a Keras model")
        except Exception as e:
            print(f"❌ Error when testing TensorFlow: {str(e)}")
    
    print("\n--- Related packages ---\n")
    # Check related packages
    check_package("numpy")
    check_package("pandas")
    check_package("scikit-learn")
    
    print("\n--- Environment Info ---\n")
    print(f"Python version: {sys.version}")
    print(f"Executable path: {sys.executable}")
    
    print("\nDone!") 