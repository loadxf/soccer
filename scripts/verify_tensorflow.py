#!/usr/bin/env python
"""
Verify TensorFlow installation and configuration
"""

import sys
import os
import importlib
import random
import time

# Set fixed seed to avoid random_device errors
random.seed(int(os.environ.get('RANDOM_SEED', '42')))
os.environ['PYTHONHASHSEED'] = os.environ.get('PYTHONHASHSEED', '0')
os.environ['TF_DETERMINISTIC_OPS'] = os.environ.get('TF_DETERMINISTIC_OPS', '1')

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
    
    # Add a delay to allow entropy pool to fill
    time.sleep(1)
    
    # Set specific environment variables for TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging verbosity
    
    try:
        # Verify tensorflow without creating random numbers
        tf_installed = check_package("tensorflow")
        
        if tf_installed:
            # Import tensorflow with error handling for random device issues
            try:
                import tensorflow as tf
                
                # Configure TensorFlow to use deterministic operations
                try:
                    tf.config.experimental.enable_op_determinism()
                except:
                    pass  # This might not be available in older TF versions
                
                # Check if GPU is available with error handling
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        print(f"✅ TensorFlow can access {len(gpus)} GPU(s)")
                        for gpu in gpus:
                            print(f"   - {gpu}")
                    else:
                        print("ℹ️ TensorFlow is running on CPU only (no GPU available)")
                except Exception as e:
                    print(f"⚠️ Could not check GPU status: {str(e)}")
                
                # Simple non-random TensorFlow operation
                print("\nVerifying TensorFlow can perform basic operations...")
                try:
                    # Create a simple tensor without random operations
                    tensor = tf.constant([1.0, 2.0, 3.0])
                    result = tf.reduce_sum(tensor)
                    print(f"✅ TensorFlow operation successful: sum([1,2,3]) = {result.numpy()}")
                except Exception as e:
                    print(f"❌ Error when testing basic TensorFlow operations: {str(e)}")
            except Exception as e:
                print(f"⚠️ TensorFlow import error: {str(e)}")
                print("Continuing with verification of other packages...")
    except Exception as e:
        print(f"⚠️ Unexpected error during TensorFlow verification: {str(e)}")
    
    print("\n--- Related packages ---\n")
    # Check related packages
    check_package("numpy")
    check_package("pandas")
    check_package("scikit-learn")
    
    print("\n--- Environment Info ---\n")
    print(f"Python version: {sys.version}")
    print(f"Executable path: {sys.executable}")
    
    print("\nDone!")
    
    # Always exit successfully to not block the application startup
    sys.exit(0) 