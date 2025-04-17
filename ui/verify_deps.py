import streamlit as st
import importlib
import sys
import os
import subprocess

st.set_page_config(page_title="Dependency Verifier", page_icon="🔍", layout="wide")

st.title("Dependency Verification")
st.markdown("This tool checks if required dependencies are properly installed.")

# Function to check if a package is installed
def check_package(package_name):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("ML Libraries")
    
    # Check TensorFlow
    tf_installed, tf_version = check_package("tensorflow")
    if tf_installed:
        st.success(f"✅ TensorFlow is installed (version: {tf_version})")
        
        # Check if GPU is available
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            st.info(f"TensorFlow can access {len(gpus)} GPU(s)")
        else:
            st.warning("TensorFlow is running on CPU only (no GPU available)")
            
        # Test TensorFlow functionality
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            model = Sequential([Dense(1, input_shape=(1,))])
            model.compile(optimizer='adam', loss='mse')
            st.success("TensorFlow can compile models successfully")
        except Exception as e:
            st.error(f"Error testing TensorFlow: {str(e)}")
    else:
        st.error("❌ TensorFlow is NOT installed")
        
    # Check other ML libs
    libraries = ["sklearn", "xgboost", "lightgbm", "catboost", "prophet"]
    for lib in libraries:
        installed, version = check_package(lib)
        if installed:
            st.success(f"✅ {lib} is installed (version: {version})")
        else:
            st.error(f"❌ {lib} is NOT installed")

with col2:
    st.subheader("Data Processing Libraries")
    
    # Check data processing libraries
    data_libs = ["pandas", "numpy", "matplotlib", "seaborn", "plotly"]
    for lib in data_libs:
        installed, version = check_package(lib)
        if installed:
            st.success(f"✅ {lib} is installed (version: {version})")
        else:
            st.error(f"❌ {lib} is NOT installed")
    
    st.subheader("Environment Info")
    st.code(f"Python version: {sys.version}")
    st.code(f"Executable path: {sys.executable}")

st.subheader("Docker Environment")
if os.path.exists("/.dockerenv"):
    st.info("Running inside Docker container")
else:
    st.warning("Not running in Docker")

# Add button to install TensorFlow if missing
if not tf_installed:
    if st.button("Install TensorFlow"):
        with st.spinner("Installing TensorFlow..."):
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"], 
                                      capture_output=True, text=True, check=True)
                st.success("TensorFlow installed successfully! Please refresh the page.")
                st.code(result.stdout)
            except subprocess.CalledProcessError as e:
                st.error("Failed to install TensorFlow")
                st.code(e.stderr) 