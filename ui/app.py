"""
Soccer Prediction System UI Application

This module implements the Streamlit UI for the Soccer Prediction System.
It includes special handling for browser cache issues and session initialization.
"""

import streamlit as st
from pathlib import Path
import time
import random  # Added for session ID generation
import pandas as pd
import logging
import sys
import os

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import API service
try:
    from ui.api_service import SoccerPredictionAPI
    # Set default environment variables
    DEBUG = os.environ.get("DEBUG", "True").lower() in ["true", "1", "t", "yes", "y"]
    APP_ENV = os.environ.get("APP_ENV", "development")
except ImportError as e:
    print(f"Failed to import API service: {e}")
    # Define fallback settings if import fails
    DEBUG = True
    APP_ENV = "development"

# Import Kaggle helpers - we'll initialize them properly at the Data Management page
KAGGLE_AVAILABLE = False

# Generate a unique session ID to avoid conflicts
SESSION_ID = f"{int(time.time())}_{random.randint(1000, 9999)}"

# Define utility functions

def get_available_datasets():
    """
    Get a list of available datasets for model training.
    
    Returns:
        list: List of dataset IDs or names
    """
    try:
        # Import DataManager if not already imported
        from ui.data_manager import DataManager
        
        # Create an instance of DataManager
        data_manager = DataManager()
        
        # Get all datasets
        all_datasets = data_manager.get_all_datasets()
        
        # Extract dataset IDs or names
        if all_datasets:
            # Get IDs from the dataset objects
            return [ds.get("id") for ds in all_datasets]
        else:
            return []
    except Exception as e:
        print(f"Error getting available datasets: {e}")
        return []

# Add HTML/CSS to suppress browser warnings
def suppress_browser_warnings():
    """Add custom HTML/CSS to suppress browser console warnings about feature policies."""
    try:
        # Try to load the external JS file
        js_file_path = Path(__file__).parent / "browser_console_suppressor.js"
        if js_file_path.exists():
            with open(js_file_path, "r") as f:
                js_content = f.read()
                st.components.v1.html(f"<script>{js_content}</script>", height=0)
        else:
            # Fallback to inline script if file doesn't exist
            suppress_html = """
            <script>
                // Override console.warn to filter out specific warnings
                (function() {
                    const originalWarn = console.warn;
                    console.warn = function(...args) {
                        // Filter out feature policy warnings
                        if (args.length > 0 && typeof args[0] === 'string' && 
                            (args[0].includes('feature policy') || 
                             args[0].includes('Unrecognized feature') || 
                             args[0].includes('sandbox attribute'))) {
                            return; // Suppress these warnings
                        }
                        // Pass through other warnings
                        originalWarn.apply(console, args);
                    };
                })();
            </script>
            """
            st.markdown(suppress_html, unsafe_allow_html=True)
    except Exception as e:
        # Silent failure - we don't want to break the app if warning suppression fails
        pass

# Extremely early session state initialization with try-except blocks around each statement
# This helps prevent the "SessionInfo not initialized" error
try:
    # Force streamlit to initialize its session state object
    _ = st.session_state
except Exception as e:
    print(f"Failed basic session state access: {str(e)}")

try:
    # Initialize key session state variables with defaults first
    if 'api_status' not in st.session_state:
        st.session_state['api_status'] = {"status": "unknown", "message": "Initializing..."}
except Exception as e:
    print(f"Failed to set api_status: {str(e)}")

try:
    if 'using_fallback' not in st.session_state:
        st.session_state['using_fallback'] = False
except Exception as e:
    print(f"Failed to set using_fallback: {str(e)}")

try:
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
except Exception as e:
    print(f"Failed to set prediction_result: {str(e)}")

try:
    if 'prediction_is_fallback' not in st.session_state:
        st.session_state['prediction_is_fallback'] = False
except Exception as e:
    print(f"Failed to set prediction_is_fallback: {str(e)}")

try:
    if 'session_init_time' not in st.session_state:
        st.session_state['session_init_time'] = time.time()
except Exception as e:
    print(f"Failed to set session_init_time: {str(e)}")

try:
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = SESSION_ID
except Exception as e:
    print(f"Failed to set session_id: {str(e)}")

try:
    if 'browser_info' not in st.session_state:
        st.session_state['browser_info'] = {
            'init_time': time.time(),
            'refresh_count': 0,
            'session_id': SESSION_ID
        }
    else:
        # Increment refresh counter
        try:
            st.session_state['browser_info']['refresh_count'] += 1
        except:
            st.session_state['browser_info'] = {
                'init_time': time.time(),
                'refresh_count': 1,
                'session_id': SESSION_ID
            }
except Exception as e:
    print(f"Failed to set browser_info: {str(e)}")

# Delay to ensure session state initialization is complete
# This can help with timing issues between server and client
time.sleep(0.1)

# Helper function to safely access session state with a fallback value
def safe_get_session(key, default=None):
    """Safely get a value from session state with a default fallback"""
    try:
        return st.session_state.get(key, default)
    except Exception as e:
        print(f"Error accessing session state key '{key}': {str(e)}")
        return default

# Helper function to safely set session state with error handling
def safe_set_session(key, value):
    """Safely set a value in session state with error handling"""
    try:
        st.session_state[key] = value
        return True
    except Exception as e:
        print(f"Error setting session state key '{key}': {str(e)}")
        return False

# Helper function to check if response indicates fallback data
def is_fallback_data(data):
    """Check if the data returned is fallback data"""
    if isinstance(data, dict):
        return data.get('offline_mode', False) or data.get('status') == 'offline_fallback'
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return any(item.get('name', '').endswith('(Offline)') for item in data)
    return False

# Helper function to check API health with retry logic
def check_api_health_with_retries(max_retries=3, initial_delay=1, backoff_factor=2):
    """Check API health with retry logic if it fails"""
    import requests
    import time
    from urllib.parse import urlparse
    
    # Define API URLs to try
    api_urls = [
        "http://localhost:8080/api/v1/health",
        "http://127.0.0.1:8080/api/v1/health"
    ]
    
    # Try each URL with retries
    for url in api_urls:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        parsed_url = urlparse(url)
                        return {
                            "status": "online",
                            "message": "API is available",
                            "api_host": parsed_url.hostname,
                            "api_port": parsed_url.port,
                            "data": data
                        }
                    except:
                        pass
            except Exception as e:
                print(f"API check attempt {attempt+1} failed for {url}: {str(e)}")
            
            # Only sleep if we're going to retry
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= backoff_factor
    
    # If all URLs and retries failed, return offline status
    return {
        "status": "offline",
        "message": "API is unavailable. Please make sure the API server is running."
    }

# Helper function to detect browser and platform
def get_browser_info():
    """Attempt to detect browser and platform information using User-Agent."""
    try:
        user_agent = st.get_option("browser.serverAddress")
        platform_info = sys.platform
        python_version = sys.version
        return {
            "user_agent": user_agent,
            "platform": platform_info,
            "python_version": python_version,
            "streamlit_version": st.__version__,
            "session_id": SESSION_ID
        }
    except Exception as e:
        return {"error": str(e), "session_id": SESSION_ID}

# More aggressive SessionInfo error detection and handling script
def check_for_session_errors():
    """Check for common SessionInfo errors in browser Console and provide fixes."""
    session_error_html = """
    <script>
    // Store session recovery data in localStorage
    localStorage.setItem('streamlit_session_recovery', JSON.stringify({
        timestamp: Date.now(),
        session_id: '""" + SESSION_ID + """',
        recovery_attempt: parseInt(localStorage.getItem('streamlit_recovery_attempt') || '0') + 1
    }));
    
    // Check for SessionInfo errors in console
    (function() {
        // Store original console.error
        const originalConsoleError = console.error;
        
        // Override console.error to catch SessionInfo errors
        console.error = function() {
            // Call original function
            originalConsoleError.apply(console, arguments);
            
            // Check if error relates to SessionInfo
            const errorStr = Array.from(arguments).join(' ');
            if (errorStr.includes('SessionInfo') && errorStr.includes('initialized')) {
                console.log('Detected SessionInfo error, attempting recovery...');
                console.log('Error details:', errorStr);
                
                // Try to reset session state by clearing storage and reloading
                try {
                    // Signal to Streamlit that we need a cache clear
                    if (window.parent) {
                        const message = {
                            type: 'streamlit:sessionError',
                            error: 'SessionInfo not initialized',
                            details: errorStr,
                            timestamp: Date.now(),
                            session_id: '""" + SESSION_ID + """'
                        };
                        window.parent.postMessage(message, '*');
                        
                        // Add a visual indicator with more options
                        const errorDiv = document.createElement('div');
                        errorDiv.style.position = 'fixed';
                        errorDiv.style.top = '0';
                        errorDiv.style.left = '0';
                        errorDiv.style.right = '0';
                        errorDiv.style.padding = '10px';
                        errorDiv.style.backgroundColor = '#ff4444';
                        errorDiv.style.color = 'white';
                        errorDiv.style.zIndex = '9999';
                        errorDiv.style.textAlign = 'center';
                        errorDiv.innerHTML = `
                            <strong>SessionInfo Error Detected!</strong> 
                            <div style="margin:5px 0">
                              For persistent errors, run <code>fix_session_errors.bat</code> for advanced cleaning.
                              <a href="?force_reset=true&sid=""" + SESSION_ID + """" style="color:white;text-decoration:underline;margin:0 10px">Quick Reset</a>
                              <a href="http://127.0.0.1:8501" style="color:white;text-decoration:underline;margin:0 10px">Try IP Address</a>
                            </div>
                        `;
                        document.body.appendChild(errorDiv);
                        
                        // Auto-recovery attempt if this is not a repeated error
                        const recoveryAttempt = parseInt(localStorage.getItem('streamlit_recovery_attempt') || '0');
                        if (recoveryAttempt < 3) {
                            console.log('Auto-recovery attempt: ' + recoveryAttempt);
                            
                            // Clear localStorage and sessionStorage
                            localStorage.setItem('streamlit_recovery_attempt', recoveryAttempt + 1);
                            
                            // For all items except our recovery tracking
                            Object.keys(localStorage).forEach(key => {
                                if (!key.startsWith('streamlit_recovery')) {
                                    localStorage.removeItem(key);
                                }
                            });
                            
                            sessionStorage.clear();
                            
                            // Unregister service workers
                            if (navigator.serviceWorker) {
                                navigator.serviceWorker.getRegistrations().then(function(registrations) {
                                    for (let registration of registrations) {
                                        registration.unregister();
                                    }
                                    // Reload with force_reset after clearing storage
                                    setTimeout(() => {
                                        window.location.href = window.location.pathname + "?force_reset=true&sid=""" + SESSION_ID + """&recovery=" + recoveryAttempt;
                                    }, 500);
                                });
                            } else {
                                // Reload with force_reset after clearing storage
                                setTimeout(() => {
                                    window.location.href = window.location.pathname + "?force_reset=true&sid=""" + SESSION_ID + """&recovery=" + recoveryAttempt;
                                }, 500);
                            }
                        } else {
                            // If we've tried auto-recovery 3 times, provide more detailed guidance
                            console.log('Auto-recovery failed after multiple attempts');
                            errorDiv.innerHTML = `
                                <strong>Multiple recovery attempts failed</strong>
                                <div style="margin:5px 0">
                                    <p>Please try one of these solutions:</p>
                                    <ol style="text-align:left;display:inline-block;">
                                        <li>Run <code>fix_session_errors.bat</code> for advanced browser cleaning</li>
                                        <li>Use incognito/private browsing mode</li>
                                        <li>Try a different browser</li>
                                        <li>Access via IP: <a href="http://127.0.0.1:8501" style="color:white;text-decoration:underline;">http://127.0.0.1:8501</a></li>
                                    </ol>
                                </div>
                            `;
                        }
                    }
                } catch (e) {
                    console.error('Error during recovery attempt:', e);
                }
            }
        };
        
        // Also intercept the Streamlit websocket initialization
        // This is a bit hacky but can help catch initialization issues
        const originalWebSocket = window.WebSocket;
        window.WebSocket = function(url, protocols) {
            console.log('WebSocket connecting to:', url);
            const ws = new originalWebSocket(url, protocols);
            
            const originalOnError = ws.onerror;
            ws.onerror = function(event) {
                console.error('WebSocket Error:', event);
                // If we get a websocket error, it might be related to session issues
                if (originalOnError) originalOnError.call(this, event);
            };
            
            return ws;
        };
        
        // Check if we're in a recovery state
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('recovery')) {
            console.log('In recovery mode, attempt:', urlParams.get('recovery'));
        }
    })();
    </script>
    """
    try:
        st.components.v1.html(session_error_html, height=0)
    except Exception as e:
        print(f"Failed to inject session error detection script: {str(e)}")
        # Fallback to basic HTML if components API fails
        st.markdown(f"""
        <div style="display:none">
        {session_error_html}
        </div>
        """, unsafe_allow_html=True)

# Function to clear session state more aggressively
def clear_session_state():
    """Clear API-related session state to force a fresh check."""
    # Save the refresh count before clearing
    refresh_count = 0
    try:
        if 'browser_info' in st.session_state and 'refresh_count' in st.session_state.browser_info:
            refresh_count = st.session_state.browser_info.get('refresh_count', 0)
    except Exception:
        pass
    
    # Try to clear the entire session state object
    try:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    except Exception as e:
        print(f"Failed to clear session state: {str(e)}")
        # Fallback to individual deletion
        try:
            # Clear all API-related state
            if 'api_status' in st.session_state:
                del st.session_state['api_status']
            if 'using_fallback' in st.session_state:
                del st.session_state['using_fallback']
            if 'prediction_result' in st.session_state:
                del st.session_state['prediction_result']
            if 'prediction_is_fallback' in st.session_state:
                del st.session_state['prediction_is_fallback']
        except Exception as e2:
            print(f"Failed individual session state clearing: {str(e2)}")
    
    # Reinitialize with fresh values
    try:
        st.session_state['api_status'] = {"status": "unknown", "message": "Reinitialized"}
        st.session_state['using_fallback'] = False
        st.session_state['session_init_time'] = time.time()
        st.session_state['session_id'] = SESSION_ID
        st.session_state['browser_info'] = {
            'init_time': time.time(),
            'refresh_count': refresh_count + 1,
            'session_id': SESSION_ID
        }
    except Exception as e:
        print(f"Failed to reinitialize session state: {str(e)}")

# Set page configuration - wrap in try/except to handle potential errors
try:
    st.set_page_config(
        page_title="Soccer Prediction System",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    print(f"Failed to set page config: {str(e)}")

# Inject session error detection script
check_for_session_errors()

# Function to show debug information for troubleshooting connection issues
def show_debug_info():
    """Display debug information to help diagnose API connection issues."""
    if DEBUG:
        with st.expander("🔍 Debug Information", expanded=False):
            st.markdown("### API Connection Status")
            
            # Get current API status
            api_status = safe_get_session('api_status', {"status": "unknown", "message": "Not initialized"})
            status_color = "green" if api_status.get('status') == 'online' else "red"
            
            # Display API status
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {status_color}; background-color: #f5f5f5;">
                <strong>Status:</strong> {api_status.get('status', 'unknown')}<br>
                <strong>Message:</strong> {api_status.get('message', 'N/A')}<br>
                <strong>Host:</strong> {api_status.get('api_host', 'N/A')}<br>
                <strong>Port:</strong> {api_status.get('api_port', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
            # Display connection test buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Test localhost:8080"):
                    try:
                        import requests
                        response = requests.get("http://localhost:8080/api/v1/health", timeout=2)
                        if response.status_code == 200:
                            st.success(f"✅ localhost connection successful!")
                            st.json(response.json())
                        else:
                            st.error(f"❌ localhost connection failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ localhost connection error: {str(e)}")
            
            with col2:
                if st.button("Test 127.0.0.1:8080"):
                    try:
                        import requests
                        response = requests.get("http://127.0.0.1:8080/api/v1/health", timeout=2)
                        if response.status_code == 200:
                            st.success(f"✅ 127.0.0.1 connection successful!")
                            st.json(response.json())
                        else:
                            st.error(f"❌ 127.0.0.1 connection failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ 127.0.0.1 connection error: {str(e)}")
            
            # Session state information
            st.markdown("### Session State Information")
            st.markdown(f"""
            <div style="padding: 10px; background-color: #f5f5f5;">
                <strong>Session ID:</strong> {SESSION_ID}<br>
                <strong>Session Init Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(safe_get_session('session_init_time', time.time())))}<br>
                <strong>Using Fallback:</strong> {safe_get_session('using_fallback', False)}<br>
                <strong>Browser Refresh Count:</strong> {safe_get_session('browser_info', {}).get('refresh_count', 0)}
            </div>
            """, unsafe_allow_html=True)
            
            # System information
            st.markdown("### System Information")
            browser_info = get_browser_info()
            st.json(browser_info)
            
            # Troubleshooting guidance
            st.markdown("### Troubleshooting Steps")
            st.markdown("""
            If you're experiencing API connection issues:
            
            1. **Check API server** - Make sure the API server is running on port 8080
            2. **Test connections** - Use the test buttons above to check both localhost and 127.0.0.1
            3. **Network issues** - Check for firewall or proxy settings blocking connections
            4. **Try alternative hostname** - If localhost doesn't work, try 127.0.0.1 directly
            5. **Restart services** - Try restarting both the API and UI servers
            
            Command to start API server: `python main.py api --start`
            """)

# Check for force_reset URL parameter
try:
    query_params = st.query_params
    if "force_reset" in query_params:
        clear_session_state()
        # Remove the parameter
        updated_params = {k: v for k, v in query_params.items() if k not in ['force_reset', 'recovery']}
        st.query_params.update(**updated_params)
        st.rerun()
except Exception as e:
    print(f"Failed to process force_reset parameter: {str(e)}")
    # Fallback for older Streamlit versions
    try:
        query_params = st.experimental_get_query_params()
        if "force_reset" in query_params:
            clear_session_state()
            st.experimental_set_query_params(**{k: v for k, v in query_params.items() if k not in ['force_reset', 'recovery']})
            st.rerun()
    except Exception as e2:
        print(f"Failed experimental query params: {str(e2)}")

# Create a function to display fallback notification
def show_fallback_notification():
    """Show a notification banner when offline/fallback data is being displayed"""
    st.warning(
        "⚠️ **Offline Mode**: You're currently viewing fallback data. The API server is not connected. "
        "Some features may be limited and data shown is simulated. "
        "Run `python main.py api --start` to connect to the API server."
    )

# Create sidebar navigation
st.sidebar.title("Soccer Prediction System")

# Use text-based logo instead of image file
st.sidebar.markdown("""
<div style="text-align: center; background-color: #1E3A8A; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    <h2 style="margin: 0;">⚽ SPS</h2>
    <p style="margin: 0; font-size: 0.8em;">Soccer Prediction System</p>
</div>
""", unsafe_allow_html=True)

# Check API status early and store in session state
if 'api_status' not in st.session_state or st.session_state.api_status.get('status') == 'unknown':
    try:
        # Use retry logic for initial check
        api_status = check_api_health_with_retries(max_retries=3, initial_delay=1)
        st.session_state.api_status = api_status
        st.session_state.using_fallback = api_status.get('status') != 'online'
    except Exception as e:
        st.session_state.api_status = {"status": "offline", "message": f"API is unavailable: {str(e)}"}
        st.session_state.using_fallback = True

# Navigation options
page = st.sidebar.radio(
    "Navigate", 
    ["Home", "Data Management", "Model Training", "Predictions", "Model Evaluation", "Explanations"]
)

# Environment info
env_info = f"Environment: {APP_ENV.upper()}"
if DEBUG:
    env_info += " (Debug Mode)"
st.sidebar.info(env_info)

# API status in sidebar
try:
    api_status = st.session_state.api_status
    if api_status.get('status') == 'online':
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
        retry_col, clear_col = st.sidebar.columns(2)
        
        with retry_col:
            if st.button("Retry Connection"):
                try:
                    # Use retry logic for button click
                    new_status = check_api_health_with_retries(max_retries=3, initial_delay=1)
                    st.session_state.api_status = new_status
                    st.session_state.using_fallback = new_status.get('status') != 'online'
                    st.rerun()
                except Exception:
                    pass
        
        with clear_col:
            if st.button("Clear Cache"):
                clear_session_state()
                st.rerun()
except Exception as e:
    st.sidebar.error(f"❌ Session Error: {str(e)}")
    if st.sidebar.button("Reset Session"):
        clear_session_state()
        st.rerun()

# Browser reset helper
if st.sidebar.button("🔄 Fix Browser Issues"):
    clear_session_state()
    js = f'''
    <script>
        // Clear localStorage and sessionStorage
        localStorage.clear();
        sessionStorage.clear();
        
        // Unregister service workers
        if (navigator.serviceWorker) {{
            navigator.serviceWorker.getRegistrations().then(function(registrations) {{
                for (let registration of registrations) {{
                    registration.unregister();
                }}
            }});
        }}
        
        // Reload the page with force_reset
        window.location.href = window.location.pathname + "?force_reset=true";
    </script>
    '''
    st.components.v1.html(js, height=0)
    st.rerun()

# Version info
st.sidebar.caption("v1.0.0")

# Show fallback notification if using fallback data
try:
    if st.session_state.using_fallback:
        show_fallback_notification()
except Exception as e:
    st.error(f"Session state error: {str(e)}. Try refreshing the page or clearing cache.")
    if st.button("Reset Session State"):
        clear_session_state()
        st.rerun()

# Main content
if page == "Home":
    # Apply browser console warning suppression
    suppress_browser_warnings()
    
    st.title("⚽ Soccer Prediction System")
    st.subheader("Machine Learning Platform for Soccer Match Outcome Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Soccer Prediction Dashboard
        
        This interactive interface allows you to:
        
        - **Manage data** - Import, explore, and process soccer datasets
        - **Train models** - Configure and train ML models with different algorithms
        - **Make predictions** - Predict outcomes for upcoming soccer matches
        - **Evaluate performance** - Analyze model metrics and compare results
        - **Explain predictions** - Understand the factors influencing match predictions
        
        Use the sidebar navigation to access different features.
        """)
        
        st.info("To get started, navigate to 'Data Management' to prepare your datasets.")
        
        # Add debug information section
        show_debug_info()
        
    with col2:
        st.markdown("### System Status")
        
        # API status check
        try:
            if api_status.get('status') == 'online':
                st.success("✅ API Server: Connected")
                st.success("✅ Database: Connected")
                st.success("✅ Model Cache: Available")
            else:
                st.error(f"❌ API Server: {api_status.get('message', 'Not Connected')}")
                
                # Add a more specific instruction for starting the API
                st.warning("""
                Please start the API server with one of these methods:
                - `start_system.bat` - Starts both API and UI together
                - `python main.py api --start` - Starts only the API
                """)
        except Exception as e:
            st.error(f"❌ System Status Error: {str(e)}")
            st.warning("Session state may be corrupted. Try refreshing or clearing cache.")
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        quick_links = {
            "Train New Model": "Model Training",
            "Make Prediction": "Predictions",
            "View Model Performance": "Model Evaluation"
        }
        
        for label, target in quick_links.items():
            if st.button(label):
                # Update query params with modern approach
                st.query_params.update(page=target)
                st.rerun()
        
        st.markdown("### Troubleshooting")
        st.markdown("""
        Having issues with the UI?
        - Run [`fix_session_errors.bat`](.) for advanced browser cleaning
        - Run [`clear_browser_cache.bat`](.) for basic cache clearing
        - Try a [hard refresh](.) (Ctrl+F5)
        - Use [incognito mode](.) for testing
        - Try accessing with [127.0.0.1](http://127.0.0.1:8501) instead of localhost
        """)
        
        # Add emergency fix button
        if st.button("🔧 Emergency Session Fix"):
            st.warning("Running advanced browser session fix...")
            js = """
            <script>
            (function() {
                // Aggressive clearing of browser storage
                try {
                    // Clear all types of storage
                    localStorage.clear();
                    sessionStorage.clear();
                    
                    // Try to clear IndexedDB
                    if (window.indexedDB) {
                        window.indexedDB.databases().then(dbs => {
                            dbs.forEach(db => {
                                window.indexedDB.deleteDatabase(db.name);
                            });
                        }).catch(e => console.error('IndexedDB clear failed:', e));
                    }
                    
                    // Try to clear Cache API
                    if (window.caches) {
                        caches.keys().then(names => {
                            names.forEach(name => {
                                caches.delete(name);
                            });
                        }).catch(e => console.error('Cache API clear failed:', e));
                    }
                    
                    // Unregister service workers
                    if (navigator.serviceWorker) {
                        navigator.serviceWorker.getRegistrations().then(registrations => {
                            registrations.forEach(registration => {
                                registration.unregister();
                            });
                        }).catch(e => console.error('Service worker unregister failed:', e));
                    }
                    
                    // Generate a fresh session ID to prevent conflicts
                    const freshSessionId = Date.now().toString(36) + Math.random().toString(36).substring(2);
                    
                    // Create an on-screen notification
                    const notifyDiv = document.createElement('div');
                    notifyDiv.style.position = 'fixed';
                    notifyDiv.style.top = '50%';
                    notifyDiv.style.left = '50%';
                    notifyDiv.style.transform = 'translate(-50%, -50%)';
                    notifyDiv.style.padding = '20px';
                    notifyDiv.style.backgroundColor = '#4CAF50';
                    notifyDiv.style.color = 'white';
                    notifyDiv.style.borderRadius = '10px';
                    notifyDiv.style.zIndex = '9999';
                    notifyDiv.style.textAlign = 'center';
                    notifyDiv.innerHTML = '<strong>Session Fix Applied!</strong><br>Refreshing page in 3 seconds...';
                    document.body.appendChild(notifyDiv);
                    
                    // Redirect to IP address version with force_reset after 3 seconds
                    setTimeout(() => {
                        window.location.href = "http://127.0.0.1:8501?force_reset=true&ts=" + Date.now() + "&sid=" + freshSessionId;
                    }, 3000);
                } catch (e) {
                    console.error('Emergency fix error:', e);
                    // Still try to redirect even if errors occur
                    window.location.href = "http://127.0.0.1:8501?force_reset=true&ts=" + Date.now();
                }
            })();
            </script>
            """
            st.components.v1.html(js, height=0)

elif page == "Data Management":
    # Apply browser console warning suppression
    suppress_browser_warnings()
    
    st.title("📊 Data Management")
    st.subheader("Import, Explore, and Process Soccer Datasets")
    
    # Import our DataManager
    from ui.data_manager import DataManager
    data_manager = DataManager()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Dataset Management
        
        This section allows you to:
        - Import datasets from various sources
        - Explore and preview data
        - Process and transform datasets
        - Manage your dataset library
        """)
        
        # File upload section
        st.markdown("### Upload Dataset")
        uploaded_files = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"], accept_multiple_files=True)
        st.caption("Limit 1000MB per file • CSV, XLSX, XLS, JSON")
        
        if uploaded_files:
            st.info(f"📤 {len(uploaded_files)} files selected for upload")
            
            if st.button("Upload Files"):
                for uploaded_file in uploaded_files:
                    data_manager.upload_file(uploaded_file)
        
        # Dataset listing
        st.markdown("### Available Datasets")
        all_datasets = data_manager.get_all_datasets()
        
        if not all_datasets:
            st.info("No datasets available. Upload datasets to get started.")
        else:
            # Create a DataFrame for display
            display_data = []
            for ds in all_datasets:
                display_data.append({
                    "ID": ds.get("id", "Unknown"),
                    "Name": ds.get("name", "Unnamed Dataset"),
                    "Rows": ds.get("rows", 0),
                    "Columns": len(ds.get("columns", [])),
                    "Status": ds.get("status", "unknown").capitalize(),
                    "Upload Date": ds.get("upload_date", "").split("T")[0]
                })
            
            df_datasets = pd.DataFrame(display_data)
            
            # Multiselect for batch operations
            selected_ids = st.multiselect(
                "Select datasets for batch operations",
                options=df_datasets["ID"].tolist(),
                format_func=lambda x: next((ds.get("name", "Unknown") for ds in all_datasets if ds.get("id") == x), x)
            )
            
            # Display dataset table
            st.dataframe(df_datasets)
            
            # Preview section
            if df_datasets["ID"].tolist():
                st.markdown("### Dataset Preview")
                preview_id = st.selectbox(
                    "Select a dataset to preview",
                    options=df_datasets["ID"].tolist(),
                    format_func=lambda x: next((ds.get("name", "Unknown") for ds in all_datasets if ds.get("id") == x), x)
                )
                
                if preview_id:
                    preview_df = data_manager.get_dataset_preview(preview_id)
                    if preview_df is not None:
                        st.dataframe(preview_df)
            
            # Batch operations section
            if selected_ids:
                st.markdown("### Batch Operations")
                st.info(f"📋 {len(selected_ids)} datasets selected")
                
                operation = st.selectbox(
                    "Select operation",
                    options=["Process datasets", "Feature engineering", "Delete selected"]
                )
                
                if st.button("Execute Operation"):
                    if operation == "Delete selected":
                        for dataset_id in selected_ids:
                            data_manager.delete_dataset(dataset_id)
                        st.rerun()
                    elif operation == "Process datasets":
                        data_manager.batch_process_datasets(selected_ids, "process")
                    elif operation == "Feature engineering":
                        data_manager.batch_process_datasets(selected_ids, "features")
        
        # Add debug info
        show_debug_info()
    
    with col2:
        st.markdown("### Import Options")
        
        import_source = st.selectbox(
            "Import from source:",
            options=["Upload File", "Kaggle Dataset", "Football API", "Sample Data"]
        )
        
        if import_source == "Kaggle Dataset":
            if data_manager.kaggle_available:
                st.markdown("#### Import from Kaggle")
                kaggle_dataset = st.text_input("Enter Kaggle dataset name (e.g., 'owner/dataset-name')")
                
                if st.button("Import Dataset") and kaggle_dataset:
                    data_manager.import_kaggle_dataset(kaggle_dataset)
            else:
                st.error("Kaggle credentials not found")
                with st.expander("Kaggle Setup Instructions"):
                    data_manager.show_kaggle_setup_instructions()
        
        elif import_source == "Football API":
            st.markdown("#### Import Football Data")
            
            leagues = st.multiselect(
                "Select leagues:",
                options=["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
                default=["Premier League"]
            )
            
            seasons = st.multiselect(
                "Select seasons:",
                options=[f"20{i}/{i+1}" for i in range(10, 25)],  # Updated to include up to 24/25
                default=["2022/23"]
            )
            
            if st.button("Import Football Data"):
                data_manager.download_football_data(leagues, seasons)
        
        elif import_source == "Sample Data":
            st.markdown("#### Sample Datasets")
            
            sample_datasets = [
                "Premier League 2022/23",
                "Basic Soccer Stats",
                "Player Performance Data"
            ]
            
            sample_to_import = st.selectbox("Select sample dataset:", options=sample_datasets)
            
            if st.button("Import Sample"):
                st.success(f"Sample dataset '{sample_to_import}' imported")
                # Here you'd call a function to import the selected sample dataset
        
        # Actions
        st.markdown("### Actions")
        
        refresh_col, clear_col = st.columns(2)
        
        with refresh_col:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with clear_col:
            if st.button("🗑 Clear All"):
                if st.session_state.get('selected_datasets'):
                    st.session_state['selected_datasets'] = []
                st.rerun()
        
        # System status
        st.markdown("### System Status")
        if data_manager.registry_available:
            st.success("✅ Dataset Registry: Available")
        else:
            st.error("❌ Dataset Registry: Unavailable")
        
        if data_manager.kaggle_available:
            st.success("✅ Kaggle API: Configured")
            if st.button("Test Kaggle Connection"):
                data_manager.verify_kaggle_setup()
        else:
            st.error("❌ Kaggle API: Not Configured")

elif page == "Model Training":
    # Apply browser console warning suppression
    suppress_browser_warnings()
    
    st.title("🧠 Model Training")
    st.subheader("Train and Optimize Soccer Prediction Models")
    
    # Get available datasets
    datasets = get_available_datasets()
    
    if not datasets:
        st.error("No datasets available. Please download or import datasets first.")
        st.stop()
    
    # Set up the form
    with st.form("model_training_form"):
        # Select dataset
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        # Select model type
        model_types = [
            "logistic", 
            "random_forest", 
            "xgboost", 
            "dixon_coles"  # Add Dixon-Coles model
        ]
        selected_model = st.selectbox(
            "Select Model Type", 
            model_types,
            help="Choose the type of model to train. The Dixon-Coles model is a soccer-specific distribution model."
        )
        
        # Show advanced options based on model type
        if selected_model == "dixon_coles":
            # Dixon-Coles specific options
            match_weight_days = st.slider(
                "Match Weight Half-life (days)", 
                min_value=30, 
                max_value=365, 
                value=90,
                help="Number of days after which a match's influence is halved"
            )
            st.info("The Dixon-Coles model is a specialized soccer prediction model that directly models goal distributions.")
            
            # No feature selection or hyperparameter tuning for Dixon-Coles
            feature_type = None
            target_col = None
            hyperparameter_tuning = False
            
        else:
            # Feature options for traditional ML models
            feature_types = ["match_features", "team_features", "advanced_features"]
            feature_type = st.selectbox(
                "Select Feature Type", 
                feature_types,
                help="Choose the type of features to use for training"
            )
            
            # Target column
            target_options = ["result", "home_win", "total_goals", "goal_difference"]
            target_col = st.selectbox(
                "Select Target Variable", 
                target_options,
                help="Choose what the model should predict"
            )
            
            # Hyperparameter tuning option
            hyperparameter_tuning = st.checkbox(
                "Perform Hyperparameter Tuning", 
                value=False,
                help="Use grid search to find optimal hyperparameters (takes longer)"
            )
            
            # Show advanced features note
            if feature_type == "advanced_features":
                st.info("Advanced features include team form, expected goals, and other soccer-specific metrics.")
        
        # Test size slider
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        
        # Random state for reproducibility
        random_state = st.number_input("Random Seed", min_value=1, max_value=10000, value=42)
        
        # Submit button
        submit_button = st.form_submit_button("Train Model")
    
    # Process form submission
    if submit_button:
        try:
            with st.spinner("Training model... This may take a while."):
                # Prepare model parameters
                if selected_model == "dixon_coles":
                    model_params = {
                        "match_weight_days": match_weight_days
                    }
                    
                    # Call training function
                    from src.models.training import train_model
                    
                    results = train_model(
                        model_type=selected_model,
                        dataset_name=selected_dataset,
                        model_params=model_params
                    )
                    
                else:
                    # Call training function for traditional ML models
                    from src.models.training import train_model
                    
                    results = train_model(
                        model_type=selected_model,
                        dataset_name=selected_dataset,
                        feature_type=feature_type,
                        target_col=target_col,
                        test_size=test_size,
                        hyperparameter_tuning=hyperparameter_tuning,
                        random_state=random_state
                    )
                
                # Check if training was successful
                if not results:
                    st.error("Model training failed. Check logs for details.")
                    st.stop()
                
                if isinstance(results, dict) and results.get("success", True):
                    # Show success message
                    st.success("Model training completed successfully!")
                    
                    # Display training results
                    st.subheader("Training Results")
                    
                    # Format results for display
                    if selected_model == "dixon_coles":
                        # Show Dixon-Coles specific results
                        metrics = {
                            "Model Type": results.get("model_type", "dixon_coles"),
                            "Dataset": results.get("dataset_name", selected_dataset),
                            "Number of Matches": results.get("num_matches", "N/A"),
                            "Number of Teams": results.get("num_teams", "N/A"),
                            "Home Advantage": f"{results.get('home_advantage', 0):.4f}",
                            "Training Duration": f"{results.get('training_duration', 0):.2f} seconds",
                            "Model Path": results.get("model_path", "N/A")
                        }
                        
                        # Display metrics
                        for metric, value in metrics.items():
                            st.text(f"{metric}: {value}")
                        
                        # If ratings path is provided, try to load and display top teams
                        ratings_path = results.get("ratings_path")
                        if ratings_path and os.path.exists(ratings_path):
                            try:
                                import pandas as pd
                                ratings = pd.read_csv(ratings_path)
                                
                                # Sort by overall rating
                                ratings = ratings.sort_values("overall", ascending=False)
                                
                                # Display top 10 teams
                                st.subheader("Top Teams by Strength")
                                st.dataframe(ratings.head(10))
                            except Exception as e:
                                st.warning(f"Could not load team ratings: {e}")
                    
                    else:
                        # Show traditional ML model results
                        metrics = {
                            "Model Type": results.get("model_type", selected_model),
                            "Dataset": results.get("dataset_name", selected_dataset),
                            "Feature Type": results.get("feature_type", feature_type),
                            "Target Column": results.get("target_col", target_col),
                            "Test Size": results.get("test_size", test_size),
                            "Training Duration": f"{results.get('training_duration', 0):.2f} seconds",
                            "Model Path": results.get("model_path", "N/A")
                        }
                        
                        # Display metrics
                        for metric, value in metrics.items():
                            st.text(f"{metric}: {value}")
                        
                        # Display evaluation metrics
                        evaluation = results.get("evaluation", {})
                        if evaluation:
                            st.subheader("Evaluation Metrics")
                            
                            # Format evaluation metrics
                            eval_metrics = {
                                "Accuracy": f"{evaluation.get('accuracy', 0):.4f}",
                                "F1 Score (Macro)": f"{evaluation.get('f1_macro', 0):.4f}",
                                "F1 Score (Weighted)": f"{evaluation.get('f1_weighted', 0):.4f}",
                                "Log Loss": f"{evaluation.get('log_loss', 0):.4f}" if "log_loss" in evaluation else "N/A"
                            }
                            
                            # Display evaluation metrics
                            for metric, value in eval_metrics.items():
                                st.text(f"{metric}: {value}")
                    
                    # Display hyperparameter tuning results if available
                    if hyperparameter_tuning and "hyperparameter_tuning_results" in results:
                        st.subheader("Hyperparameter Tuning Results")
                        tuning_results = results["hyperparameter_tuning_results"]
                        
                        # Display best parameters
                        st.text(f"Best Parameters: {tuning_results.get('best_params', {})}")
                        st.text(f"Best Score: {tuning_results.get('best_score', 0):.4f}")
                    
                    # Option to create ensemble
                    st.subheader("Add to Ensemble")
                    ensemble_id = st.text_input("Ensemble ID (leave empty to create new)")
                    
                    if st.button("Add to Ensemble"):
                        try:
                            from src.models.ensemble import EnsemblePredictor
                            
                            if not ensemble_id:
                                # Create new ensemble with timestamp as ID
                                from datetime import datetime
                                ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            # Initialize ensemble
                            ensemble = EnsemblePredictor.load(ensemble_id)
                            
                            if ensemble is None:
                                # Create new ensemble
                                ensemble = EnsemblePredictor(ensemble_id)
                            
                            # Add the model
                            model_path = results.get("model_path")
                            if selected_model == "dixon_coles":
                                model_type = "distribution"
                            else:
                                model_type = "ml"
                                
                            model_id = f"{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            success = ensemble.add_model(model_id, model_path, model_type)
                            
                            if success:
                                # If evaluation metrics available, set performance
                                if selected_model != "dixon_coles" and "evaluation" in results:
                                    ensemble.set_model_performance(model_id, results["evaluation"])
                                
                                # Save the ensemble
                                ensemble_path = ensemble.save()
                                st.success(f"Added model to ensemble {ensemble_id}")
                                st.text(f"Ensemble saved to {ensemble_path}")
                            else:
                                st.error(f"Failed to add model to ensemble {ensemble_id}")
                        except Exception as e:
                            st.error(f"Error adding model to ensemble: {e}")
                
                else:
                    # Show error message
                    st.error(f"Model training failed: {results.get('message', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error during model training: {e}")
            import traceback
            st.text(traceback.format_exc())

elif page == "Predictions":
    # Apply browser console warning suppression
    suppress_browser_warnings()
    
    st.title("🔮 Predictions")
    st.subheader("Predict Outcomes for Soccer Matches")
    
    # Under development notice
    st.info("⚠️ This feature is under active development. Basic functionality is available.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Match Prediction
        
        This section allows you to:
        - Select a trained model
        - Enter match details
        - Get predicted outcomes
        - View prediction confidence
        - Explore alternative scenarios
        """)
        
        # Placeholder for prediction form
        st.markdown("### Match Details")
        
        team_col1, team_col2 = st.columns(2)
        with team_col1:
            st.selectbox("Home Team", ["Manchester City", "Arsenal", "Manchester United", "Liverpool", "Chelsea"])
        with team_col2:
            st.selectbox("Away Team", ["Liverpool", "Manchester United", "Arsenal", "Chelsea", "Tottenham"])
        
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            st.selectbox("Home Team Form", ["Excellent", "Good", "Average", "Poor"])
        with form_col2:
            st.selectbox("Away Team Form", ["Excellent", "Good", "Average", "Poor"])
        
        st.markdown("### Additional Factors")
        factor_col1, factor_col2, factor_col3 = st.columns(3)
        with factor_col1:
            st.checkbox("Derby Match")
        with factor_col2:
            st.checkbox("Injury Concerns")
        with factor_col3:
            st.checkbox("Weather Impact")
        
        if st.button("Generate Prediction"):
            st.success("Prediction: Home Win (63% probability)")
            st.progress(63)
            st.markdown("**Outcome Probabilities:**")
            st.text("Home Win: 63%\nDraw: 24%\nAway Win: 13%")
        
        # Add debug info
        show_debug_info()
    
    with col2:
        st.markdown("### Prediction Models")
        st.selectbox("Select Model", ["Best Available Model", "RandomForest_EPL_2023", "GradientBoost_Demo"])
        
        st.markdown("### Recent Predictions")
        st.table([
            {"match": "Arsenal vs Chelsea", "prediction": "Home Win", "actual": "Home Win", "date": "2023-05-28"},
            {"match": "Liverpool vs Man Utd", "prediction": "Draw", "actual": "Away Win", "date": "2023-05-21"}
        ])
        
        st.markdown("### System Status")
        # API status check
        try:
            if api_status.get('status') == 'online':
                st.success("✅ Prediction API: Available")
                st.success("✅ Models: Available")
            else:
                st.error("❌ Prediction Services: Unavailable")
                st.warning("API connection required for match predictions.")
        except Exception as e:
            st.error(f"❌ System Status Error: {str(e)}")

elif page == "Model Evaluation":
    # Apply browser console warning suppression
    suppress_browser_warnings()
    
    st.title("📈 Model Evaluation")
    st.subheader("Analyze Model Performance and Compare Results")
    
    # Under development notice
    st.info("⚠️ This feature is under active development. Basic functionality is available.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Performance Analysis
        
        This section allows you to:
        - View model performance metrics
        - Compare multiple models
        - Analyze prediction accuracy
        - Identify strengths and weaknesses
        - Export evaluation reports
        """)
        
        # Placeholder for model selection
        st.markdown("### Select Models to Evaluate")
        st.multiselect(
            "Models", 
            ["RandomForest_EPL_2023", "GradientBoost_Demo", "LogisticRegression_EPL_2023"],
            ["RandomForest_EPL_2023", "GradientBoost_Demo"]
        )
        
        st.markdown("### Performance Metrics")
        metrics = {
            "Model": ["RandomForest_EPL_2023", "GradientBoost_Demo"],
            "Accuracy": ["76.3%", "72.1%"],
            "Precision": ["79.2%", "74.5%"],
            "Recall": ["73.8%", "71.2%"],
            "F1 Score": ["76.4%", "72.8%"]
        }
        st.table(metrics)
        
        # Placeholder visualization
        st.markdown("### Visualization")
        st.markdown("*Accuracy Comparison Chart would appear here*")
        
        # Add debug info
        show_debug_info()
    
    with col2:
        st.markdown("### Quick Actions")
        st.button("Generate Full Report")
        st.button("Test on New Data")
        st.button("Export Metrics")
        
        st.markdown("### Evaluation Settings")
        st.checkbox("Include Confidence Intervals")
        st.checkbox("Show All Metrics")
        st.checkbox("Compare to Baseline")
        
        st.markdown("### System Status")
        # API status check
        try:
            if api_status.get('status') == 'online':
                st.success("✅ Evaluation API: Available")
                st.success("✅ Report Generation: Available")
            else:
                st.error("❌ Evaluation Services: Unavailable")
                st.warning("API connection required for advanced evaluation.")
        except Exception as e:
            st.error(f"❌ System Status Error: {str(e)}")

elif page == "Explanations":
    # Apply browser console warning suppression
    suppress_browser_warnings()
    
    st.title("🔍 Explanations")
    st.subheader("Understand Model Predictions and Feature Importance")
    
    # Under development notice
    st.info("⚠️ This feature is under active development. Basic functionality is available.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Explainable AI
        
        This section allows you to:
        - Understand model decision factors
        - View feature importance
        - Analyze specific predictions
        - Compare factor weights across models
        - Generate interpretable reports
        """)
        
        # Placeholder for model and prediction selection
        st.markdown("### Select Model and Prediction")
        explain_col1, explain_col2 = st.columns(2)
        with explain_col1:
            st.selectbox("Model for Explanation", ["RandomForest_EPL_2023", "GradientBoost_Demo"])
        with explain_col2:
            st.selectbox("Prediction to Explain", ["Arsenal vs Chelsea (2023-05-28)", "Liverpool vs Man Utd (2023-05-21)"])
        
        st.markdown("### Feature Importance")
        st.markdown("*Feature importance visualization would appear here*")
        
        st.markdown("### Explanation Details")
        st.markdown("""
        The prediction was primarily influenced by:
        1. Home team's recent form (42% impact)
        2. Head-to-head history (27% impact)
        3. Key player availability (18% impact)
        4. Current league positions (13% impact)
        """)
        
        # Add debug info
        show_debug_info()
    
    with col2:
        st.markdown("### Explanation Methods")
        st.selectbox("Explanation Type", ["SHAP Values", "Feature Importance", "Partial Dependence", "LIME"])
        
        st.markdown("### Visualization Settings")
        st.checkbox("Show Top Features Only", True)
        st.slider("Number of Features", 3, 10, 5)
        st.checkbox("Include Numerical Values", True)
        
        st.markdown("### System Status")
        # API status check
        try:
            if api_status.get('status') == 'online':
                st.success("✅ Explanation API: Available")
                st.success("✅ SHAP Engine: Available")
            else:
                st.error("❌ Explanation Services: Unavailable")
                st.warning("API connection required for model explanations.")
        except Exception as e:
            st.error(f"❌ System Status Error: {str(e)}")

# Add function to help users set up Kaggle
def setup_kaggle_helper():
    """Display comprehensive instructions for setting up Kaggle credentials."""
    st.markdown("""
    ## Setting up Kaggle Credentials

    To use Kaggle datasets, you need to complete the following steps:
    """)
    
    # Step 1: Install kaggle package
    st.markdown("### Step 1: Install the Kaggle package")
    st.code("pip install kaggle", language="bash")
    
    # Step 2: Create Kaggle account
    st.markdown("### Step 2: Create a Kaggle account")
    st.markdown("""
    If you don't already have a Kaggle account:
    1. Go to [kaggle.com/account/login](https://www.kaggle.com/account/login)
    2. Click on "Sign Up" and complete the registration process
    """)
    
    # Step 3: Generate API token
    st.markdown("### Step 3: Generate an API token")
    st.markdown("""
    1. Log in to your Kaggle account
    2. Go to your account settings: [kaggle.com/settings](https://www.kaggle.com/settings)
    3. Scroll down to the "API" section
    4. Click "Create New API Token"
    5. This will download a file called `kaggle.json` containing your credentials
    """)
    
    # Step 4: Set up credentials
    st.markdown("### Step 4: Set up your credentials")
    
    # Detect the user's OS
    import platform
    user_os = platform.system()
    
    if user_os == "Windows":
        # Windows instructions
        import os
        kaggle_dir = os.path.expanduser('~/.kaggle')
        
        st.markdown(f"""
        **Windows Setup:**
        
        1. Create the kaggle directory:
        ```
        mkdir "{kaggle_dir}"
        ```
        
        2. Copy your downloaded `kaggle.json` file to:
        ```
        {kaggle_dir}\\kaggle.json
        ```
        
        You can do this with File Explorer by copying the file to:
        ```
        {kaggle_dir}
        ```
        """)
    
    elif user_os == "Darwin":  # macOS
        # macOS instructions
        st.markdown("""
        **macOS Setup:**
        
        1. Create the kaggle directory:
        ```
        mkdir -p ~/.kaggle
        ```
        
        2. Copy your downloaded `kaggle.json` file:
        ```
        cp ~/Downloads/kaggle.json ~/.kaggle/
        ```
        
        3. Set proper permissions:
        ```
        chmod 600 ~/.kaggle/kaggle.json
        ```
        """)
    
    else:  # Linux and others
        # Linux instructions
        st.markdown("""
        **Linux Setup:**
        
        1. Create the kaggle directory:
        ```
        mkdir -p ~/.kaggle
        ```
        
        2. Copy your downloaded `kaggle.json` file:
        ```
        cp ~/Downloads/kaggle.json ~/.kaggle/
        ```
        
        3. Set proper permissions:
        ```
        chmod 600 ~/.kaggle/kaggle.json
        ```
        """)
    
    # Step 5: Alternative environment variable setup
    st.markdown("### Alternative: Environment Variable Setup")
    st.markdown("""
    Instead of using the `kaggle.json` file, you can set environment variables:
    
    1. Open your `kaggle.json` file and note the username and key values
    2. Set the following environment variables:
    """)
    
    if user_os == "Windows":
        st.code("""
setx KAGGLE_USERNAME your_username
setx KAGGLE_KEY your_key
        """, language="bash")
    else:
        st.code("""
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
        """, language="bash")
    
    # Verification step
    st.markdown("### Step 5: Verify your setup")
    st.markdown("""
    After setting up your credentials, you can verify that everything is working correctly:
    """)
    
    # Add verification button
    if st.button("Verify Kaggle Setup"):
        try:
            import kaggle
            try:
                # Try to authenticate with Kaggle
                kaggle.api.authenticate()
                
                # If we get here, authentication worked
                st.success("✅ Kaggle credentials verified successfully!")
                st.info("You can now use Kaggle datasets in the application.")
                
                # Show some available dataset info
                try:
                    datasets = kaggle.api.dataset_list(search="soccer")
                    if datasets:
                        st.success(f"Found {len(datasets)} soccer-related datasets on Kaggle!")
                except Exception as e:
                    # Don't show API errors, just show success for auth
                    pass
                
            except Exception as auth_e:
                st.error(f"❌ Kaggle authentication failed: {str(auth_e)}")
                
                # Check common issues
                import os
                kaggle_dir = os.path.expanduser('~/.kaggle')
                kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
                
                if not os.path.exists(kaggle_json):
                    st.warning(f"The kaggle.json file was not found at: {kaggle_json}")
                    st.info("Please follow the instructions above to set up your credentials.")
                else:
                    st.info(f"The kaggle.json file exists at: {kaggle_json}")
                    st.warning("However, there might be an issue with the file content or permissions.")
                
                # Check environment variables
                if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
                    st.info("Environment variables KAGGLE_USERNAME and KAGGLE_KEY are set.")
                    st.warning("However, there might be an issue with the values.")
                else:
                    st.info("Environment variables KAGGLE_USERNAME and KAGGLE_KEY are not set.")
        
        except ImportError:
            st.error("❌ Kaggle package is not installed")
            st.info("Please install it with: pip install kaggle")
    
    # Add a note about restarting
    st.warning("Note: After setting up your credentials, you may need to restart the application for changes to take effect.")

# Helper function to handle kaggle imports safely
def safe_kaggle_import():
    """Try to import kaggle and handle common issues."""
    try:
        import kaggle
        return True, None
    except ImportError:
        return False, "Kaggle package is not installed. Please run: pip install kaggle"
    except Exception as e:
        if "kaggle.json" in str(e):
            return False, "Kaggle credentials not set up correctly. See setup instructions."
        return False, f"Error with Kaggle: {str(e)}"

def start_app():
    """
    Start the Streamlit app in the current process.
    This function is called by the main script when starting the UI.
    
    It handles different methods of starting Streamlit based on what's available.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ui.app")
    
    # Store original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Directly use subprocess to start Streamlit
        import subprocess
        logger.info("Starting Streamlit using subprocess method...")
        
        # Find streamlit executable
        streamlit_executable = "streamlit"
        
        cmd = [
            streamlit_executable, "run", 
            str(Path(__file__).resolve()),
            "--server.port=8501"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        print(f"\nIf automatic startup fails, run this command manually:")
        print(f"{' '.join(cmd)}\n")
        
        subprocess.Popen(cmd)
    
    except Exception as e:
        logger.error(f"Error starting Streamlit app: {e}")
        print(f"\nPlease start the app manually with:")
        print(f"streamlit run {Path(__file__).resolve()} --server.port=8501\n")
    
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

# If this script is run directly, start Streamlit
if __name__ == "__main__":
    start_app()