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
    import os
    from urllib.parse import urlparse
    
    # Check if we're running in Docker
    is_docker = os.path.exists('/.dockerenv')
    
    # Define base API hosts to try
    api_hosts = []
    
    # Check for remote host in environment variable
    remote_host = os.environ.get('REMOTE_API_HOST')
    remote_port = os.environ.get('API_PORT', '8000')
    
    # If remote host is defined, prioritize it
    if remote_host:
        api_hosts.append((remote_host, remote_port))
        print(f"Will check remote API host: {remote_host}:{remote_port}")
    
    # In Docker, try the container name next
    if is_docker:
        api_hosts.append(("app", "8000"))
    
    # Then try localhost and IP address
    api_hosts.extend([
        ("localhost", "8000"),
        ("127.0.0.1", "8000")
    ])
    
    # Define endpoint patterns to try for each host
    endpoint_patterns = [
        "/health",             # Direct health endpoint
        "/api/v1/health",      # API v1 health endpoint
        "/api/health"          # Alternative health endpoint
    ]
    
    # Generate all URLs to try (host + endpoint combinations)
    api_urls = []
    for host, port in api_hosts:
        for endpoint in endpoint_patterns:
            api_urls.append(f"http://{host}:{port}{endpoint}")
    
    # Print which URLs we're checking
    print(f"API URLs to check: {api_urls}")
    
    # Try each URL with retries
    for url in api_urls:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt+1} for {url}")
                response = requests.get(url, timeout=5)  # Increased timeout
                if response.status_code == 200:
                    try:
                        data = response.json()
                        parsed_url = urlparse(url)
                        return {
                            "status": "online",
                            "message": "API is available",
                            "api_host": parsed_url.hostname,
                            "api_port": parsed_url.port,
                            "endpoint": parsed_url.path,
                            "data": data
                        }
                    except Exception as e:
                        print(f"Failed to parse JSON from {url}: {str(e)}")
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
                if st.button("Test localhost:8000"):
                    try:
                        import requests
                        response = requests.get("http://localhost:8000/health", timeout=2)
                        if response.status_code == 200:
                            st.success(f"✅ localhost connection successful!")
                            st.json(response.json())
                        else:
                            st.error(f"❌ localhost connection failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ localhost connection error: {str(e)}")
            
            with col2:
                if st.button("Test 127.0.0.1:8000"):
                    try:
                        import requests
                        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
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
            
            1. **Check API server** - Make sure the API server is running on port 8000
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

# Add a sidebar with navigation and settings
def create_sidebar():
    """Create the sidebar with navigation and settings"""
    with st.sidebar:
        st.title("Soccer Prediction System")
        
        # Add navigation
        st.header("Navigation")
        
        # Main Navigation
        pages = {
            "Home": "Home page with overview and recent matches",
            "Predictions": "Make match outcome predictions",
            "Data Management": "Manage datasets and features",
            "Model Management": "Train and manage prediction models",
            "Evaluation": "Evaluate model performance",
            "System Status": "View system health and status",
            "Dependency Check": "Verify system dependencies"
        }
        
        selected_page = st.radio("Select a page", list(pages.keys()), 
                                index=0, help="Navigate between different sections of the app")
        
        # Display help text for the selected page
        st.info(pages[selected_page])
        
        # Show API status
        api_status = safe_get_session('api_status', {"status": "unknown", "message": "Initializing..."})
        st.write("---")
        
        # System Version
        st.caption("Version: 0.9.5-beta")
        
        # Return the selected page
        return selected_page

# Add the dependency page to the page mapping
def show_verify_deps_page():
    """Show the dependency verification page"""
    import ui.verify_deps
    # This imports and runs the verify_deps.py script

# Main navigation handler
def main():
    """Main navigation handler"""
    
    # Create the sidebar
    selected_page = create_sidebar()
    
    # Suppress browser warnings
    suppress_browser_warnings()
    
    # Navigate to the selected page
    if selected_page == "Home":
        show_home_page()
    elif selected_page == "Predictions":
        show_predictions_page()
    elif selected_page == "Data Management":
        from ui.data_manager import show_data_management_page
        show_data_management_page()
    elif selected_page == "Model Management":
        from ui.model_manager import show_model_management_page
        show_model_management_page()
    elif selected_page == "Evaluation":
        from ui.evaluation import show_evaluation_page
        show_evaluation_page()
    elif selected_page == "System Status":
        show_system_status_page()
    elif selected_page == "Dependency Check":
        show_verify_deps_page()

# If this script is run directly, start Streamlit
if __name__ == "__main__":
    start_app()