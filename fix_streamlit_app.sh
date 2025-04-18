#!/bin/bash
set -e

echo "=== Fixing Streamlit App Python Error ==="
echo

# Create backup of the ui/app.py file
echo "Creating backup of ui/app.py..."
mkdir -p backup
[ -f ui/app.py ] && cp ui/app.py backup/app.py.backup || echo "app.py not found locally, will check in container"

# Function to fix the app.py file
fix_app_py() {
  local APP_PY_PATH="$1"
  
  echo "Fixing $APP_PY_PATH..."
  
  # Check if the file exists
  if [ ! -f "$APP_PY_PATH" ]; then
    echo "❌ Error: $APP_PY_PATH does not exist."
    return 1
  fi
  
  # Check if the file calls start_app but doesn't define it
  if grep -q "start_app()" "$APP_PY_PATH" && ! grep -q "def start_app" "$APP_PY_PATH"; then
    echo "Found call to start_app() but no function definition. Adding function definition..."
    
    # Create a temporary file with the fixed content
    cat > temp_fix.py << 'EOL'
# Define the missing start_app function
def start_app():
    """Main entry point to the application."""
    # Set up page configuration
    st.set_page_config(
        page_title="Soccer Prediction System",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Include custom CSS
    with open("ui/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Run the app
    main()

# Only call start_app if this script is run directly, not when imported
if __name__ == "__main__":
    start_app()
EOL
    
    # Append the fix to the file if it doesn't already have a main guard
    if ! grep -q "if __name__ == \"__main__\":" "$APP_PY_PATH"; then
      echo "Adding missing function and main guard..."
      cat temp_fix.py >> "$APP_PY_PATH"
      echo "✅ Function added successfully."
    else
      echo "File already has a main guard. Need to modify differently..."
      
      # Create a new file by replacing the direct start_app() call with proper guard
      if grep -q "start_app()" "$APP_PY_PATH" && ! grep -q "if __name__ == \"__main__\": *\n *start_app()" "$APP_PY_PATH"; then
        echo "Replacing direct start_app() call with proper guard..."
        # Use sed to replace the direct call with function definition + guard
        sed -i.bak 's/start_app()/\
# Define the missing start_app function\
def start_app():\
    """Main entry point to the application."""\
    # Set up page configuration\
    st.set_page_config(\
        page_title="Soccer Prediction System",\
        page_icon="⚽",\
        layout="wide",\
        initial_sidebar_state="expanded"\
    )\
    \
    # Include custom CSS\
    with open("ui\/styles.css") as f:\
        st.markdown(f"<style>{f.read()}<\/style>", unsafe_allow_html=True)\
    \
    # Run the app\
    main()\
\
# Only call start_app if this script is run directly, not when imported\
if __name__ == "____main__":\
    start_app()/g' "$APP_PY_PATH"
        echo "✅ Function added successfully."
      fi
    fi
    
    # Clean up
    rm -f temp_fix.py
  elif grep -q "def start_app" "$APP_PY_PATH"; then
    echo "✅ start_app function already defined. No changes needed."
  else
    echo "❌ Cannot determine how to fix app.py. Manual intervention required."
    echo "The error is: NameError: name 'start_app' is not defined"
    echo "Please check the file and add a start_app function or modify the code accordingly."
  fi
}

# Try to fix the local file if it exists
if [ -f ui/app.py ]; then
  fix_app_py "ui/app.py"
  
  echo
  echo "=== Deploying Fix to Remote Server ==="
  echo "To deploy this fix to the remote server, run:"
  echo "scp ui/app.py root@103.163.186.204:~/soccer/ui/"
  echo "ssh root@103.163.186.204 'cd ~/soccer && docker compose restart ui'"
else
  echo "Local ui/app.py not found. Creating a remote fix script..."
  
  cat > remote_fix_app.sh << 'EOL'
#!/bin/bash
set -e

echo "=== Fixing Streamlit App Python Error on Remote Server ==="

cd ~/soccer

# Backup the app.py file
echo "Creating backup of app.py..."
mkdir -p backup
cp ui/app.py backup/app.py.backup

# Extract the ui container ID
UI_CONTAINER=$(docker ps -a --filter "name=ui" --format "{{.ID}}")

if [ -z "$UI_CONTAINER" ]; then
  echo "❌ UI container not found. Please start it first."
  exit 1
fi

# Function to fix the app.py file within the container
echo "Fixing app.py in the container..."

# Create a temporary fix file
cat > /tmp/fix_app.py << 'EOF'
# Define the missing start_app function
def start_app():
    """Main entry point to the application."""
    # Set up page configuration
    st.set_page_config(
        page_title="Soccer Prediction System",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Include custom CSS
    with open("ui/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Run the app
    main()

# Only call start_app if this script is run directly, not when imported
if __name__ == "__main__":
    start_app()
EOF

# Append the fix to the end of the app.py file if needed
if grep -q "start_app()" ui/app.py && ! grep -q "def start_app" ui/app.py; then
  echo "Adding missing start_app function..."
  cat /tmp/fix_app.py >> ui/app.py
  echo "✅ Function added successfully."
  
  # Restart the UI container
  echo "Restarting UI container..."
  docker compose restart ui
  
  # Check logs
  echo "Checking UI container logs..."
  sleep 3
  docker compose logs --tail=20 ui
else
  echo "Cannot determine how to fix app.py automatically."
  echo "Please edit ui/app.py manually to add the missing start_app function."
fi

# Clean up
rm -f /tmp/fix_app.py

echo
echo "=== Fix Complete ==="
echo "Please refresh your browser to see if the issue is resolved."
EOL

  chmod +x remote_fix_app.sh
  
  echo
  echo "=== Remote Fix Script Created ==="
  echo "To run the fix on the remote server:"
  echo "1. Copy the fix script:"
  echo "   scp remote_fix_app.sh root@103.163.186.204:~/soccer/"
  echo "2. Run it on the remote server:"
  echo "   ssh root@103.163.186.204 'cd ~/soccer && chmod +x remote_fix_app.sh && ./remote_fix_app.sh'"
fi 