#!/bin/bash

# Script to update Kubernetes manifests with actual values
# Usage: ./update-k8s-scripts.sh [aws|gcp|azure]

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 [aws|gcp|azure]"
  exit 1
fi

CLOUD_PROVIDER=$1
MANIFEST_DIR="deployment/$CLOUD_PROVIDER/k8s"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists envsubst; then
  echo "Error: 'envsubst' command not found. Please install it."
  echo "On Ubuntu/Debian: sudo apt-get install gettext-base"
  echo "On MacOS: brew install gettext && brew link --force gettext"
  exit 1
fi

if ! command_exists kubectl; then
  echo "Warning: 'kubectl' command not found. You will need it to apply the manifests."
fi

# Create a directory for processed manifests
PROCESSED_DIR="$MANIFEST_DIR/processed"
mkdir -p "$PROCESSED_DIR"

# Load environment variables
if [ -f ".env.$CLOUD_PROVIDER" ]; then
  echo "Loading environment variables from .env.$CLOUD_PROVIDER"
  source ".env.$CLOUD_PROVIDER"
else
  echo "Warning: .env.$CLOUD_PROVIDER file not found. Using existing environment variables."
fi

# Process each YAML file
echo "Processing Kubernetes manifests for $CLOUD_PROVIDER..."
for file in "$MANIFEST_DIR"/*.yaml; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Processing $filename..."
    
    # Skip secrets file if it contains placeholder text
    if [[ "$filename" == "secrets.yaml" ]] && grep -q "# This is a template file" "$file"; then
      echo "Skipping $filename - contains template comments. Please update this file manually."
      continue
    fi
    
    # Process the file with environment variables
    envsubst < "$file" > "$PROCESSED_DIR/$filename"
    echo "Created $PROCESSED_DIR/$filename"
  fi
done

echo ""
echo "All manifests processed. You can find them in $PROCESSED_DIR"
echo ""
echo "To apply these manifests to your Kubernetes cluster, run:"
echo "kubectl apply -f $PROCESSED_DIR"
echo ""
echo "IMPORTANT: Before applying, make sure to:"
echo "1. Update secrets.yaml with actual secret values"
echo "2. Verify the generated manifests match your requirements"
echo "3. Ensure you're connected to the correct Kubernetes cluster" 