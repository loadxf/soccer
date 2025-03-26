#!/bin/bash
# scheduled_backup.sh - Automated database backup script for Soccer Prediction System
#
# This script is designed to be run as a scheduled task (cron job or systemd timer)
# It creates a database backup and can optionally upload it to a remote storage location

# Exit on any error
set -e

# Change to the project root directory
cd "$(dirname "$0")/.."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Configuration (can be overridden by environment variables)
KEEP_BACKUPS=${KEEP_BACKUPS:-10}
UPLOAD_BACKUP=${UPLOAD_BACKUP:-false}
REMOTE_STORAGE=${REMOTE_STORAGE:-""}  # Options: "s3", "gcs", "azure", empty for no upload
REMOTE_PATH=${REMOTE_PATH:-""}
NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL:-""}

# Timestamp for logs
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Log message to console and file
log() {
    local message="[$(timestamp)] $1"
    echo "$message"
    echo "$message" >> logs/backup.log
}

# Ensure logs directory exists
mkdir -p logs

log "Starting scheduled database backup..."

# Create a new backup and clean old ones
log "Creating database backup..."
python scripts/db_backup.py backup --clean --keep $KEEP_BACKUPS

# Get the most recent backup file
LATEST_BACKUP=$(python -c "
import json
import glob
import os
import sys

try:
    metadata_files = sorted(glob.glob('backups/*.json'), reverse=True)
    if not metadata_files:
        sys.exit(1)
        
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    print(os.path.join('backups', metadata['backup_file']))
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ] || [ -z "$LATEST_BACKUP" ]; then
    log "ERROR: Failed to determine latest backup file"
    exit 1
fi

log "Latest backup: $LATEST_BACKUP"

# Upload to remote storage if configured
if [ "$UPLOAD_BACKUP" = "true" ] && [ ! -z "$REMOTE_STORAGE" ]; then
    log "Uploading backup to remote storage ($REMOTE_STORAGE)..."
    
    BACKUP_FILENAME=$(basename "$LATEST_BACKUP")
    
    case "$REMOTE_STORAGE" in
        "s3")
            # AWS S3
            if command -v aws &> /dev/null; then
                aws s3 cp "$LATEST_BACKUP" "$REMOTE_PATH/$BACKUP_FILENAME"
                log "Uploaded to S3: $REMOTE_PATH/$BACKUP_FILENAME"
            else
                log "ERROR: aws CLI not found. Install it with 'pip install awscli'"
            fi
            ;;
            
        "gcs")
            # Google Cloud Storage
            if command -v gsutil &> /dev/null; then
                gsutil cp "$LATEST_BACKUP" "$REMOTE_PATH/$BACKUP_FILENAME"
                log "Uploaded to GCS: $REMOTE_PATH/$BACKUP_FILENAME"
            else
                log "ERROR: gsutil not found. Install Google Cloud SDK"
            fi
            ;;
            
        "azure")
            # Azure Blob Storage
            if command -v az &> /dev/null; then
                # Parse container name and path from REMOTE_PATH (format: "container/path")
                CONTAINER=$(echo "$REMOTE_PATH" | cut -d'/' -f1)
                REMOTE_DIR=$(echo "$REMOTE_PATH" | cut -d'/' -f2-)
                
                if [ ! -z "$REMOTE_DIR" ]; then
                    DEST_PATH="$REMOTE_DIR/$BACKUP_FILENAME"
                else
                    DEST_PATH="$BACKUP_FILENAME"
                fi
                
                az storage blob upload --container-name "$CONTAINER" --file "$LATEST_BACKUP" --name "$DEST_PATH"
                log "Uploaded to Azure: $CONTAINER/$DEST_PATH"
            else
                log "ERROR: Azure CLI not found. Install it with 'pip install azure-cli'"
            fi
            ;;
            
        *)
            log "ERROR: Unsupported remote storage type: $REMOTE_STORAGE"
            ;;
    esac
fi

# Send email notification if configured
if [ ! -z "$NOTIFICATION_EMAIL" ]; then
    if command -v mail &> /dev/null; then
        log "Sending notification email to $NOTIFICATION_EMAIL"
        
        HOSTNAME=$(hostname)
        BACKUP_SIZE=$(du -h "$LATEST_BACKUP" | cut -f1)
        
        {
            echo "Subject: Soccer Prediction System - Database Backup Completed"
            echo "From: Soccer Prediction System <noreply@soccerprediction.system>"
            echo "To: $NOTIFICATION_EMAIL"
            echo "MIME-Version: 1.0"
            echo "Content-Type: text/plain; charset=utf-8"
            echo ""
            echo "Soccer Prediction System - Database Backup Report"
            echo "========================================"
            echo ""
            echo "Host: $HOSTNAME"
            echo "Date: $(date)"
            echo "Backup file: $BACKUP_FILENAME"
            echo "Size: $BACKUP_SIZE"
            echo ""
            if [ "$UPLOAD_BACKUP" = "true" ] && [ ! -z "$REMOTE_STORAGE" ]; then
                echo "Remote storage: $REMOTE_STORAGE"
                echo "Remote path: $REMOTE_PATH/$BACKUP_FILENAME"
            fi
            echo ""
            echo "This is an automated message. Please do not reply."
        } | sendmail -t
    else
        log "WARNING: 'mail' command not found. Cannot send notification email."
    fi
fi

log "Scheduled backup completed successfully"
exit 0 