#!/bin/bash
# Database initialization script for Soccer Prediction System

set -e

echo "=== Initializing PostgreSQL Database ==="

# Wait for PostgreSQL to start
echo "Waiting for PostgreSQL to start..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -U "$POSTGRES_USER" -c '\q'; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

echo "PostgreSQL is up - checking for soccer_prediction database"

# Check if database exists
db_exists=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -U "$POSTGRES_USER" -tAc "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'")

if [ "$db_exists" != "1" ]; then
  echo "Creating database $POSTGRES_DB..."
  PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -U "$POSTGRES_USER" -c "CREATE DATABASE $POSTGRES_DB;"
else
  echo "Database $POSTGRES_DB already exists."
fi

echo "Database setup completed successfully!"
exit 0 