# Database Backup and Restore Procedures

This document outlines the database backup and restore procedures for the Soccer Prediction System.

## Overview

The Soccer Prediction System includes a comprehensive utility for managing database backups. This utility provides functionality for:

1. **Creating Backups**: Automated backups of your database
2. **Listing Backups**: Viewing available backups with metadata
3. **Restoring Backups**: Recovering database state from a previous backup
4. **Cleaning Old Backups**: Managing storage by removing older backups

The backup system supports multiple database types, including PostgreSQL, MySQL, SQLite, and MongoDB.

## Prerequisites

Before using the backup and restore functionality, ensure:

1. The database connection is properly configured in your environment
2. You have the required client tools installed for your database type:
   - PostgreSQL: `pg_dump` and `pg_restore`
   - MySQL: `mysqldump` and `mysql` client
   - MongoDB: `mongodump` and `mongorestore`
3. You have sufficient permissions to read from and write to the database

## Using the Backup and Restore Utility

### Creating a Backup

To create a new backup of your database:

```bash
python scripts/db_backup.py backup
```

This will create a backup of the database configured in your environment settings.

#### Options

- `--type`: Specify the database type (postgres, mysql, sqlite, mongodb)
- `--clean`: Clean old backups after creating a new one
- `--keep`: Number of backups to keep when cleaning (default: 10)

Example:

```bash
# Create a backup and keep only the 5 most recent backups
python scripts/db_backup.py backup --clean --keep 5
```

### Listing Available Backups

To see a list of all available backups:

```bash
python scripts/db_backup.py list
```

This will display a table showing all backups with:
- ID: A sequential identifier for use with the restore command
- Date: When the backup was created
- Type: Database type (postgres, mysql, sqlite, mongodb)
- Database: Name of the database
- Size: Size of the backup file
- File: Name of the backup file

### Restoring from a Backup

To restore your database from a backup:

```bash
python scripts/db_backup.py restore --id <backup_id>
```

Where `<backup_id>` is the ID shown in the list command output.

#### Options

- `--id`: Backup ID to restore (from the list output)
- `--file`: Path to a backup file to restore (alternative to --id)
- `--type`: Database type (required if --file is specified)

Example:

```bash
# Restore from backup ID 3
python scripts/db_backup.py restore --id 3

# Restore from a specific backup file
python scripts/db_backup.py restore --file /path/to/backup.sql --type postgres
```

> ⚠️ **Warning**: Restoring a database will OVERWRITE the current database. All existing data will be lost. The script will ask for confirmation before proceeding.

### Cleaning Old Backups

To remove old backups while keeping the most recent ones:

```bash
python scripts/db_backup.py clean
```

#### Options

- `--keep`: Number of backups to keep (default: 10)

Example:

```bash
# Keep only the 3 most recent backups
python scripts/db_backup.py clean --keep 3
```

## Implementation Details

### Backup Storage

All backups are stored in the `backups/` directory at the project root. Each backup consists of:

1. **Backup File**: The actual backup data in a format appropriate for the database type
2. **Metadata File**: A JSON file containing information about the backup

### Backup Naming Convention

Backup files follow this naming convention:

`<db_type>_<timestamp>.<extension>`

Example: `postgres_20231010_120000.sql`

### Database-Specific Notes

#### PostgreSQL

- Backups use the PostgreSQL custom format (`-F c` option in pg_dump)
- Authentication uses the PGPASSWORD environment variable

#### MySQL

- Backups include routines, triggers, and events

#### SQLite

- Backups are direct copies of the database file

#### MongoDB

- Backups are created with mongodump and stored as ZIP archives

## Automating Backups

### Scheduled Backups

For production environments, it's recommended to schedule regular backups using cron (Linux/macOS) or Task Scheduler (Windows).

Example cron entry for daily backups at 2 AM:

```
0 2 * * * cd /path/to/soccer_prediction && python scripts/db_backup.py backup --clean
```

### Backup During Deployment

For critical operations like deployments, consider adding a pre-deployment backup step. For example, in your deployment script:

```bash
# Create a backup before deploying
python scripts/db_backup.py backup

# Apply migrations
python scripts/db_migrate.py migrate

# Start the application
# ...
```

## Best Practices

1. **Regular Backups**: Set up automated, scheduled backups
2. **Verification**: Periodically verify that backups can be successfully restored
3. **Rotation**: Set up backup rotation to manage storage (using the `--clean` and `--keep` options)
4. **Off-site Storage**: Copy critical backups to an off-site location
5. **Pre-Update Backups**: Always create a backup before major database schema changes

## Troubleshooting

### Common Issues

#### Backup Creation Fails

- Check that the database is accessible with the provided credentials
- Verify that you have the necessary database client tools installed
- Ensure the user has sufficient permissions to read the database

#### Restore Fails

- Check that the backup file exists and is not corrupted
- Ensure you have the necessary permissions to write to the database
- Verify that the database can be connected to with the provided credentials

## Integration with Cloud Environments

When deploying to cloud environments, consider these modifications:

### GCP (Google Cloud Platform)

For PostgreSQL on Cloud SQL:
- Use the Cloud SQL Proxy for secure connections
- Consider using GCP's built-in backup mechanisms alongside this utility
- Store backups in Google Cloud Storage

### Azure

For Azure Database for PostgreSQL:
- Use Azure Backup or the backup utility with appropriate connection settings
- Store backups in Azure Blob Storage

### AWS

For Amazon RDS:
- Use the RDS snapshot mechanism or this backup utility
- Store backups in Amazon S3

## Security Considerations

- Backup files may contain sensitive data
- Ensure backups are stored securely with appropriate access controls
- Consider encrypting backup files if they contain particularly sensitive information 