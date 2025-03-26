# Database Migrations

This document outlines the database migration system used in the Soccer Prediction System.

## Overview

The Soccer Prediction System uses a migration-based approach to manage database schema changes. This allows for:

1. **Version Control**: All database schema changes are tracked and versioned
2. **Reproducibility**: Database schemas can be recreated consistently across environments
3. **Rollback Capability**: Migrations can be reversed if needed
4. **Documentation**: Each migration serves as documentation for schema changes

## Migration Architecture

The migration system is based on:

- A `schema_migrations` table that records which migrations have been applied
- A directory of migration scripts using a version-based naming convention
- A set of utility functions for managing migrations

## Getting Started

### Prerequisites

Before using the migration system, ensure:

1. The database connection is properly configured in your environment
2. You have the necessary permissions to modify the database schema

### Initializing the Migration System

To initialize the migration system, run:

```bash
python scripts/db_migrate.py init
```

This command creates the necessary migrations directory and the `schema_migrations` table if they don't already exist.

## Using the Migration CLI

The Soccer Prediction System provides a command-line interface for managing migrations.

### Viewing Migration Status

To see the status of all migrations:

```bash
python scripts/db_migrate.py status
```

This displays a table showing all available migrations, their status (Applied/Pending), and when they were applied.

### Creating a New Migration

To create a new migration:

```bash
python scripts/db_migrate.py create "Description of the migration"
```

This creates a new migration file with the next available version number. Edit this file to define the schema changes.

### Applying Migrations

To apply all pending migrations:

```bash
python scripts/db_migrate.py migrate
```

To apply migrations up to a specific version:

```bash
python scripts/db_migrate.py migrate 0005
```

### Rolling Back Migrations

To roll back the most recent migration:

```bash
python scripts/db_migrate.py rollback
```

To roll back multiple migrations:

```bash
python scripts/db_migrate.py rollback 3  # Rolls back 3 most recent migrations
```

### Applying a Specific Migration

To apply a specific migration:

```bash
python scripts/db_migrate.py apply 0003
```

### Reverting a Specific Migration

To revert a specific migration:

```bash
python scripts/db_migrate.py revert 0003
```

## Migration File Structure

Each migration file follows this naming convention:

`V{version}__{name}.py`

For example: `V0001__create_users_table.py`

### Structure of a Migration File

Each migration file must define two functions:

1. `upgrade(engine)`: Contains the SQL statements to apply the migration
2. `downgrade(engine)`: Contains the SQL statements to revert the migration

Example:

```python
"""
Migration: Create users table
Created: 2023-10-10 12:00:00
"""

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def upgrade(engine):
    connection = engine.connect()
    metadata = MetaData()
    
    # Create users table
    users = Table(
        'users',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('username', String(50), nullable=False, unique=True),
        # ... more columns ...
    )
    
    metadata.create_all(connection)
    connection.close()


def downgrade(engine):
    connection = engine.connect()
    connection.execute(text("DROP TABLE IF EXISTS users"))
    connection.close()
```

## Best Practices

1. **One Change Per Migration**: Each migration should handle one logical change to the schema
2. **Always Include Downgrade**: Always implement the `downgrade` function, even if you don't plan to roll back
3. **Test Both Ways**: Test both applying and reverting your migrations
4. **Keep Migrations Small**: Small, focused migrations are easier to understand and less likely to cause issues
5. **Use Sequential Versions**: Don't skip version numbers
6. **Include Comments**: Document what you're doing in the migration file

## Migration Internals

The migration system works by:

1. Scanning the migrations directory for migration files
2. Comparing them against the list of already-applied migrations in the `schema_migrations` table
3. Applying any pending migrations in order of version number

Each applied migration is recorded in the `schema_migrations` table with:
- Version number
- Name
- Timestamp of when it was applied
- Checksum of the migration file (to detect if a file has been tampered with)
- Execution time

## Handling Migration Errors

If a migration fails:

1. The system will roll back any changes made by the current migration
2. The migration will not be recorded in the `schema_migrations` table
3. An error message will be displayed with details about the failure

You should fix the issue and then try to apply the migration again.

## Integrating with Deployment

When deploying the application to a new environment, always run migrations as part of your deployment process:

```bash
python scripts/db_migrate.py migrate
```

If using continuous integration/deployment, include this command in your CI/CD pipeline. 