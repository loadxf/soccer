# Dependencies Update

## Summary of Changes

We have made the following changes to ensure that all dependencies, including those that were previously marked as optional, are now required for the project:

1. **Merged requirements files**: 
   - All dependencies from `optional-requirements.txt` have been merged into `requirements.txt`
   - This ensures that all packages are installed by default

2. **Updated Docker configurations**:
   - Modified `Dockerfile` and `ui/Dockerfile` to only use the unified `requirements.txt` file
   - Removed references to `optional-requirements.txt` since it's no longer needed

3. **Created verification scripts**:
   - Added `scripts/verify_all_dependencies.py` to check that all dependencies are properly installed
   - This script is run during container startup to ensure all packages are available

4. **Updated docker-compose.yml**:
   - Added the dependency verification script to the startup command for the UI container

## Why This Change?

This change ensures that TensorFlow and other previously optional ML libraries are always installed and available, preventing errors like `name 'tf' is not defined` when running models or training scripts.

## Dependencies Now Required

The following previously optional dependencies are now required:

- **ML Libraries**:
  - TensorFlow
  - PyTorch
  - LightGBM
  - CatBoost
  - Optuna
  - Hyperopt
  - MLflow
  - Prophet

- **Web Scraping Tools**:
  - BeautifulSoup4
  - Selenium
  - Scrapy

## Rebuilding Docker Images

To apply these changes, rebuild your Docker images:

```bash
docker-compose build
docker-compose up -d
```

When the containers start, the verification scripts will run to ensure all dependencies are properly installed.

### Port Conflicts

If you encounter port conflicts (like "port is already allocated"), we've changed the frontend port from 3000 to 3001 in the docker-compose.yml file. If you still encounter port conflicts, you can modify the port mappings in docker-compose.yml to use different ports.

To check which ports are currently in use on your system:
```bash
sudo netstat -tulpn | grep LISTEN
```

## Reverting Changes (If Needed)

If you need to revert to the previous setup with optional dependencies, a backup of the original `optional-requirements.txt` has been created as `optional-requirements.txt.bak`. 