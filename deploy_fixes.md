# Fixing TensorFlow Random Device Error on Remote Server

This guide contains instructions to fix the "random_device could not be read" error that occurs with TensorFlow in Docker containers.

## Changes Made Locally

1. Modified `docker-compose.yml` to:
   - Mount host's `/dev/urandom` to container's `/dev/random`
   - Add `PYTHONHASHSEED=0` environment variable

2. Updated `ui/Dockerfile` to:
   - Install `rng-tools` package
   - Modify startup script to run `rngd` daemon

## Deployment Steps

### Option 1: Deploy from Local Machine

1. Commit and push changes to GitHub:
   ```bash
   git add docker-compose.yml ui/Dockerfile
   git commit -m "Fix: Add entropy source for TensorFlow random_device error"
   git push origin master
   ```

2. On the remote server:
   ```bash
   cd ~/soccer
   git pull origin master
   docker compose down
   docker compose up -d
   ```

### Option 2: Manual Changes on Remote Server

If you can't push to GitHub or pull on the remote server, make these changes manually:

1. Edit `docker-compose.yml` on the remote server:
   ```bash
   ssh root@speedy1
   cd ~/soccer
   # Back up original file
   cp docker-compose.yml docker-compose.yml.backup
   
   # Edit the file to add these lines to the ui service:
   # Under volumes section add:
   # - /dev/urandom:/dev/random:ro
   # 
   # Under environment section add:
   # - PYTHONHASHSEED=0
   ```

2. Edit `ui/Dockerfile` on the remote server:
   ```bash
   # Back up original file
   cp ui/Dockerfile ui/Dockerfile.backup
   
   # Edit to add rng-tools to the apt-get install line
   # And add the rngd command to the start script
   ```

3. Rebuild and restart the containers:
   ```bash
   docker compose down
   docker compose up -d --build ui
   ```

### Monitoring

After deploying the changes, monitor the container logs to ensure everything is working:

```bash
docker compose logs -f ui
```

If the error persists, check the system logs for more information:

```bash
dmesg | grep -i random
```

## Alternative Solutions

If the above solutions don't work, try these alternatives:

1. Install `haveged` for more entropy:
   ```bash
   apt-get update && apt-get install -y haveged
   service haveged start
   ```

2. Use a different random source in the container:
   ```bash
   # Add to environment section in docker-compose.yml
   - NUMPY_SEED=42
   - TF_DETERMINISTIC_OPS=1
   - TF_CUDNN_DETERMINISTIC=1
   ```

3. Disable hardware random number generation:
   ```bash
   # Add to the ui service command in docker-compose.yml
   export PYTHONHASHSEED=0 && export TF_FORCE_GPU_ALLOW_GROWTH=true && streamlit run ...
   ``` 