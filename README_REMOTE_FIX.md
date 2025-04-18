# Fixing Streamlit UI Connection Issues

This README provides instructions for deploying the fix to resolve the "site cannot be reached" error with the Streamlit UI container.

## Root Cause

The issue is due to a `random_device could not be read` error in the TensorFlow dependency check, which is caused by:

1. Insufficient entropy in the Docker container
2. The container not having proper access to `/dev/random`
3. The dependency verification script crashing before Streamlit can start

## Fix Scripts

We've created two scripts to solve this issue:

1. `fix_ui_entropy.sh` - Addresses the random_device error by creating a fixed Dockerfile
2. `fix_streamlit_connection.sh` - Diagnoses and fixes common Streamlit connection issues

## Deployment Instructions

### Option 1: Copy Scripts to Remote Server and Run

1. Copy the fix scripts to the remote server:

```bash
scp fix_ui_entropy.sh fix_streamlit_connection.sh root@103.163.186.204:~/soccer/
```

2. SSH into the remote server:

```bash
ssh root@103.163.186.204
```

3. Run the entropy fix script:

```bash
cd ~/soccer
chmod +x fix_ui_entropy.sh
./fix_ui_entropy.sh
```

4. Rebuild and restart the UI container:

```bash
docker compose up -d --build ui
```

5. Check if the container started correctly:

```bash
docker compose logs ui
```

### Option 2: Deploy Changes Manually

If you prefer to make changes manually on the remote server:

1. SSH into the remote server:

```bash
ssh root@103.163.186.204
```

2. Create a new Dockerfile for the UI:

```bash
cd ~/soccer
cp ui/Dockerfile ui/Dockerfile.backup
```

3. Edit the new Dockerfile:

```bash
vi ui/Dockerfile
```

Add these changes:
- Add `PYTHONHASHSEED=0` to the environment variables
- Add `rng-tools` to the apt-get install packages
- Modify the start script to skip the dependency verification

4. Update docker-compose.yml:

```bash
vi docker-compose.yml
```

Make these changes:
- Change the UI service command to directly use the start.sh script
- Add `/dev/urandom:/dev/random:ro` to the volumes section

5. Restart the UI container:

```bash
docker compose down
docker compose up -d
```

## Troubleshooting

If the fix doesn't work immediately:

1. Run the diagnostics script:

```bash
cd ~/soccer
chmod +x fix_streamlit_connection.sh
./fix_streamlit_connection.sh
```

2. Check the UI container logs:

```bash
docker compose logs ui
```

3. Verify if port 8501 is accessible:

```bash
netstat -tuln | grep 8501
```

4. Check firewall settings:

```bash
ufw status
```

5. Try bypassing Docker by running Streamlit directly on the server:

```bash
pip install streamlit
cd ~/soccer
streamlit run ui/app.py --server.address=0.0.0.0
```

## Further Assistance

If you continue to experience issues, consider:

1. Checking the server's system logs:
```bash
dmesg | grep -i random
```

2. Installing additional entropy sources directly on the host:
```bash
apt-get update && apt-get install -y haveged
service haveged start
```

3. Trying alternative TensorFlow configuration:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_DETERMINISTIC_OPS=1
``` 