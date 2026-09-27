#!/bin/sh
set -e

# Default to 80 if PORT is not set (Railway and Render provide $PORT)
export PORT="${PORT:-80}"

# Optimize Python and TensorFlow memory usage for cloud containers
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="-1"
export TF_ENABLE_ONEDNN_OPTS="0"
export TF_CPP_MIN_LOG_LEVEL="3"

echo "=================================================="
echo " Starting KRISHI-AI Container on PORT: ${PORT}"
echo "=================================================="

# Ensure upload directory exists
mkdir -p /app/static/uploads/ai_images

# Dynamically bind Nginx to the cloud provider's $PORT
sed -i "s/listen [0-9]\+;/listen ${PORT};/g" /etc/nginx/sites-available/default
sed -i "s/listen [0-9]\+;/listen ${PORT};/g" /etc/nginx/sites-enabled/default

# Test nginx config syntax
nginx -t

# Hand off execution to supervisord (runs Nginx, FastAPI, and Flask)
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf
