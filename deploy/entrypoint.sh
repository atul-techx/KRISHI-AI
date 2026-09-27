#!/bin/sh
set -e

# Default to 80 if PORT is not set (Render provides $PORT, e.g. 10000)
export PORT="${PORT:-80}"

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
