FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS="0"
ENV TF_CPP_MIN_LOG_LEVEL="3"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Flask dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI dependencies
COPY backend/requirements.txt ./backend_reqs.txt
RUN pip install --no-cache-dir -r backend_reqs.txt

# Copy application files
COPY . .

# Setup configurations
COPY deploy/nginx.conf /etc/nginx/sites-available/default
COPY deploy/nginx.conf /etc/nginx/sites-enabled/default
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deploy/entrypoint.sh /app/entrypoint.sh

# Ensure proper execution permissions, linux line-endings, and upload directory
RUN chmod +x /app/entrypoint.sh && sed -i 's/\r$//' /app/entrypoint.sh && mkdir -p /app/static/uploads/ai_images

EXPOSE 80 10000 8080
ENTRYPOINT ["/app/entrypoint.sh"]
