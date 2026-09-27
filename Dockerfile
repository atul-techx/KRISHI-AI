FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Flask dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI dependencies
COPY backend/requirements.txt ./backend_reqs.txt
RUN pip install --no-cache-dir -r backend_reqs.txt

# Copy application files
COPY . .

# Copy built Next.js static export (served directly by Nginx)
COPY --from=frontend-build /app/frontend/out ./frontend/out

# Setup configurations
COPY deploy/nginx.conf /etc/nginx/sites-available/default
COPY deploy/nginx.conf /etc/nginx/sites-enabled/default
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Setup env variables for Next.js to reach FastAPI during runtime
ENV NEXT_PUBLIC_API_URL=/api

EXPOSE 80
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
