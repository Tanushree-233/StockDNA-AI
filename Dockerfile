# ==============================================================================
# Multi-Stage Production Dockerfile for StockDNA-AI
# Single-container deployment: React 18 Frontend + FastAPI Backend + XGBoost Model
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build the React + Vite Frontend SPA
# ------------------------------------------------------------------------------
FROM node:18-alpine AS frontend-builder

WORKDIR /build/frontend

# Copy dependency manifests
COPY frontend/package*.json ./
RUN npm ci --silent || npm install --silent

# Copy frontend source code and build production assets
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python Runtime with FastAPI & Locked XGBoost Model
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS runner

# System dependencies (compiler headers for numpy/scipy/xgboost if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PORT=8000

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code and scientific modules
COPY backend/ ./backend/
COPY config/ ./config/
COPY data_tool/ ./data_tool/
COPY models/production/ ./models/production/
COPY results/ ./results/
COPY docs/ ./docs/

# Copy compiled frontend static assets from Stage 1
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Expose runtime port
EXPOSE 8000

# Production startup command using hosting provider's dynamic PORT
CMD ["sh", "-c", "python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
