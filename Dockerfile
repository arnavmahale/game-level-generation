# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim
WORKDIR /app

# Install torch CPU-only first (smaller image)
RUN pip install --no-cache-dir torch==2.2.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy application code
COPY app.py ./
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Copy built React frontend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "app:app"]
