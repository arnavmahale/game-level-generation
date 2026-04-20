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

# Copy application code. No data/ needed at runtime — the reference tile
# distribution lives inside models/ as a precomputed .npy (see
# scripts/build_reference.py), so we don't ship the 150MB training JSONL.
COPY app.py ./
COPY scripts/ ./scripts/
COPY models/ ./models/

# Copy built React frontend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Default to port 7860 so the container runs on Hugging Face Spaces out of the
# box (HF's Docker SDK expects that port unless overridden via README metadata).
# Shell-form CMD lets us honor a runtime $PORT override for other hosts.
ENV PORT=7860
EXPOSE 7860

CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --timeout 120 --workers 1 app:app
