# ==============================================================================
# Dockerfile for RAG System Deployment
# ==============================================================================

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY src/ ./src/
COPY app/ ./app/
COPY outputs/ ./outputs/
COPY .env.example ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Mount outputs/ as a volume in production for updatable FAISS index
VOLUME /app/outputs
