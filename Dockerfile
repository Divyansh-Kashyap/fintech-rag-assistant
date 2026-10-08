# ==============================================================================
# FinTech RAG Assistant - Production Multi-Service Container
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS dependencies for network operations & security
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data assets
COPY app /app/app
COPY scripts /app/scripts
COPY frontend /app/frontend
COPY evaluation /app/evaluation
COPY tests /app/tests
COPY data /app/data
COPY .env.example /app/.env.example

# Set default .env if not mounted
RUN if [ ! -f /app/.env ]; then cp /app/.env.example /app/.env; fi

# Pre-run ingestion to warm vector cache within container layer
RUN python scripts/ingest_dataset.py

# Expose backend (8000) and frontend (8501) ports
EXPOSE 8000 8501

# Default startup command (can be overridden by docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
