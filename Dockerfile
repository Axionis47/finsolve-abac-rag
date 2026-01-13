# =============================================================================
# Dockerfile for RAG Chatbot Application
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Install optional dependencies for local embeddings
RUN pip install --no-cache-dir sentence-transformers

# Copy application code
COPY app/ app/
COPY resources/ resources/
COPY templates/ templates/
COPY static/ static/
COPY docs/ docs/

# Create data directories
RUN mkdir -p /data/chroma /data/cache logs/audit

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

