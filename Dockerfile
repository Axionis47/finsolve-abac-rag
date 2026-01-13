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

# Copy pre-indexed ChromaDB data (if exists)
COPY .chroma_db/ .chroma_db/

# Create data directories
RUN mkdir -p logs/audit

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8080
ENV CHROMA_DB_DIR=.chroma_db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose the port Cloud Run expects
EXPOSE 8080

# Run the application on port 8080 (Cloud Run expects this)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

