# === Stage: UV ===
# Pull the official UV image with a fixed version.
FROM ghcr.io/astral-sh/uv:0.6.6 AS uv

# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

# Set environment variables for performance and to avoid writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build and system dependencies in one layer and clean up afterward.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy UV binaries from the UV stage into /usr/local/bin so that "uv" is in PATH.
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

# (Optional) Verify UV installation.
RUN echo "UV location: $(which uv)" && uv --version

# Copy dependency file first for better caching.
COPY pyproject.toml ./

# Install Python dependencies using UV's pip wrapper.
# Install to a dedicated target directory
RUN uv pip install --no-cache-dir -e . --target=/app/site-packages

# Copy the rest of your application code.
COPY . .

# Create a non‑root user and fix file ownership.
RUN useradd --create-home appuser && chown -R appuser:appuser /app

# Create an entrypoint script using a multi‑line echo chain.
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'if [ "$1" = "api" ]; then' >> /app/entrypoint.sh && \
    echo '  exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000' >> /app/entrypoint.sh && \
    echo 'else' >> /app/entrypoint.sh && \
    echo '  exec "$@"' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# === Stage 2: Final Runtime Image ===
FROM python:3.12-slim AS final

WORKDIR /app

# Set environment variables for runtime.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/site-packages \
    PORT=8000

# Install minimal runtime dependencies and clean up.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy UV binaries from the UV stage.
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

# Copy only the necessary files from the builder stage.
COPY --from=builder /app/site-packages /app/site-packages
COPY --from=builder /app/entrypoint.sh /app/entrypoint.sh
COPY --from=builder /app/pyproject.toml /app/
COPY --from=builder /app/src /app/src

# Create a non‑root user for runtime and set proper ownership.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Add a health check to ensure the container is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
