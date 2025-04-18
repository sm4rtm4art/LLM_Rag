# === Stage 1: UV Base ===
FROM ghcr.io/astral-sh/uv:0.6.6 AS uv

# === Stage 2: Builder ===
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=600 \
    BNB_CUDA_VERSION=0 \
    UV_HTTP_TIMEOUT=120

WORKDIR /app

# Install build and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Copy UV binary
COPY --from=uv /uv /usr/local/bin/uv

# Copy dependency specifications
COPY pyproject.toml ./

# Create a virtual environment
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Use UV sync for fast parallel installation and handle bitsandbytes separately
RUN echo "Installing dependencies with UV sync (parallel installation)..." && \
    # Use uv sync to install dependencies quickly (parallel)
    uv sync --no-install-project || true && \
    # Then handle bitsandbytes separately
    pip install --no-cache-dir bitsandbytes==0.42.0 && \
    # Finally install the project itself
    pip install -e .

# Copy the rest of your code
COPY . .

# Create entrypoint script
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'if [ "$1" = "api" ]; then' >> /app/entrypoint.sh && \
    echo '  exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000' >> /app/entrypoint.sh && \
    echo 'else' >> /app/entrypoint.sh && \
    echo '  exec "$@"' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# === Stage 3: Runtime ===
FROM python:3.12-slim AS final

WORKDIR /app

# Set environment variables for runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    BNB_CUDA_VERSION=0

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    openjdk-17-jre \
    && rm -rf /var/lib/apt/lists/* \
    && JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) \
    && echo "export JAVA_HOME=$JAVA_HOME" >> /etc/environment

# Copy the entire virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to path
ENV PATH="/app/.venv/bin:${PATH}"

# Copy application code and entry point
COPY --from=builder /app/src /app/src
COPY --from=builder /app/entrypoint.sh /app/entrypoint.sh

# Create a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
