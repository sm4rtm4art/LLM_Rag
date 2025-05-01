# === Stage 1: UV Base ===
FROM ghcr.io/astral-sh/uv:0.7.2 AS uv

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

# Copy UV binary first (this rarely changes)
COPY --from=uv /uv /usr/local/bin/uv

# Install build dependencies - keep this early since it rarely changes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Setup Python venv early - this rarely changes
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy ONLY dependency specifications first for better layer caching
COPY pyproject.toml ./

# Install dependencies (with explicit cache busting if needed)
RUN echo "Installing dependencies with uv - $(date +%s)" && \
    uv pip install -e . || true && \
    pip install --no-cache-dir bitsandbytes==0.42.0

# Now copy the rest of the code (this changes frequently)
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

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    BNB_CUDA_VERSION=0

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    openjdk-17-jre \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/* \
    && JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) \
    && echo "export JAVA_HOME=$JAVA_HOME" >> /etc/environment

# Copy the virtual environment and app code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/entrypoint.sh /app/entrypoint.sh
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Add venv to path
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}"

# Install the package in development mode
RUN pip install -e .

# Create a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD nc -z localhost 8008 || exit 1

EXPOSE 8008

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
