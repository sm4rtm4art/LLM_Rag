# === Stage 1: UV Base ===
FROM ghcr.io/astral-sh/uv:0.8.14 AS uv

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

# Install build dependencies - Combine steps and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    openjdk-17-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup Python venv early - this rarely changes
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy ONLY dependency specifications first for better layer caching
COPY pyproject.toml ./

# Install dependencies (with explicit cache busting if needed)
RUN echo "Installing dependencies with uv - $(date +%s)" && \
    uv pip install -e . && \
    pip install --no-cache-dir bitsandbytes==0.42.0

# Now copy the rest of the code (this changes frequently)
COPY src/ ./src/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# === Stage 3: Runtime ===
FROM python:3.12-slim AS final

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    BNB_CUDA_VERSION=0

WORKDIR /app

# Install runtime dependencies - Combine steps and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    openjdk-17-jre \
    netcat-traditional \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) \
    && echo "export JAVA_HOME=$JAVA_HOME" >> /etc/environment

# Copy the virtual environment and entrypoint from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/entrypoint.sh /app/entrypoint.sh

# Add venv to path
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH:+:$PYTHONPATH}"

# Create a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD nc -z localhost 8008 || exit 1

EXPOSE 8008

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
