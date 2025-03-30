# === Stage: UV ===
# Pull the official UV image with a fixed version.
FROM ghcr.io/astral-sh/uv:0.6.6 AS uv

# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

# Set environment variables for performance and to avoid writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=600

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

# (Optional) Verify UV installation; remove if not needed in production.
RUN echo "UV location: $(which uv)" && echo "UV version: $(uv --version)"

# Copy dependency files first for better caching.
COPY pyproject.toml ./

# Install Python dependencies using UV's pip wrapper.
RUN uv pip install --no-cache-dir --system -e .

# Copy the rest of your application code.
COPY . .

# Create a non‑root user and adjust file ownership.
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

# Create an entrypoint script that uses UV or uvicorn.
RUN echo '#!/bin/bash\n\
    if [ "$1" = "api" ]; then\n\
    exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000\n\
    else\n\
    exec "$@"\n\
    fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# === Stage 2: Final Runtime Image ===
FROM python:3.12-slim AS final

WORKDIR /app

# Install minimal runtime dependencies and clean up.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy UV binaries from the UV stage.
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

# Copy installed Python packages and application from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Reinstall the package in the final stage to ensure consistency.
RUN uv pip install --no-cache-dir --system -e /app

# Ensure proper ownership and switch to a non‑root user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Add a health check to ensure container is ready to handle requests
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
