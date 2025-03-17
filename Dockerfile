# === Stage: UV ===
# Pull the official UV image to get the pre-built binaries.
FROM ghcr.io/astral-sh/uv:0.6.6 AS uv

# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

# Set environment variables for performance and timeouts.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=600

WORKDIR /app

# Install build and system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy UV binaries from the UV stage into /usr/local/bin so that "uv" is in PATH.
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

# Verify UV installation and print version
RUN echo "UV location: $(which uv)" && echo "UV version: $(uv --version)"

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Install dependencies using UV
RUN uv pip install --system -e .

# Now copy the rest of your application code
COPY . .

# Create a non‑root user and adjust file ownership.
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

# Create an entrypoint script that uses UV (or uvicorn) to run your tasks.
RUN echo '#!/bin/bash\nif [ "$1" = "api" ]; then\n  exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000\nelse\n  exec "$@"\nfi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# === Stage 2: Final Runtime Image ===
FROM python:3.12-slim AS final

WORKDIR /app

# Install only the minimal runtime dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy UV from the UV stage
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

# Copy installed Python packages and application from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Install the package in development mode in the final stage
RUN uv pip install --system -e /app

# Ensure proper ownership and switch to a non‑root user.
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
