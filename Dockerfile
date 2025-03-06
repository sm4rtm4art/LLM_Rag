# Stage 1: Build stage
FROM python:3.12-slim AS builder

# Set environment variables for building
ENV UV_LINK_MODE=copy
ENV CMAKE_ARGS="-DLLAMA_NATIVE=OFF -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_AVX512=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF"
ENV FORCE_CMAKE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Copy only the files needed for dependency installation
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache \
    uv pip install --system --no-cache-dir .

# Copy the source code
COPY src/ /build/src/

# Stage 2: Runtime stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and application from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src /app/src

# Create symbolic link for package to be importable
RUN ln -s /app/src/llm_rag /usr/local/lib/python3.12/site-packages/llm_rag

# Expose port for API
EXPOSE 8000

# Create entrypoint script to support both API and CLI modes
RUN echo '#!/bin/bash\n\
    if [ "$1" = "api" ]; then\n\
    exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000\n\
    else\n\
    exec python -m llm_rag "$@"\n\
    fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default to CLI mode if no arguments provided
CMD []
