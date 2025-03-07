# Stage 1: Build Stage
FROM python:3.12-slim AS builder

# Set environment variables for building
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /build

# Install required system dependencies in one layer and clean up afterward
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    pkg-config \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install UV (used for dependency management)
RUN pip install --no-cache-dir --upgrade pip \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy only the files needed for dependency installation
COPY pyproject.toml ./

# Generate requirements file and install dependencies in two parts:
# 1. Core dependencies (excluding heavy ML ones)
# 2. ML dependencies (if any, installed separately)
RUN python -c "import tomllib; f = open('pyproject.toml', 'rb'); data = tomllib.load(f); deps = data.get('project', {}).get('dependencies', []); print('\n'.join(deps))" > requirements.txt \
    && grep -v -E "llama-cpp-python|torch|transformers|sentence-transformers|accelerate|safetensors|bitsandbytes|optimum" requirements.txt > requirements_core.txt \
    && /root/.cargo/bin/uv pip install --system --no-cache-dir -r requirements_core.txt \
    && grep -E "torch|transformers|sentence-transformers|accelerate|safetensors|bitsandbytes|optimum" requirements.txt > requirements_ml.txt || true \
    && if [ -s requirements_ml.txt ]; then /root/.cargo/bin/uv pip install --system --no-cache-dir -r requirements_ml.txt; fi \
    && rm -rf requirements.txt requirements_core.txt requirements_ml.txt /root/.cache/* /tmp/*

# Copy only the application source code
COPY src/ /build/src/

# Stage 2: Runtime Stage
FROM python:3.12-slim

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies and clean up afterward
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and application code from builder stage
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src /app/src

# Create symbolic link for the package to be importable
RUN ln -s /app/src/llm_rag /usr/local/lib/python3.12/site-packages/llm_rag

# Expose the API port
EXPOSE 8000

# Create an entrypoint script for both API and CLI modes
RUN echo '#!/bin/bash\n\
    if [ "$1" = "api" ]; then\n\
    exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000\n\
    else\n\
    exec python -m llm_rag "$@"\n\
    fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint and default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
