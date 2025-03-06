# Stage 1: Build stage
FROM python:3.12-slim AS builder

# Set environment variables for building
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade it
RUN pip install --upgrade pip

# Copy only the files needed for dependency installation
COPY pyproject.toml ./

# Create a requirements.txt file from pyproject.toml
RUN pip install tomli && \
    python -c "import tomli; import json; f = open('pyproject.toml', 'rb'); data = tomli.load(f); deps = data.get('project', {}).get('dependencies', []); print('\n'.join(deps))" > requirements.txt

# Remove llama-cpp-python from requirements if it exists
RUN grep -v "llama-cpp-python" requirements.txt > requirements_no_llama.txt || true

# Install Python dependencies
RUN pip install -r requirements_no_llama.txt

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
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
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
