# Use the same base image
FROM python:3.12-slim

# Add this line near the top
ENV UV_LINK_MODE=copy

WORKDIR /app

# Change 1: Combine system dependency installation with cache mount
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Change 2: Add cache mount for UV installation
RUN --mount=type=cache,target=/root/.cache \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Change 3: Copy only dependency files first
COPY pyproject.toml .

# Change 4: Add cache mount for UV dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

# Change 5: Copy only necessary directories instead of everything
COPY src/ src/
COPY tests/ tests/

CMD ["python", "-m", "llm_rag"]