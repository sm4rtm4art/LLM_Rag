FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install UV using the official installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Ensure UV is installed
RUN uv --version

# Copy only pyproject.toml for better caching
COPY pyproject.toml .

# Install dependencies using UV (system-wide)
RUN uv pip install --system .

# Copy the rest of the application code
COPY . .

CMD ["python", "-m", "llm_rag"]
