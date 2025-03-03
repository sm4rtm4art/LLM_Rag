# Use the same base image
FROM python:3.12-slim

# Add this line near the top
ENV UV_LINK_MODE=copy
# Add environment variables for llama-cpp-python build
ENV CMAKE_ARGS="-DLLAMA_NATIVE=OFF -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_AVX512=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF"
ENV FORCE_CMAKE=1

WORKDIR /app

# System dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install UV
RUN --mount=type=cache,target=/root/.cache \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Copy only necessary dependency files
COPY pyproject.toml .

# Install all dependencies at once with llama-cpp-python
RUN --mount=type=cache,target=/root/.cache \
    uv pip install --system --no-cache-dir \
    chromadb>=0.6.3 \
    pydantic>=2.0 \
    duckdb>=0.8.0 \
    fastapi>=0.115.8 \
    langchain>=0.1.0 \
    langchain-community>=0.0.10 \
    langchain-openai>=0.0.2 \
    numpy>=1.26.4,\<2 \
    pandas>=2.2.0 \
    pytest>=8.3.4 \
    PyPDF2>=3.0.0 \
    scikit-learn>=1.6.1 \
    sentence-transformers>=2.2.2 \
    tiktoken>=0.9.0 \
    torch>=2.1.0 \
    transformers>=4.49.0 \
    uvicorn>=0.34.0 \
    llama-cpp-python==0.2.56

# Copy project files
COPY src/ /app/src/
COPY tests/ /app/tests/

# Create symbolic link for package to be importable as llm_rag
RUN ln -s /app/src/llm_rag /usr/local/lib/python3.12/site-packages/llm_rag

# Install the package in development mode
RUN --mount=type=cache,target=/root/.cache \
    uv pip install --system -e .

CMD ["python", "-m", "llm_rag"]
