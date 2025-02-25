FROM python:3.12-slim

WORKDIR /app

# Install UV
RUN pip install uv

# Copy requirements
COPY requirements.txt .

# Install dependencies using UV
RUN uv pip install -r requirements.txt

# Copy application code
COPY . .

CMD ["python", "-m", "llm_rag"] 