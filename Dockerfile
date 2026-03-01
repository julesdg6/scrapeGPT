FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY scrapeGPT_gradio_app.py .
COPY scrapeGPT.py .
COPY db.json .

# Persistent data directory
RUN mkdir -p /data /app/tmp/local_qdrant

EXPOSE 7860

ENV OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=qwen:0.5b \
    GRADIO_PORT=7860 \
    DB_PATH=/data/db.json \
    CONFIG_PATH=/data/config.json

CMD ["python", "scrapeGPT_gradio_app.py"]
