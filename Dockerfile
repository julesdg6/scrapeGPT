FROM python:3.11-slim

WORKDIR /app

# Ensure pip build tools are up to date
RUN pip install --no-cache-dir --upgrade pip wheel

# Install CPU-only PyTorch first so that sentence-transformers does not pull
# in the much larger CUDA-enabled wheel (~1.5 GB saved).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies.
# Build tools (gcc, g++) are installed for packages that compile C extensions,
# then purged in the same layer so they don't bloat the final image.
# libsndfile1 must be kept as a runtime dependency for soundfile/librosa.
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc g++ \
    && rm -rf /var/lib/apt/lists/*

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
    CONFIG_PATH=/data/config.json \
    WHISPERLIVE_HOST=

CMD ["python", "scrapeGPT_gradio_app.py"]
