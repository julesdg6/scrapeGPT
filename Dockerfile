FROM python:3.11-slim

WORKDIR /app

# Select the PyTorch variant at build time.
# Use 'cpu' (default, ~250 MB, no GPU required) or a CUDA tag such as
# 'cu121' or 'cu118' for NVIDIA GPU acceleration (~2 GB).
# Example: docker build --build-arg PYTORCH_FLAVOR=cu121 .
ARG PYTORCH_FLAVOR=cpu

# Ensure pip build tools are up to date
RUN pip install --no-cache-dir --upgrade pip wheel

# Install PyTorch for the selected flavor before requirements.txt so that
# sentence-transformers (and other packages) reuse whichever wheel is already
# present rather than pulling the default (CUDA-enabled) variant.
RUN pip install --no-cache-dir torch --index-url "https://download.pytorch.org/whl/${PYTORCH_FLAVOR}"

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

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["python", "scrapeGPT_gradio_app.py"]
