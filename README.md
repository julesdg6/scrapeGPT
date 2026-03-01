# ScrapeGoat
## Web App & Telegram Bot for Web Content Analysis and Question Answering

ScrapeGoat is a Gradio web app (and optional Telegram bot) that scrapes websites and answers questions about them using a local LLM via [Ollama](https://ollama.com). The idea came from a job interview assignment to "study our company website and provide new product recommendations" — this automates the entire process.

> **History:** This project was originally developed as [scrapeGPT](https://github.com/julesdg6/scrapeGPT) and has since been renamed and split into its own repository.

> **Note:** There are two entry points:
> - `ScrapeGoat_gradio_app.py` — Gradio web UI (recommended, used by Docker)
> - `ScrapeGoat.py` — Telegram bot (includes a commented CLI `main()` at the bottom)

## Features

- **Flexible Chat**: A single unified interface that automatically detects the type of input and routes to the correct handler:
  - **URL in prompt** → scrapes the page(s) via [SearXNG](https://github.com/searxng/searxng) (or direct proxy crawl as fallback) and answers using retrieved content
  - **Image upload** → analyses the image using a multimodal Ollama model (e.g. `llava`)
  - **Audio upload** → transcribes speech (via [WhisperLive](https://github.com/collabora/WhisperLive) when configured) *or*, if no clear speech is found, analyses the audio as music (BPM, estimated key, spectral centroid via librosa)
  - **Plain text** → answers directly with the configured LLM
- **Web Scraping**: Uses [SearXNG](https://github.com/searxng/searxng) as the primary backend for discovering and retrieving web content. Falls back to direct proxy crawling when SearXNG is not configured.
- **Context Retrieval**: Utilises embeddings and retrieval models to extract relevant context from scraped content.
- **Question Answering**: Generates answers to user questions based on the retrieved context.
- **Robots.txt Parsing**: Respects the website's robots.txt to avoid scraping restricted areas.
- **Database Management**: Stores scraped content in a database for future reference and quick access.
- **Proxy Support**: Uses rotating proxies to bypass geo-restrictions and anonymise requests.
- **LLM-based**: Connects to a local [Ollama](https://ollama.com) instance (including Ollama running in another container).
- **Image generation (optional)**: Generates images via an Automatic1111 (stable-diffusion-webui) instance using its `--api` HTTP API.

### Additional models required for new features

| Feature | Required model / package | How to enable |
|---|---|---|
| Image analysis | A multimodal Ollama model, e.g. `llava` | `ollama pull llava` then set `OLLAMA_VISION_MODEL=llava` |
| Image generation | Automatic1111 (stable-diffusion-webui) with `--api` enabled | Set `A1111_HOST` / `A1111_PORT` (see below) |
| Speech transcription | [WhisperLive](https://github.com/collabora/WhisperLive) container (`ghcr.io/collabora/whisperlive-gpu:latest`) | Set `WHISPERLIVE_HOST=http://whisperlive:9090` (see below) |
| Music analysis | librosa + soundfile Python packages | Already in `requirements.txt`; no extra setup required |

---

## Companion Services

ScrapeGoat integrates with several independent Docker services. Each service runs in its own container and is configured via environment variables. You do **not** need all of them — each is optional or can be replaced by an external instance you already have running.

| Service | Image | Purpose | Required? |
|---|---|---|---|
| **Ollama** | `ollama/ollama:latest` | Local LLM inference (text, vision) | Yes — or point `OLLAMA_HOST` at an existing instance |
| **SearXNG** | `searxng/searxng:latest` | Privacy-friendly meta-search engine used as the web scraping backend | Recommended — falls back to direct proxy crawling if absent |
| **WhisperLive** | `ghcr.io/collabora/whisperlive-gpu:latest` | GPU-accelerated speech transcription | Optional (requires NVIDIA GPU) |

> **Tip for Unraid / existing setups:** If you already run Ollama or SearXNG as standalone containers, simply point ScrapeGoat's `OLLAMA_HOST` / `SEARXNG_HOST` environment variables at those instances. You do not need to install them again via this compose file.

---

## Getting Started

### Option A — Docker (recommended)

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.

#### 1. Clone the repository
```bash
git clone https://github.com/julesdg6/ScrapeGoat.git
cd ScrapeGoat
```

#### 2. Configure environment variables
```bash
cp .env.example .env
# Edit .env as needed (set OLLAMA_MODEL, ports, etc.)
```

Key variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://ollama:11434` | URL of the Ollama API |
| `OLLAMA_MODEL` | `qwen:0.5b` | Ollama model name |
| `OLLAMA_VISION_MODEL` | `llava` | Ollama vision model for image analysis (pull first) |
| `GRADIO_PORT` | `7860` | Host port for the Gradio web UI |
| `TELEGRAM_BOT_TOKEN` | *(empty)* | Telegram token (bot mode only) |
| `WHISPERLIVE_HOST` | *(empty)* | URL of the WhisperLive service for speech transcription (optional) |
| `SEARXNG_HOST` | `http://searxng:8080` | URL of the SearXNG search backend (recommended; leave empty to use legacy proxy scraping) |
| `SEARXNG_PORT` | `8080` | Host port for the bundled SearXNG web UI |
| `A1111_HOST` | *(empty)* | Automatic1111 base URL (host) for image generation (enables Image Generation tab) |
| `A1111_PORT` | `7860` | Automatic1111 API/web UI port |
| `A1111_MODEL` | *(empty)* | Optional checkpoint name (leave empty to use currently loaded model) |
| `A1111_WIDTH` | `256` | Generated image width |
| `A1111_HEIGHT` | `256` | Generated image height |

#### 3. Start the stack
```bash
docker compose up -d
```
This starts **ScrapeGoat**, **Ollama**, and **SearXNG** containers.

> **Already running Ollama or SearXNG elsewhere?**  
> Set `OLLAMA_HOST` and/or `SEARXNG_HOST` in your `.env` to point at your existing instances, then remove the corresponding service from the compose file (or just ignore it — it won't affect ScrapeGoat).

#### Optional: Enable GPU speech transcription with WhisperLive

To enable speech transcription, start the WhisperLive service alongside the stack:

```bash
# In your .env:
WHISPERLIVE_HOST=http://whisperlive:9090

# Then start all services including the whisper profile:
docker compose --profile whisper up -d
```

The WhisperLive service uses `ghcr.io/collabora/whisperlive-gpu:latest` and requires an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.  If speech transcription is not needed, leave `WHISPERLIVE_HOST` empty — audio uploads will still be analysed for music features (BPM, key, etc.).

#### Optional: Enable image generation with Automatic1111 (A1111)

Start your Automatic1111 (stable-diffusion-webui) container with the `--api` flag enabled, then set:

```bash
# In your .env:
A1111_HOST=http://a1111
A1111_PORT=7860
# Optional:
A1111_MODEL=
A1111_WIDTH=256
A1111_HEIGHT=256
```

#### 4. Pull the Ollama model
```bash
docker exec -it ollama ollama pull qwen:0.5b
```

**Optional — image analysis (llava):**  
To enable image analysis, pull a multimodal vision model and set `OLLAMA_VISION_MODEL` in your `.env`:
```bash
docker exec -it ollama ollama pull llava
# In your .env:
OLLAMA_VISION_MODEL=llava
```

#### 5. Open the web UI
Navigate to **http://localhost:7860** in your browser.

---

#### Manual `docker run` (pre-built image)

The image is published to GitHub Container Registry and can be pulled directly:

```bash
# Run with an existing Ollama container on the same network
docker run -d \
  --name scrapegoat \
  -p 7860:7860 \
  -e OLLAMA_HOST=http://ollama:11434 \
  -e OLLAMA_MODEL=qwen:0.5b \
  -v scrapegoat_data:/data \
  -v scrapegoat_qdrant:/app/tmp/local_qdrant \
  --network <your-docker-network> \
  ghcr.io/julesdg6/scrapegoat:latest
```

> **Connecting to Ollama in another container:**  
> Both containers must share a Docker network. With Compose this is automatic. For manual setups, create a network first:
> ```bash
> docker network create scrapegoat-net
> docker run -d --name ollama --network scrapegoat-net -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama
> docker run -d --name scrapegoat --network scrapegoat-net -p 7860:7860 -e OLLAMA_HOST=http://ollama:11434 ghcr.io/julesdg6/scrapegoat:latest
> ```

---

### Option B — Unraid Install

ScrapeGoat includes an Unraid Docker template for easy container configuration. The Docker image is published to GitHub Container Registry and will be pulled automatically — no manual build step required.

#### Prerequisites
- Unraid with Docker enabled.
- An **Ollama** container already running on your Unraid server (install `ollama/ollama` from Community Applications or via the Docker tab).
- *(Optional but recommended)* A **SearXNG** container running on your Unraid server (install `searxng/searxng` from Community Applications or via the Docker tab).

#### Step-by-step

1. **Install Ollama** (if not already running):
   - In Unraid, go to **Apps** → search for `Ollama` → install `ollama/ollama`.
   - After it starts, pull your model via the Unraid terminal:
     ```bash
     docker exec -it ollama ollama pull qwen:0.5b
     ```

2. **Install SearXNG** (recommended — provides the web scraping backend):
   - In Unraid, go to **Apps** → search for `SearXNG` → install `searxng/searxng`.
   - No extra configuration is required for basic use; ScrapeGoat's default `SEARXNG_HOST=http://searxng:8080` will connect to it automatically once both containers are on the same Docker network.

3. **Add the ScrapeGoat container using the template**:
   - Open the Unraid terminal (Tools → Terminal) or SSH into your server and download the template:
     ```bash
     wget -O /boot/config/plugins/dockerMan/templates-user/ScrapeGoat.xml \
       https://raw.githubusercontent.com/julesdg6/ScrapeGoat/main/unraid/ScrapeGoat.xml
     ```
   - In Unraid, go to **Docker** → **Add Container**.
   - In the *Template* dropdown at the top, select **ScrapeGoat** to auto-populate the settings.
   - Review and configure the variables:

     | Variable | Recommended Value | Notes |
     |---|---|---|
     | **Ollama Host** | `http://ollama:11434` | Use container name if on same Docker network; otherwise use your server IP, e.g. `http://192.168.1.100:11434` |
     | **Ollama Model** | `qwen:0.5b` | Must be pulled into Ollama first |
     | **Ollama Vision Model** | `llava` | For image analysis; must be pulled first: `ollama pull llava` |
     | **WhisperLive Host** | `http://whisperlive:9090` | (Optional) For speech transcription; run `ghcr.io/collabora/whisperlive-gpu:latest` on the same network. Leave empty to disable. |
     | **SearXNG Host** | `http://searxng:8080` | (Recommended) URL of your SearXNG container. Leave empty to fall back to direct proxy scraping. |
     | **A1111 Host** | `http://a1111` | (Optional) Enables Image Generation tab (A1111 must run with `--api`) |
     | **A1111 Port** | `7860` | (Optional) Automatic1111 API/web UI port |
     | **A1111 Default Model** | *(empty)* | (Optional) Leave blank to use the currently loaded model |
     | **A1111 Resolution** | `256x256` | (Optional) Image generation width/height |
     | **WebUI Port** | `7860` | Host port to access Gradio |
     | **App Data** | `/mnt/user/appdata/scrapegoat` | Stores `db.json` |
     | **Qdrant Vector Store** | `/mnt/user/appdata/scrapegoat/qdrant` | Stores embedding vectors |

   - Click **Apply** to start the container. Unraid will pull `ghcr.io/julesdg6/scrapegoat:latest` automatically.

4. **Access the UI**:  
   Open `http://<unraid-ip>:7860` in your browser.

> **Tip:** Make sure `scrapegoat`, `ollama`, and `searxng` containers are on the same custom Docker network (e.g., `br0` bridge or a custom bridge) so that the hostnames resolve correctly. You can set the network for each container in the Docker settings under *Advanced View*.

---

### Option C — Local (non-Docker) Setup

1. Clone the repository:
```bash
git clone https://github.com/julesdg6/ScrapeGoat.git
cd ScrapeGoat
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install and start [Ollama](https://ollama.com), then pull a model:
```bash
ollama pull qwen:0.5b
```
   **Optional — image analysis (llava):**  
   To enable image analysis, pull a multimodal vision model and set the environment variable:
   ```bash
   ollama pull llava
   export OLLAMA_VISION_MODEL=llava
   ```
4. Run the Gradio app:
```bash
python3 ScrapeGoat_gradio_app.py
```
Or the Telegram bot (set `TELEGRAM_BOT_TOKEN` env var first):
```bash
export TELEGRAM_BOT_TOKEN=your_token_here
python3 ScrapeGoat.py
```

---

## Usage

### Gradio Web UI
1. Open **http://localhost:7860** (or your server IP + port).
2. Use the **Flexible Chat** tab to interact with the bot in a single unified interface:
   - Type any question and click **Submit** for a direct LLM answer.
   - Include a URL in your message to have it scraped automatically before answering.
   - Upload an image to analyse it (requires a vision model such as `llava`).
   - Upload an audio file to transcribe speech or analyse music (BPM, key, etc.).
3. The **URL Input** / **QnA with Website** tabs remain available for the original two-step workflow.
4. Use the **Image Generation** tab to generate images (requires `A1111_HOST` / `A1111_PORT`).
5. Use the **Settings** tab to customise the system prompt.

### Telegram Bot
1. Set up your bot via [BotFather](https://core.telegram.org/bots#botfather) and copy the token into `TELEGRAM_BOT_TOKEN`.
2. Run `python3 ScrapeGoat.py`.
3. Send `/start` in Telegram, provide a URL, then ask questions.

---

## Contributing

Contributions are welcome! To contribute:

- Fork the repository.
- Make changes in your fork.
- Submit a pull request with a clear description of your changes.

## License

This project is licensed under the terms of the MIT license. See the LICENSE file for details.

## Contact

If you have any questions or suggestions, feel free to open an issue on GitHub.

## No liability for the Developer

- Usage of this software for attacking targets without prior mutual consent is illegal.
- It is the end user's responsibility to obey all applicable local, state and federal laws.
- Developers of this software assume no liability and are not responsible for any misuse or damage caused by this program
by third parties using the software in violation of laws and regulations.
