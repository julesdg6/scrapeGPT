# ScrapeGPT 
## Web App & Telegram Bot for Web Content Analysis and Question Answering

ScrapeGPT is a Gradio web app (and optional Telegram bot) that scrapes websites and answers questions about them using a local LLM via [Ollama](https://ollama.com). The idea came from a job interview assignment to "study our company website and provide new product recommendations" — this automates the entire process.

> **Note:** There are two entry points:
> - `scrapeGPT_gradio_app.py` — Gradio web UI (recommended, used by Docker)
> - `scrapeGPT.py` — Telegram bot (includes a commented CLI `main()` at the bottom)

## Features

- **Flexible Chat**: A single unified interface that automatically detects the type of input and routes to the correct handler:
  - **URL in prompt** → scrapes the page(s) and answers using retrieved content
  - **Image upload** → analyses the image using a multimodal Ollama model (e.g. `llava`)
  - **Audio upload** → transcribes speech (OpenAI Whisper) *or*, if no clear speech is found, analyses the audio as music (BPM, estimated key, spectral centroid via librosa)
  - **Plain text** → answers directly with the configured LLM
- **Web Scraping**: Automatically scrapes text from provided URLs, including PDF files.
- **Context Retrieval**: Utilises embeddings and retrieval models to extract relevant context from scraped content.
- **Question Answering**: Generates answers to user questions based on the retrieved context.
- **Robots.txt Parsing**: Respects the website's robots.txt to avoid scraping restricted areas.
- **Database Management**: Stores scraped content in a database for future reference and quick access.
- **Proxy Support**: Uses rotating proxies to bypass geo-restrictions and anonymise requests.
- **LLM-based**: Connects to a local [Ollama](https://ollama.com) instance (including Ollama running in another container).

### Additional models required for new features

| Feature | Required model / package | How to enable |
|---|---|---|
| Image analysis | A multimodal Ollama model, e.g. `llava` | `ollama pull llava` then set `OLLAMA_VISION_MODEL=llava` |
| Speech transcription | OpenAI Whisper (`openai-whisper` Python package) | Already in `requirements.txt`; Whisper downloads its weights on first use |
| Music analysis | librosa + soundfile Python packages | Already in `requirements.txt`; no extra setup required |

---

## Getting Started

### Option A — Docker (recommended)

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.

#### 1. Clone the repository
```bash
git clone https://github.com/julesdg6/scrapeGPT.git
cd scrapeGPT
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

#### 3. Start the stack
```bash
docker compose up -d
```
This starts both **scrapeGPT** and **Ollama** containers.

#### 4. Pull the Ollama model
```bash
docker exec -it ollama ollama pull qwen:0.5b
```

#### 5. Open the web UI
Navigate to **http://localhost:7860** in your browser.

---

#### Manual `docker build` (without Compose)

```bash
# Build the image
docker build -t scrapegpt:latest .

# Run with an existing Ollama container on the same network
docker run -d \
  --name scrapegpt \
  -p 7860:7860 \
  -e OLLAMA_HOST=http://ollama:11434 \
  -e OLLAMA_MODEL=qwen:0.5b \
  -v scrapegpt_data:/data \
  -v scrapegpt_qdrant:/app/tmp/local_qdrant \
  --network <your-docker-network> \
  scrapegpt:latest
```

> **Connecting to Ollama in another container:**  
> Both containers must share a Docker network. With Compose this is automatic. For manual setups, create a network first:
> ```bash
> docker network create scrapegpt-net
> docker run -d --name ollama --network scrapegpt-net -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama
> docker run -d --name scrapegpt --network scrapegpt-net -p 7860:7860 -e OLLAMA_HOST=http://ollama:11434 scrapegpt:latest
> ```

---

### Option B — Unraid Install

ScrapeGPT includes an [Unraid Community Applications](https://unraid.net/community/apps) template.

#### Prerequisites
- Unraid with the **Community Applications** plugin installed.
- An **Ollama** container already running on your Unraid server (install `ollama/ollama` from Community Applications or via the Docker tab).

#### Step-by-step

1. **Install Ollama** (if not already running):
   - In Unraid, go to **Apps** → search for `Ollama` → install `ollama/ollama`.
   - After it starts, pull your model via the Unraid terminal:
     ```bash
     docker exec -it ollama ollama pull qwen:0.5b
     ```

2. **Install ScrapeGPT using the template**:
   - In Unraid, go to **Docker** → **Add Container**.
   - Paste the template URL in the *Template URL* field:
     ```
     https://raw.githubusercontent.com/julesdg6/scrapeGPT/main/unraid/scrapeGPT.xml
     ```
   - Click **Load** (or save to your templates folder).
   - Configure the variables:

     | Variable | Recommended Value | Notes |
     |---|---|---|
     | **Ollama Host** | `http://ollama:11434` | Use container name if on same Docker network; otherwise use your server IP, e.g. `http://192.168.1.100:11434` |
     | **Ollama Model** | `qwen:0.5b` | Must be pulled into Ollama first |
     | **Ollama Vision Model** | `llava` | For image analysis; must be pulled first: `ollama pull llava` |
     | **WebUI Port** | `7860` | Host port to access Gradio |
     | **App Data** | `/mnt/user/appdata/scrapegpt` | Stores `db.json` |
     | **Qdrant Vector Store** | `/mnt/user/appdata/scrapegpt/qdrant` | Stores embedding vectors |

   - Click **Apply** to start the container.

3. **Access the UI**:  
   Open `http://<unraid-ip>:7860` in your browser.

> **Tip:** Make sure both `scrapegpt` and `ollama` containers are on the same custom Docker network (e.g., `br0` bridge or a custom bridge) so that `http://ollama:11434` resolves correctly. You can set the network for each container in the Docker settings under *Advanced View*.

---

### Option C — Local (non-Docker) Setup

1. Clone the repository:
```bash
git clone https://github.com/julesdg6/scrapeGPT.git
cd scrapeGPT
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install and start [Ollama](https://ollama.com), then pull a model:
```bash
ollama pull qwen:0.5b
```
4. Run the Gradio app:
```bash
python scrapeGPT_gradio_app.py
```
Or the Telegram bot (set `TELEGRAM_BOT_TOKEN` env var first):
```bash
export TELEGRAM_BOT_TOKEN=your_token_here
python scrapeGPT.py
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
4. Use the **Settings** tab to customise the system prompt.

### Telegram Bot
1. Set up your bot via [BotFather](https://core.telegram.org/bots#botfather) and copy the token into `TELEGRAM_BOT_TOKEN`.
2. Run `python scrapeGPT.py`.
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

