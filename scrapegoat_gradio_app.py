import gradio as gr
import os
import time
import base64
import logging
import requests, json, re, ollama
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fp.fp import FreeProxy
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen:0.5b")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava")
# Optional: URL of a running WhisperLive service (e.g. http://whisperlive:9090).
# When set, audio transcription is delegated to that service instead of using a
# locally-installed openai-whisper model.  Leave empty to use local whisper
# (if installed) or to disable speech transcription entirely.
WHISPERLIVE_HOST = os.environ.get("WHISPERLIVE_HOST", "")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions based only on the provided context."

# Path to shared configuration file (persistent across restarts)
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.json")

# Minimum number of words in a Whisper transcript to classify audio as speech
_SPEECH_WORD_THRESHOLD = 3

# Local Whisper model – loaded once on first audio request (only used when
# WHISPERLIVE_HOST is not configured and openai-whisper is installed)
_whisper_model = None


def _load_config() -> dict:
    """Load configuration from CONFIG_PATH, returning an empty dict on failure."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(data: dict) -> None:
    """Persist *data* to CONFIG_PATH, creating parent directories as needed."""
    config_dir = os.path.dirname(CONFIG_PATH)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


# Runtime-configurable settings (can be changed via the Settings tab)
_cfg = _load_config()
_raw_ids = _cfg.get("allowed_telegram_user_ids", [])
try:
    _allowed_ids = [int(uid) for uid in _raw_ids]
except (ValueError, TypeError):
    _allowed_ids = []
current_settings = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    # List of permitted Telegram user IDs (integers); empty = no restriction
    "allowed_telegram_user_ids": _allowed_ids,
}

# Shared state for the URL Input / QnA tabs
shared_result = None

# Proxy init
def get_proxy():
    print("Starting proxy ...")
    proxy_url = FreeProxy(country_id=['US','CA','FR','NZ','SE','PT','CZ','NL','ES','SK','UK','PL','IT','DE','AT','JP'],https=True,rand=True,timeout=3).get()
    proxy_obj = {
        "http": proxy_url,
        "https": proxy_url,
    }

    print(f"Proxy generated: {proxy_url}")
    
    return proxy_obj

def scrape_webpages(urls,proxy):
    print("Scraping text from webpages from each of the links ...")
    scraped_texts = []
    for url in urls:
        try:
            if url.endswith('.pdf'):
                response = requests.get(url, proxies=proxy)
                reader = PdfReader(BytesIO(response.content))
                number_of_pages = len(reader.pages)

                for p in range(number_of_pages):

                    page = reader.pages[p]
                    text = page.extract_text()
                    scraped_texts.append(text)
            else:
                page = requests.get(url,proxies=proxy)
                soup = BeautifulSoup(page.content, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                scraped_texts.append(text)

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    
    all_scraped_text = '\n'.join(scraped_texts)
    print("Finished scraping the text from webpages!")
    return all_scraped_text

def get_domain(url):
    return urlparse(url).netloc

def get_robots_file(url,proxy):
    robots_url = urljoin(url, '/robots.txt')
    try:
        response = requests.get(robots_url,proxies=proxy)
        return response.text
    except Exception as e:
        print(f"Error fetching robots.txt: {e}")
        return None

def parse_robots(content):
    # This function assumes simple rules without wildcards, comments, etc.
    # For a full parser, consider using a library like robotparser.
    if not content:
        return []
    disallowed = []
    for line in content.splitlines():
        if line.startswith('Disallow:'):
            path = line[len('Disallow:'):].strip()
            disallowed.append(path)
    return disallowed

def is_allowed(url, disallowed_paths, base_domain):
    parsed_url = urlparse(url)
    if parsed_url.netloc != base_domain:
        return False
    for path in disallowed_paths:
        if parsed_url.path.startswith(path):
            return False
    return True

def scrape_site_links(start_url, proxy):
    visited_links = set()
    not_visited_links = set()
    to_visit = [start_url]
    base_domain = get_domain(start_url)
    disallowed_paths = parse_robots(get_robots_file(start_url, proxy))
    last_found_time = time.time()  # Track the last time a link was found

    while to_visit:
        # Break the loop if  30 seconds have passed without finding a new link
        if time.time() - last_found_time >  15:
            print("FINISHED scraping the links")
            break

        current_url = to_visit.pop(0)
        if current_url not in visited_links and is_allowed(current_url, disallowed_paths, base_domain):
            visited_links.add(current_url)
            try:   
                print(f"{current_url}")
                response = requests.get(current_url, proxies=proxy)
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(current_url, link['href'])
                    if new_url not in visited_links:
                        to_visit.append(new_url)
                        last_found_time = time.time()  # Update the last found time
            except Exception as e:
                print(f" !!! COULD NOT VISIT: {current_url}")
                not_visited_links.add(current_url)

    return visited_links

def analyze_website(start_url):
    global shared_result
    
    proxy = get_proxy()
    
    # Scrape all the links from the given start URL using the proxy
    all_links = scrape_site_links(start_url, proxy)

    # Scrape the content from all the links obtained, using the proxy
    full_text = scrape_webpages(all_links, proxy)
    
    shared_result = full_text
    
    txt = "Website Analyzed!"
    
    return txt

def ask_questions(question_text):
    global shared_result
    if shared_result is None:
        return "No result available yet."
    else:
        return _answer_from_text(question_text, shared_result)

def save_settings(system_prompt):
    system_prompt = system_prompt.strip()
    if not system_prompt:
        return "Error: System Prompt cannot be empty."
    current_settings["system_prompt"] = system_prompt
    return "Settings saved successfully!"


def save_user_ids(user_ids_text):
    """Parse *user_ids_text* (one numeric ID per line) and persist to config."""
    ids = []
    for line in user_ids_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            return f"Error: '{line}' is not a valid numeric Telegram user ID."
    current_settings["allowed_telegram_user_ids"] = ids
    cfg = _load_config()
    cfg["allowed_telegram_user_ids"] = ids
    _save_config(cfg)
    return f"Saved {len(ids)} allowed Telegram user ID(s)."

# ── Flexible chat helpers ──────────────────────────────────────────────────────

def _answer_from_text(question, scraped_text):
    """Build a RAG answer from *scraped_text* for the given *question*.

    Shared by both ``ask_questions`` and ``handle_flexible_request``.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([scraped_text])
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        path="./tmp/local_qdrant",
        collection_name="data",
    )
    retriever = qdrant.as_retriever()
    template = f"""{current_settings["system_prompt"]}

Context: {{context}}

Question: {{question}}
"""
    prompt_tmpl = ChatPromptTemplate.from_template(template)
    model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt_tmpl
        | model
        | StrOutputParser()
    )
    return chain.invoke(question)

def extract_urls(text):
    """Return all http/https URLs found in *text*, stripping trailing punctuation."""
    raw = re.findall(r'https?://[^\s]+', text)
    return [url.rstrip('.,;:!?)]\'"') for url in raw]

def analyze_image_with_ollama(image_path, question=""):
    """Analyze an image using a multimodal Ollama vision model (e.g. llava).

    Requires OLLAMA_VISION_MODEL to be set to a model that supports image
    inputs (e.g. 'llava', 'bakllava').  Pull it first:
        ollama pull llava
    """
    if not question:
        question = (
            "Describe this image in detail. "
            "Transcribe any text visible in it. "
            "If you can identify the subject, provide additional context."
        )
    with open(image_path, "rb") as fh:
        image_data = base64.b64encode(fh.read()).decode("utf-8")
    response = ollama.chat(
        model=OLLAMA_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [image_data],
            }
        ],
    )
    return response["message"]["content"]

def _transcribe_with_whisperlive(audio_path):
    """Transcribe *audio_path* using the external WhisperLive service.

    The service must expose an OpenAI-compatible REST endpoint at
    ``WHISPERLIVE_HOST/v1/audio/transcriptions``.  This is satisfied by
    ``ghcr.io/collabora/whisperlive-gpu:latest`` as well as other
    compatible self-hosted Whisper servers.

    Returns the transcript string, or raises on failure.
    """
    url = f"{WHISPERLIVE_HOST.rstrip('/')}/v1/audio/transcriptions"
    with open(audio_path, "rb") as fh:
        response = requests.post(url, files={"file": fh}, data={"model": "whisper-1"})
    if not response.ok:
        raise RuntimeError(
            f"WhisperLive service at {WHISPERLIVE_HOST} returned HTTP {response.status_code}: {response.text[:200]}"
        )
    return response.json().get("text", "").strip()


def _transcribe_with_local_whisper(audio_path):
    """Transcribe *audio_path* using the locally installed openai-whisper package.

    Returns the transcript string, or raises ImportError if the package is
    not installed.
    """
    global _whisper_model
    import whisper  # optional dependency
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    result = _whisper_model.transcribe(audio_path)
    return result.get("text", "").strip()


def analyze_audio_file(audio_path):
    """Transcribe speech **or** analyse music depending on the audio content.

    Speech transcription is attempted first:
      - If ``WHISPERLIVE_HOST`` is set, the external WhisperLive service is
        used (``ghcr.io/collabora/whisperlive-gpu:latest`` or compatible).
      - Otherwise the locally-installed ``openai-whisper`` package is used as
        a fallback (if available).
      - If neither is configured the transcription step is skipped and the
        audio is analysed as music.

    Music analysis uses librosa to extract BPM, estimated key, and spectral
    centroid.
    """
    import librosa
    import numpy as np

    # ── Speech transcription ──────────────────────────────────────────────
    transcript = ""
    if WHISPERLIVE_HOST:
        try:
            transcript = _transcribe_with_whisperlive(audio_path)
        except Exception as exc:
            logging.warning(
                f"WhisperLive transcription failed at {WHISPERLIVE_HOST}: {exc}. "
                "Audio will be analysed as music instead."
            )
    else:
        try:
            transcript = _transcribe_with_local_whisper(audio_path)
        except ImportError:
            logging.info(
                "openai-whisper is not installed and WHISPERLIVE_HOST is not set; "
                "speech transcription is unavailable. Audio will be analysed as music."
            )

    # If we got a meaningful transcript treat it as speech
    if transcript and len(transcript.split()) > _SPEECH_WORD_THRESHOLD:
        return f"Speech transcription:\n{transcript}"

    # ── Music analysis ────────────────────────────────────────────────────
    y, sr = librosa.load(audio_path)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Estimated key via chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = key_names[int(np.argmax(chroma.mean(axis=1)))]

    # Spectral centroid (brightness indicator)
    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

    lines = [
        "Music Analysis:",
        f"- Estimated BPM: {float(tempo):.1f}",
        f"- Estimated Key: {key}",
        f"- Spectral Centroid: {spectral_centroid:.1f} Hz",
    ]
    if transcript:
        lines.append(f"- Whisper transcript (partial): {transcript}")
    return "\n".join(lines)

def handle_flexible_request(text, image, audio):
    """Route the request to the appropriate handler.

    Priority order:
      1. Image input  → multimodal image analysis (llava / OLLAMA_VISION_MODEL)
      2. Audio input  → speech transcription or music analysis
      3. URL in text  → scrape URL(s) and answer based on scraped content
      4. Plain text   → direct LLM answer
    """
    # 1. Image
    if image is not None:
        try:
            return analyze_image_with_ollama(image, question=text or "")
        except Exception as exc:
            return f"Image analysis failed: {exc}"

    # 2. Audio
    if audio is not None:
        try:
            return analyze_audio_file(audio)
        except Exception as exc:
            return f"Audio analysis failed: {exc}"

    if not text or not text.strip():
        return "Please enter a message, upload an image, or upload an audio file."

    # 3. URL(s) in text
    urls = extract_urls(text)
    if urls:
        scraped_parts = []
        for url in urls:
            try:
                proxy = get_proxy()
                all_links = scrape_site_links(url, proxy)
                page_text = scrape_webpages(all_links, proxy)
                scraped_parts.append(page_text)
            except Exception as exc:
                scraped_parts.append(f"[Could not scrape {url}: {exc}]")
        combined_text = "\n".join(scraped_parts)
        try:
            return _answer_from_text(text, combined_text)
        except Exception as exc:
            return f"Could not answer from scraped content: {exc}\n\nScraped text preview:\n{combined_text[:500]}"

    # 4. Plain Q&A
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": current_settings["system_prompt"]},
                {"role": "user", "content": text},
            ],
        )
        return response["message"]["content"]
    except Exception as exc:
        return f"LLM error: {exc}"

# Create the interfaces for each task
iface1 = gr.Interface(fn=analyze_website, inputs="text", outputs="text",title="Enter website URL")
iface2 = gr.Interface(fn=ask_questions, inputs="text", outputs="text",title="Ask questions to the website")

with gr.Blocks() as iface_chat:
    gr.Markdown("## Flexible Chat")
    gr.Markdown(
        "Enter a question or message below. "
        "The bot will automatically:\n"
        "- **Scrape a URL** if your message contains one, then answer from the page content\n"
        "- **Analyse an image** if you upload one (requires a vision-capable model, e.g. `llava`)\n"
        "- **Transcribe speech or analyse music** if you upload an audio file\n"
        "- **Answer directly** otherwise"
    )
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Your message",
                placeholder="Ask a question, paste a URL, or describe what you need…",
                lines=3,
            )
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Image (optional)",
                type="filepath",
            )
            audio_input = gr.Audio(
                label="Audio (optional)",
                type="filepath",
            )
    submit_btn = gr.Button("Submit")
    chat_output = gr.Textbox(label="Response", lines=12, interactive=False)
    submit_btn.click(
        fn=handle_flexible_request,
        inputs=[text_input, image_input, audio_input],
        outputs=chat_output,
    )

with gr.Blocks() as iface3:
    gr.Markdown("## Settings")
    gr.Markdown("Configure the system prompt used when answering questions.")
    system_prompt_input = gr.Textbox(
        label="System Prompt",
        value=current_settings["system_prompt"],
        lines=5,
        placeholder="Enter the system prompt for the LLM...",
    )
    save_btn = gr.Button("Save Settings")
    status_output = gr.Textbox(label="Status", interactive=False)
    save_btn.click(
        fn=save_settings,
        inputs=system_prompt_input,
        outputs=status_output,
    )

    gr.Markdown("---")
    gr.Markdown("### Telegram Access Control")
    gr.Markdown(
        "Enter the numeric Telegram User IDs that are permitted to use the bot, **one per line**. "
        "Leave empty to allow all users (no restriction)."
    )
    user_ids_input = gr.Textbox(
        label="Allowed Telegram User IDs",
        value="\n".join(str(uid) for uid in current_settings["allowed_telegram_user_ids"]),
        lines=8,
        placeholder="123456789\n987654321",
    )
    save_users_btn = gr.Button("Save User IDs")
    users_status_output = gr.Textbox(label="Status", interactive=False)
    save_users_btn.click(
        fn=save_user_ids,
        inputs=user_ids_input,
        outputs=users_status_output,
    )

# Combine the interfaces into a TabbedInterface
tabbed_interface = gr.TabbedInterface(
    [iface_chat, iface1, iface2, iface3],
    ["Flexible Chat", "URL Input", "QnA with Website", "Settings"],
)

# Launch the combined interface
tabbed_interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", 7860)))
