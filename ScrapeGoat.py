import requests, json, os, re, ollama, time, logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fp.fp import FreeProxy
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from datetime import datetime
from aiogram import Bot, Dispatcher, executor, types

# Path to shared configuration file (must match the Gradio app's CONFIG_PATH)
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.json")

# Path to scraped content database JSON (shared with Docker / Unraid volume mounts)
DB_PATH = os.environ.get("DB_PATH", "db.json")

# Optional: URL of a running SearXNG instance used as the web search/scraping backend.
# When set, ScrapeGoat queries SearXNG's JSON API to discover URLs instead of
# crawling sites directly.  Falls back to proxy-based crawling when empty.
SEARXNG_HOST = os.environ.get("SEARXNG_HOST", "")


def load_allowed_user_ids() -> list:
    """Return the list of permitted Telegram user IDs from the config file.

    Returns an empty list when the config file is missing or contains no IDs,
    which means all users are allowed (backward-compatible default).
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        return [int(uid) for uid in data.get("allowed_telegram_user_ids", [])]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []

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

def search_with_searxng(query):
    """Search using SearXNG's JSON API and return a list of result URLs.

    Queries ``SEARXNG_HOST/search?q=<query>&format=json`` and returns the list
    of result URLs.  Returns an empty list when SEARXNG_HOST is not configured
    or the request fails.
    """
    if not SEARXNG_HOST:
        return []
    try:
        api_url = f"{SEARXNG_HOST.rstrip('/')}/search"
        response = requests.get(
            api_url,
            params={"q": query, "format": "json"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return [r["url"] for r in data.get("results", []) if r.get("url")]
    except Exception as exc:
        logging.warning(f"SearXNG search failed for query '{query}': {exc}")
        return []

def save_to_db(text, url):
    timestamp = datetime.now().isoformat()
    # Load existing data from db.json
    try:
        with open(DB_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    # Create a new entry with the domain name as key
    website = {'date': timestamp, 'text': text}
    new_entry = {'start_url': url, 'data' : website} 

    # Append new entry to the data list
    data.append(new_entry)

    # Write data back to db.json
    with open(DB_PATH, 'w') as f:
        json.dump(data, f, indent=4)


# Get context from question and txt file
def get_context(question,text,chunk_size=500,chunk_overlap=100):
    print("Embedding model started ...")
    all_scraped_text = text
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    documents = text_splitter.create_documents([all_scraped_text])
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path=os.environ.get("QDRANT_PATH", "./tmp/local_qdrant"),
        collection_name="telegram_data",
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 3})
    
    context = retriever.get_relevant_documents(question)
    
    print(f"Emdeggind Model returned: {context}")

    return context

def scrape_webpages(urls, proxy=None):
    # proxy may be None when called via SearXNG path; requests.get accepts None gracefully.
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

def generate_answer_local(question,context):
    print("LLM is generating answer ...")

    prompt = f"""Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer only factual information based on the context.
    Context: {context}.\n
    Question: {question}
    Helpful Answer:"""

    response = ollama.chat(
        model=os.environ.get('OLLAMA_MODEL', 'qwen:0.5b'),
        messages=[
            {
                'role': 'system',
                'content': 'You are a question answering AI Bot that uses context from the user prompt to answer the question.',
            },
            {
                'role': 'user',
                'content': prompt,
            },
        ],
    )
    
    output = response['message']['content']
    
    # Return the generated text
    print("LLM output:\n")
    return output

def generate_answer_pplx(question,context):
    
    prompt = f"""Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer only factual information based on the context.
    Context: {context}.\n
    Question: {question}
    Helpful Answer:"""

    pplx_key = ""
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "pplx-7b-chat",
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a question answering AI Bot that uses context from the user prompt to answer the question."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer " + pplx_key
    }
    response = requests.post(url, json=payload, headers=headers)
    
    json_data = response.text
    
    # Parse the JSON data
    parsed_json = json.loads(json_data)

    # Access and print the "content"
    answer = parsed_json["choices"][0]["message"]["content"]
    
    print("Answer:\n", answer)
    
    return answer

def analyze_website(start_url):

    try:
        with open(DB_PATH, 'r') as file:
            data = json.load(file)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    for entry in data:
        if start_url in entry['start_url']: # ADD check for today's scraped website data, not longer
            print('Website is already scraped today!')
            all_scraped_texts = entry['data']['text']
            return all_scraped_texts

    print("Scraper activated!")

    # Prefer SearXNG for link discovery when a host is configured.
    if SEARXNG_HOST:
        domain = get_domain(start_url)
        searxng_urls = search_with_searxng(f"site:{domain}")
        if searxng_urls:
            print(f"SearXNG returned {len(searxng_urls)} URL(s) for site:{domain}")
            all_scraped_texts = scrape_webpages(searxng_urls)
            save_to_db(all_scraped_texts, start_url)
            return all_scraped_texts
        logging.warning(
            f"SearXNG returned no results for site:{domain}; falling back to direct crawl."
        )

    proxy = get_proxy()

    # Scrape all the links from the given start URL using the proxy
    all_links = scrape_site_links(start_url, proxy)

    # Scrape the content from all the links obtained, using the proxy
    all_scraped_texts = scrape_webpages(all_links, proxy)

    save_to_db(all_scraped_texts,start_url)

    return all_scraped_texts

if __name__ == '__main__':
    # Replace with your actual bot token, or set the TELEGRAM_BOT_TOKEN environment variable
    API_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not API_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required to run the Telegram bot.")

    # Initialize bot and dispatcher
    bot = Bot(token=API_TOKEN)
    dp = Dispatcher(bot)

    # State storage
    state_storage = {}

    @dp.message_handler(commands=['start'])
    async def cmd_start(message: types.Message):
        allowed = load_allowed_user_ids()
        if allowed and message.from_user.id not in allowed:
            await message.reply("Sorry, you are not authorized to use this bot.")
            return
        # Each user gets their own isolated state entry keyed by chat_id
        state_storage[message.chat.id] = {'state': 'waiting_for_link'}
        await message.reply('Please provide a website URL.')

    @dp.message_handler(content_types=types.ContentType.TEXT)
    async def process_message(message: types.Message):
        allowed = load_allowed_user_ids()
        if allowed and message.from_user.id not in allowed:
            await message.reply("Sorry, you are not authorized to use this bot.")
            return
        chat_id = message.chat.id
        state = state_storage.get(chat_id, {}).get('state')
        
        if state == 'waiting_for_link':
            website_text = analyze_website(message.text)
            state_storage[chat_id]['website_text'] = website_text
            state_storage[chat_id]['state'] = 'ready_to_chat'
            await message.reply('Link accepted and analyzed. You can now ask questions.')
            
        elif state == 'ready_to_chat':
            context = get_context(message.text, state_storage[chat_id]['website_text'])
            response = generate_answer_local(message.text, context)
            await message.reply(response)

    executor.start_polling(dp, skip_updates=True)




### CLI main() function below

'''

def main():

    print("Chat with Website")

    start_url = input("URL: ")

    if start_url is not None:
        
        website_text = analyze_website(start_url)
        
        print("Text prepared")

        while True:

            question = input("question: ")

            if question:

                # Generate the context for the question using the scraped texts from the embedding model
                context = get_context(question,         website_text)

                # Use the context to perform inference and generate an answer to the question
                #result = generate_answer_local(question, context)
                result = generate_answer_pplx(question, context)

                # Print the result of the inference
                print(result)

'''
