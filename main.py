from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import requests, re, time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import asyncio
from playwright.async_api import async_playwright

app = FastAPI()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key=" ") # Gemini Api Key
gen_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
chat_session = None  

context_store = {}

class URLRequest(BaseModel):
    url: str
    max_pages: int = 50

class QuestionRequest(BaseModel):
    question: str

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_links(base_url, soup):
    links = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        full_url = urljoin(base_url, href)
        if is_valid_url(full_url) and urlparse(base_url).netloc in full_url:
            links.add(full_url)
    return links

def get_all_links(base_url, soup):
    links = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        full_url = urljoin(base_url, href)
        if (
            is_valid_url(full_url)
            and urlparse(base_url).netloc in full_url
            and not full_url.lower().endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx"))
        ):
            links.add(full_url)
    return links

def scrape_page(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code != 200:
            return "", []
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        links = get_all_links(url, soup)
        return text, list(links)
    except Exception as e:
        print(f"[Error scraping] {url}: {e}")
        return "", []

def crawl_website(start_url, max_pages=50):
    visited = set()
    to_visit = [start_url]
    content = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        print(f"Crawling: {url}")
        page_text, links = scrape_page(url)
        visited.add(url)
        if page_text:
            content.append(page_text)
        for link in links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)
        time.sleep(1)
        print(f"Total pages crawled: {len(visited)}")
    return "\n".join(content)

async def scrape_page_js(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=15000)
            await page.wait_for_load_state("load")
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            links = get_all_links(url, soup)
            await browser.close()
            return text, list(links)
    except Exception as e:
        print(f"[JS Error scraping] {url}: {e}")
        return "", []
    
async def crawl_website_js(start_url, max_pages=50):
    visited = set()
    to_visit = [start_url]
    content = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        print(f"Crawling: {url}")
        page_text, links = await scrape_page_js(url)
        visited.add(url)
        if page_text:
            content.append(page_text)
        for link in links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)
        await asyncio.sleep(1)
        print(f"Total pages crawled: {len(visited)}")
    return "\n".join(content)

def chunk_text(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chunk_size:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
#     return chunks

def get_top_chunks(query, chunks, k=5):
    embeddings = embedding_model.encode(chunks)
    query_embedding = embedding_model.encode([query])
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    top_indices = np.argsort(distances)[:k]
    return [chunks[i] for i in top_indices]

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

@app.post("/remember")
async def remember_url(req: URLRequest):
    global chat_session, gen_model

    genai.configure(api_key=" ")  # API key

    chat_session = None
    gen_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    chat_session = gen_model.start_chat()

    # 🕸️ Crawl and chunk website content
    # text = crawl_website(req.url, max_pages=500)
    text = await crawl_website_js(req.url, max_pages=50)
    chunks = chunk_text(text)
    context_store["user"] = chunks

    context_blob = "\n".join(chunks[:30])
    chat_session.send_message(f"This is the info from the website:\n{context_blob} but you can use your own thoughts")

    return {"message": f"Scraped and stored context from {req.url} into Gemini session."}

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    global chat_session
    if chat_session is None:
        return {"answer": "Please scrape a website first using the remember URL box."}

    all_chunks = context_store.get("user", [])
    top_chunks = get_top_chunks(req.question, all_chunks, k=5)
    condensed_context = "\n".join(top_chunks)

    prompt = (
        f"From the following website content, extract the specific answer to this question: '{req.question}'.\n"
        f"If the answer is a number like an interest rate, give only the number.\n\n"
        f"Context:\n{condensed_context}"
    )

    response = chat_session.send_message(prompt)
    return {"answer": response.text}


# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from fastapi import FastAPI
# from pydantic import BaseModel
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import requests, re, time
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse

# app = FastAPI()

# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# genai.configure(api_key="AIzaSyCQRPe3ituOaIQB5Ut_-gxARgLK-96sRyc")
# gen_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
# chat_session = None
# context_store = {}

# class URLRequest(BaseModel):
#     url: str
#     max_pages: int = 10
#     query: str

# class QuestionRequest(BaseModel):
#     question: str

# def is_valid_url(url):
#     parsed = urlparse(url)
#     return bool(parsed.netloc) and bool(parsed.scheme)

# def get_all_links(base_url, soup):
#     links = set()
#     for anchor in soup.find_all("a", href=True):
#         href = anchor["href"]
#         full_url = urljoin(base_url, href)
#         if is_valid_url(full_url) and urlparse(base_url).netloc in full_url:
#             links.add(full_url)
#     return links

# def scrape_page(url):
#     try:
#         headers = {
#             "User-Agent": "Mozilla/5.0"
#         }
#         response = requests.get(url, headers=headers, timeout=8)
#         if response.status_code != 200:
#             return "", []
#         soup = BeautifulSoup(response.text, "html.parser")
#         text = soup.get_text(separator="\n", strip=True)
#         links = get_all_links(url, soup)
#         return text, list(links)
#     except Exception as e:
#         print(f"[Scraping error] {url}: {e}")
#         return "", []

# def crawl_website(start_url, max_pages=10):
#     visited = set()
#     to_visit = [start_url]
#     content = []

#     while to_visit and len(visited) < max_pages:
#         url = to_visit.pop(0)
#         if url in visited:
#             continue
#         print(f"Crawling: {url}")
#         page_text, links = scrape_page(url)
#         visited.add(url)
#         if page_text:
#             content.append(page_text)
#         for link in links:
#             if link not in visited and link not in to_visit:
#                 to_visit.append(link)
#         time.sleep(1)  # throttle

#     print(f"Total pages crawled: {len(visited)}")
#     return "\n".join(content)

# def chunk_text(text, max_chunk_size=500):
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     chunks, current = [], ""
#     for sentence in sentences:
#         if len(current) + len(sentence) <= max_chunk_size:
#             current += sentence + " "
#         else:
#             chunks.append(current.strip())
#             current = sentence + " "
#     if current:
#         chunks.append(current.strip())
#     return chunks

# def get_top_chunks(query, chunks, k=5):
#     if not chunks:
#         return []
#     embeddings = embedding_model.encode(chunks)
#     query_embedding = embedding_model.encode([query])
#     distances = np.linalg.norm(embeddings - query_embedding, axis=1)
#     top_indices = np.argsort(distances)[:k]
#     return [chunks[i] for i in top_indices]

# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/")
# async def serve_ui():
#     return FileResponse("static/index.html")

# @app.post("/remember")
# async def remember_url(req: URLRequest):
#     global chat_session, gen_model

#     chat_session = None
#     genai.configure(api_key="AIzaSyCQRPe3ituOaIQB5Ut_-gxARgLK-96sRyc")  # key
#     gen_model = genai.GenerativeModel("gemini-1.5-pro-latest")
#     chat_session = gen_model.start_chat()

#     text = crawl_website(req.url, max_pages=req.max_pages)
#     chunks = chunk_text(text)
#     context_store["user"] = chunks
#     context_blob = "\n".join(chunks[:30])
#     chat_session.send_message(
#     f"This is some info from the website. You may use it as supporting context. "
#     f"If you can’t find an answer in this content, you can answer using your own knowledge as well.\n\n"
#     f"Context:\n{context_blob}")
#     # chat_session.send_message(f"This is the context from the website:\n{context_blob}")
#     return {
#         "message": f"Done."
#     }

# @app.post("/ask")
# async def ask_question(req: QuestionRequest):
#     global chat_session
#     if chat_session is None:
#         return {"answer": "Please scrape a website first."}

#     chunks = context_store.get("user", [])
#     top_chunks = get_top_chunks(req.question, chunks)
#     context_blob = "\n".join(top_chunks)

#     prompt = (
#         f"Use this website content to answer the question.\n"
#         f"If a number is needed, just return the number.\n\n"
#         f"Context:\n{context_blob}\n\n"
#         f"Question: {req.question}"
#     )

#     try:
#         response = chat_session.send_message(prompt)
#         return {"answer": response.text}
#     except Exception as e:
#         return {"answer": f"❗ Error from Gemini: {e}"}
