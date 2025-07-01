#Date Modified 7/1/2025
#DISCLAIMER: Some terms and conditions of certain sites or apis may change which could impact the legality of this code for commercial use.

import requests
import random
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from newspaper import Article, Config
import trafilatura
from bs4 import BeautifulSoup
import yfinance as yf
import feedparser
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout

# ------------------ Setup ------------------ #
NEWSAPI_KEYS = [
    "fc6f5446794b4b96acbf024f76c6074f",
    "5eb9b7a6221846f0b4dab6df45774bad",
    "3351706eb2fe4588bff8f4cff706b9a5",
    "d6b300c0ad224f7bb455b6be4f5406cb",
    "aff32ce132f44912830e90c3e15d8a83"
]
MARKETAUX_KEYS = [
    "vGc2ngli75vh0B0TLLZBYmLoP9j6JdlIO0xnYU1Z", 
    "l3kI5uLaJ7cmiYtzaqbmnTjf0RKzyxrqqkWPqzXF", 
    "56zM77P4FypSsNSmVOU4T2XoBpeyzznVZufahSAt", 
    "MF5YF01raQLHgQhrSznF3E5d1i2c0U9DcW0msbzP", 
    "yNItEjExQVMy5qLicMx3w9IvSDCK7Zc6qtHMjUAz"
]
ALPHA_KEYS = ["2GOHOV1X36HZDPHS", "B3FTT2KZWV3EOOMM", "GWVW30578EIUMAMT", "CDCQHAZTZBMRP4SN"]
POLYGON_KEYS = ["PeJay0iXYAG_fFa8B275Gu0zCHAFFypw", "B_v9if1uHwRU08BnqNU2c9IGyy_ZXuQF"]

# ------------------ Utilities & Headers ------------------ #
USER_AGENTS = [
    # Chrome – various platforms
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ... Chrome/125.0.0.0 Safari/537.36",    
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) ... Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) ... Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) ... Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_3 like Mac OS X) ... Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) ... Chrome/125.0.0.0 Mobile Safari/537.36"
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Upgrade-Insecure-Requests": "1"
    }

# ------------------ Async Utilities ------------------ #
async def extract_article_text_fallback_async(url, sleep_between=0.31, verbose=True):
    """
    Asynchronous wrapper that runs the blocking extract_article_text_fallback in a thread.
    """
    loop = asyncio.get_running_loop()
    # Run the synchronous extraction in a thread to avoid blocking the event loop
    text, source = await loop.run_in_executor(
        None, extract_article_text_fallback, url, sleep_between, verbose
    )
    return text, source

def extract_article_text_fallback(url, sleep_between=0.31, verbose=True):
    """Original blocking text extraction logic (uses newspaper3k, trafilatura, BeautifulSoup)."""
    try:
        time.sleep(random.uniform(0.15, sleep_between))  #Random short delay before starting newspaper3k (helps avoid bans/throttling)
        config = Config()
        config.browser_user_agent = random.choice(USER_AGENTS)#Setup newspaper3k with a random user-agent
        article = Article(url, config=config)
        article.download()
        article.parse()
        #If successful and non-empty, return the result
        if article.text.strip():
            if verbose: print(f"[✔] newspaper3k succeeded for {url}")
            return article.text.strip(), "newspaper3k"
    except Exception as e:
        if verbose: print(f"[newspaper3k error] {url}: {e}")
    try:
        time.sleep(random.uniform(0.15, sleep_between))  # delay before trying trafilatura
        downloaded = trafilatura.fetch_url(url) # trys fetching and extracting content using trafilatura
        if downloaded:
            result = trafilatura.extract(downloaded)
            if result and result.strip():
                if verbose: print(f"[✔] trafilatura succeeded for {url}")
                return result.strip(), "trafilatura"
    except Exception as e:
        if verbose: print(f"[trafilatura error] {url}: {e}")
    try:
        time.sleep(random.uniform(0.15, sleep_between))  # final fallback: raw request + BeautifulSoup
        r = requests.get(url, headers=get_headers(), timeout=3) # Try making a raw GET request with a timeout limit (timout is only for request)
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 30)
        if text.strip():
            if verbose: print(f"[✔] BeautifulSoup succeeded for {url}")
            return text.strip(), "beautifulsoup"
    except requests.exceptions.Timeout:
        if verbose: print(f"[BeautifulSoup timeout] {url}")
    except Exception as e:
        if verbose: print(f"[BeautifulSoup error] {url}: {e}")
    if verbose: print(f"[✖] All fallback methods failed for {url}")
    return "", "none"

def format_date(raw_date, source):
    try:
        if source == "newsapi":
            return datetime.fromisoformat(raw_date.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        elif source == "marketaux":
            return datetime.fromisoformat(raw_date).strftime("%Y-%m-%d")
        elif source == "alphavantage":
            return datetime.strptime(raw_date, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")
        elif source == "yfinance":
            return raw_date.strftime("%Y-%m-%d")
        elif source == "bing":
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        elif source == "finviz":
            parts = raw_date.split()
            if len(parts) == 1:
                return datetime.now().strftime('%Y-%m-%d')
            try:
                return datetime.strptime(parts[0], "%b-%d-%y").strftime('%Y-%m-%d')
            except:
                return datetime.now().strftime('%Y-%m-%d')
        return raw_date
    except Exception as e:
        print(f"[date parse error - {source}] {raw_date}: {e}")
        return ""

# ------------------ Source Fetching Functions (Async) ------------------ #
async def try_newsapi(ticker, limit):
    key = random.choice(NEWSAPI_KEYS) # Randomized for safety
    query = f"{ticker} stock news" # Improved query string chnage if you cna think of a better one
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={key}&language=en&pageSize={limit}"
    async with ClientSession() as session:
        async with session.get(url, headers=get_headers()) as resp:
            if resp.status != 200:
                print(f"[newsapi error] Status: {resp.status}")
                return []
            data = await resp.json()
    results = []
    for article in data.get("articles", []):
        if not article.get('url'):
            continue
        text, _ = await extract_article_text_fallback_async(article['url'])
        if not text:
            continue
        results.append({
            "date": format_date(article['publishedAt'], "newsapi"),
            "title": article['title'],
            "text": text,
            "url": article['url'],
            "ticker": ticker
        })
        if len(results) >= limit:
            break
    return results

async def try_marketaux(ticker, limit):
    key = random.choice(MARKETAUX_KEYS)# Randomized for safety
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "api_token": key,
        "symbols": ticker,
        "language": "en",
        "limit": limit * 2,
        "published_after": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    }
    async with ClientSession() as session:
        async with session.get(url, params=params, headers=get_headers()) as resp:
            print(f"[marketaux] Status: {resp.status}")
            if resp.status != 200:
                print(f"[marketaux error] Failed with status {resp.status}")
                return []
            data = await resp.json()
    articles = data.get("data", [])
    if not articles:
        print(f"[marketaux] No articles found for {ticker}")
        return []
    results = []
    for a in articles:
        article_url = a.get('url')
        if not article_url:
            continue
        text, _ = await extract_article_text_fallback_async(article_url)
        if not text:
            continue
        results.append({
            "date": format_date(a['published_at'], "marketaux"),
            "title": a['title'],
            "text": text,
            "url": article_url,
            "ticker": ticker
        })
        if len(results) >= limit:
            break
    return results

async def try_alphavantage(ticker, limit):
    key = random.choice(ALPHA_KEYS) # Randomized for safety
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": key,
        "sort": "LATEST"
    }
    async with ClientSession() as session:
        async with session.get("https://www.alphavantage.co/query", params=params, headers=get_headers()) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    results = []
    for a in data.get("feed", []):
        if not a.get('url'):
            continue
        text, _ = await extract_article_text_fallback_async(a['url'])
        if not text:
            continue
        results.append({
            "date": format_date(a['time_published'], "alphavantage"),
            "title": a['title'],
            "text": text,
            "url": a['url'],
            "ticker": ticker
        })
        if len(results) >= limit:
            break
    return results

async def try_yahoo_rss(ticker, limit):
    print(f"[yahoo_rss] Fetching RSS feed for {ticker}")
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        loop = asyncio.get_running_loop()
        # Run feedparser.parse in executor to avoid blocking the event loop
        feed = await loop.run_in_executor(None, feedparser.parse, url)
    except Exception as e:
        print(f"[yahoo_rss parse error] {e}")
        return []
    results = []
    seen = set()
    for entry in getattr(feed, "entries", []):
        article_url = entry.get("link")
        title = (entry.get("title") or "").strip()
        published = (entry.get("published") or "").strip()
        norm_url = (article_url or "").strip().lower()
        if not norm_url or norm_url in seen or not title:
            continue
        seen.add(norm_url)
        text, _ = await extract_article_text_fallback_async(article_url)
        if not text:
            continue
        results.append({
            "date": format_date(published, "bing"),
            "title": title,
            "text": text,
            "url": article_url,
            "ticker": ticker
        })
        if len(results) >= limit:
            break
    if not results:
        print(f"[yahoo_rss] No valid articles for {ticker}")
    return results

async def try_bing_news(ticker, limit):
    print(f"🔍 Scraping Bing News for {ticker}...")
    url = "https://www.bing.com/news/search"
    params = {"q": f"{ticker} stock news", "form": "QBNH", "setmkt": "en-US", "setlang": "en-US"} # More specific query ("stock news")
    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            async with session.get(url, params=params, headers=get_headers()) as resp:
                page_text = await resp.text()
    except Exception as e:
        print(f"[bing_news error] Exception during request: {e}")
        return []
    soup = BeautifulSoup(page_text, "html.parser")
    results = []
    seen = set()
    for tag in soup.find_all("a", {"class": "title"}):
        href = tag.get("href")
        title = tag.get_text(strip=True)
        if not href or href in seen or not title:
            continue
        seen.add(href)
        text, _ = await extract_article_text_fallback_async(href)
        if not text:
            continue
        results.append({
            "date": format_date("", "bing"),
            "title": title,
            "text": text,
            "url": href,
            "ticker": ticker
        })
        if len(results) >= limit:
            break
    return results

async def try_finviz(ticker, limit):
    print(f"📰 Scraping Finviz for {ticker}...")
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            async with session.get(url, headers=get_headers()) as resp:
                page_text = await resp.text()
    except Exception as e:
        print(f"[finviz error] {e}")
        return []
    soup = BeautifulSoup(page_text, "lxml")
    news_table = soup.find(id='news-table')
    if not news_table:
        return []
    results = []
    seen = set()
    for row in news_table.find_all('tr'):
        a_tag = row.a
        td_tag = row.td
        if a_tag and td_tag:
            headline = a_tag.get_text(strip=True)
            href = a_tag.get("href")
            norm_href = (href or "").strip().lower()
            if not norm_href or norm_href in seen or not headline:
                continue
            seen.add(norm_href)
            text, _ = await extract_article_text_fallback_async(href)
            if not text:
                continue
            raw_date = td_tag.text.strip()
            results.append({
                "date": format_date(raw_date, "finviz"),
                "title": headline,
                "text": text,
                "url": href,
                "ticker": ticker
            })
            if len(results) >= limit:
                break
    return results

async def try_polygon(ticker, limit):
    print(f"[polygon] Fetching news for {ticker}")
    key = random.choice(POLYGON_KEYS)
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": limit * 2,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": key
    }
    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            async with session.get(url, params=params, headers=get_headers()) as resp:
                if resp.status != 200:
                    print(f"[polygon error] Status code: {resp.status}")
                    return []
                data = await resp.json()
    except Exception as e:
        print(f"[polygon exception] {e}")
        return []
    articles = data.get("results", [])
    results = []
    for article in articles:
        article_url = article.get("article_url")
        title = article.get("title")
        published = article.get("published_utc")
        if not article_url or not title:
            continue
        text, _ = await extract_article_text_fallback_async(article_url)
        if not text:
            continue
        results.append({
            "date": format_date(published, "marketaux"),
            "title": title,
            "text": text,
            "url": article_url,
            "ticker": ticker
        })
        if len(results) >= limit:
            break
    if not results:
        print(f"[polygon] No valid articles for {ticker}")
    return results

# ------------------ Master Function ------------------ #
async def get_articles(ticker, limit=2):
    ticker = ticker.upper()
    sources = [try_newsapi, try_marketaux, try_alphavantage, try_yahoo_rss, try_bing_news, try_finviz, try_polygon]
    random.shuffle(sources)  # randomize source order as before
    for source_func in sources:
        print(f"Trying {source_func.__name__}...")
        articles = await source_func(ticker, limit)
        if articles:
            # Return a DataFrame of results (same structure as original)
            return pd.DataFrame(articles)[["date", "title", "text", "url", "ticker"]]
    print(f"❌ All sources failed for {ticker}.")
    return pd.DataFrame(columns=["date", "title", "text", "url", "ticker"])

# #Old Code
# import requests
# import random
# import time
# import pandas as pd
# from datetime import datetime, timedelta, timezone
# from newspaper import Article, Config
# import trafilatura
# from bs4 import BeautifulSoup
# import yfinance as yf
# import feedparser
# # ------------------ Setup ------------------ #
 

# NEWSAPI_KEYS = [
#     "fc6f5446794b4b96acbf024f76c6074f",
#     "5eb9b7a6221846f0b4dab6df45774bad",
#     "3351706eb2fe4588bff8f4cff706b9a5",
#     "d6b300c0ad224f7bb455b6be4f5406cb",
#     "aff32ce132f44912830e90c3e15d8a83"
# ]
# MARKETAUX_KEYS = ["vGc2ngli75vh0B0TLLZBYmLoP9j6JdlIO0xnYU1Z", 'l3kI5uLaJ7cmiYtzaqbmnTjf0RKzyxrqqkWPqzXF', '56zM77P4FypSsNSmVOU4T2XoBpeyzznVZufahSAt', "MF5YF01raQLHgQhrSznF3E5d1i2c0U9DcW0msbzP", "yNItEjExQVMy5qLicMx3w9IvSDCK7Zc6qtHMjUAz"]
# ALPHA_KEYS = ["2GOHOV1X36HZDPHS", "B3FTT2KZWV3EOOMM","GWVW30578EIUMAMT", "CDCQHAZTZBMRP4SN"]
# POLYGON_KEYS = ["PeJay0iXYAG_fFa8B275Gu0zCHAFFypw", "B_v9if1uHwRU08BnqNU2c9IGyy_ZXuQF"]
# # ------------------ Utilities, headers, lP's ------------------ #
# USER_AGENTS = [
#     # Chrome – Windows
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",    
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
#     "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
#     "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
#     "Mozilla/5.0 (iPad; CPU OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
#     "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36"
# ]

# def get_headers():
#     return {
#         "User-Agent": random.choice(USER_AGENTS),
#         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#         "Accept-Language": "en-US,en;q=0.9",
#         "Referer": "https://www.google.com/",
#         "Connection": "keep-alive",
#         "Sec-Fetch-Site": "same-origin",
#         "Sec-Fetch-Mode": "navigate",
#         "Sec-Fetch-Dest": "document",
#         "Upgrade-Insecure-Requests": "1"
#     }

# # ------------------ Utilities ------------------ #
# def extract_article_text_fallback(url, sleep_between=0.31, verbose=True):
#     try:
#         time.sleep(random.uniform(0.15, sleep_between)) #Random short delay before starting newspaper3k (helps avoid bans/throttling)
#         config = Config()
#         config.browser_user_agent = random.choice(USER_AGENTS) #Setup newspaper3k with a random user-agent
#         article = Article(url, config=config)
#         article.download()
#         article.parse()
#         #If successful and non-empty, return the result
#         if article.text.strip():
#             if verbose: print(f"[✔] newspaper3k succeeded for {url}")
#             return article.text.strip(), "newspaper3k"
#     except Exception as e:
#         if verbose: print(f"[newspaper3k error] {url}: {e}")

#     try:
#         time.sleep(random.uniform(0.15, sleep_between)) #Sleep before trying next method: trafilatura
#         downloaded = trafilatura.fetch_url(url) # trys fetching and extracting content using trafilatura
#         if downloaded:
#             result = trafilatura.extract(downloaded)
#             if result and result.strip():
#                 if verbose: print(f"[✔] trafilatura succeeded for {url}")
#                 return result.strip(), "trafilatura"
#     except Exception as e:
#         if verbose: print(f"[trafilatura error] {url}: {e}")

#     try:
#         time.sleep(random.uniform(0.15, sleep_between)) # Sleep before final method: BeautifulSoup fallback
#         r = requests.get(url, headers=get_headers(), timeout=3)  # Try making a raw GET request with a timeout limit (timout is only for request)
#         soup = BeautifulSoup(r.content, "html.parser")
#         paragraphs = soup.find_all("p")
#         text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 30)
#         if text.strip():
#             if verbose: print(f"[✔] BeautifulSoup succeeded for {url}")
#             return text.strip(), "beautifulsoup"
#     except requests.exceptions.Timeout:  #explicit timeout catch
#         if verbose: print(f"[BeautifulSoup timeout] {url}")
#     except Exception as e:
#         if verbose: print(f"[BeautifulSoup error] {url}: {e}")

#     if verbose: print(f"[✖] All fallback methods failed for {url}")
#     return "", "none"


# def format_date(raw_date, source):
#     try:
#         if source == "newsapi":
#             return datetime.fromisoformat(raw_date.replace("Z", "+00:00")).strftime("%Y-%m-%d")
#         elif source == "marketaux":
#             return datetime.fromisoformat(raw_date).strftime("%Y-%m-%d")
#         elif source == "alphavantage":
#             return datetime.strptime(raw_date, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")
#         elif source == "yfinance":
#             return raw_date.strftime("%Y-%m-%d")
#         elif source == "bing":
#             return datetime.now(timezone.utc).strftime("%Y-%m-%d")
#         elif source == "finviz":
#             parts = raw_date.split()
#             if len(parts) == 1:
#                 return datetime.now().strftime('%Y-%m-%d')
#             try:
#                 return datetime.strptime(parts[0], "%b-%d-%y").strftime('%Y-%m-%d')
#             except:
#                 return datetime.now().strftime('%Y-%m-%d')
#         return raw_date
#     except Exception as e:
#         print(f"[date parse error - {source}] {raw_date}: {e}")
#         return ""

# # ------------------ Sources ------------------ #
# def try_newsapi(ticker, limit):
#     key = random.choice(NEWSAPI_KEYS)
#     query = f"{ticker} stock news"  # Improved query string
#     url = f"https://newsapi.org/v2/everything?q={query}&apiKey={key}&language=en&pageSize={limit}"
    
#     r = requests.get(url, headers=get_headers())
#     if r.status_code != 200:
#         print(f"[newsapi error] Status: {r.status_code}")
#         return []
    
#     results = []
#     for a in r.json().get("articles", []):
#         if not a.get('url'):
#             continue
#         text, _ = extract_article_text_fallback(a['url'])
#         if not text:
#             continue
#         results.append({
#             "date": format_date(a['publishedAt'], "newsapi"),
#             "title": a['title'],
#             "text": text,
#             "url": a['url'],
#             "ticker": ticker
#         })
#         if len(results) >= limit:
#             break
#     return results


# def try_marketaux(ticker, limit):
#     key = random.choice(MARKETAUX_KEYS)  # Randomized for safety
#     url = "https://api.marketaux.com/v1/news/all"
#     params = {
#         "api_token": key,
#         "symbols": ticker,
#         "language": "en",
#         "limit": limit * 2,
#         "published_after": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
#     }
#     r = requests.get(url, params=params, headers=get_headers())
#     print(f"[marketaux] Status: {r.status_code}")

#     if r.status_code != 200:
#         print(f"[marketaux error] Failed with status {r.status_code}")
#         return []

#     data = r.json()
#     articles = data.get("data", [])
#     if not articles:
#         print(f"[marketaux] No articles found for {ticker}")
#         return []

#     results = []
#     for a in articles:
#         url = a.get('url')
#         if not url:
#             continue
#         text, _ = extract_article_text_fallback(url)
#         if not text:
#             continue

#         results.append({
#             "date": format_date(a['published_at'], "marketaux"),
#             "title": a['title'],
#             "text": text,
#             "url": url,
#             "ticker": ticker
#         })
#         if len(results) >= limit:
#             break
#     return results





# def try_alphavantage(ticker, limit):
#     key = random.choice(ALPHA_KEYS)
#     r = requests.get("https://www.alphavantage.co/query", params={
#         "function": "NEWS_SENTIMENT",
#         "tickers": ticker,
#         "apikey": key,
#         "sort": "LATEST"
#     }, headers=get_headers())
#     if r.status_code != 200: return []
#     results = []
#     for a in r.json().get("feed", []):
#         if not a.get('url'): continue
#         text, _ = extract_article_text_fallback(a['url'])
#         if not text: continue
#         results.append({
#             "date": format_date(a['time_published'], "alphavantage"),
#             "title": a['title'],
#             "text": text,
#             "url": a['url'],
#             "ticker": ticker
#         })
#         if len(results) >= limit: break
#     return results



 

# def try_yahoo_rss(ticker, limit):
#     print(f"[yahoo_rss] Fetching RSS feed for {ticker}")
#     url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
#     try:
#         feed = feedparser.parse(url)  #
#     except Exception as e:
#         print(f"[yahoo_rss parse error] {e}")
#         return []

#     results = []
#     seen = set()

#     for entry in feed.entries:
#         article_url = entry.get("link")
#         title = entry.get("title", "").strip()
#         published = entry.get("published", "").strip()

#         norm_url = (article_url or "").strip().lower() 
#         if not norm_url or norm_url in seen or not title:
#             continue
#         seen.add(norm_url)

#         text, _ = extract_article_text_fallback(article_url)
#         if not text:
#             continue

#         results.append({
#             "date": format_date(published, "bing"),
#             "title": title,
#             "text": text,
#             "url": article_url,
#             "ticker": ticker
#         })

#         if len(results) >= limit:
#             break

#     if not results:
#         print(f"[yahoo_rss] No valid articles for {ticker}")
#     return results




# def try_bing_news(ticker, limit):
#     print(f"🔍 Scraping Bing News for {ticker}...")
#     url = "https://www.bing.com/news/search"
    
#     query = f"{ticker} stock news"  # More specific query
#     r = requests.get(url, params={
#         "q": query,
#         "form": "QBNH",
#         "setmkt": "en-US",
#         "setlang": "en-US",
#     }, headers=get_headers(), timeout=10)

#     results, seen = [], set()
#     soup = BeautifulSoup(r.text, "html.parser")

#     for tag in soup.find_all("a", {"class": "title"}):
#         href = tag.get("href")
#         title = tag.get_text(strip=True)
#         if not href or href in seen or not title:
#             continue
#         seen.add(href)
#         text, _ = extract_article_text_fallback(href)
#         if not text:
#             continue
#         results.append({
#             "date": format_date("", "bing"),
#             "title": title,
#             "text": text,
#             "url": href,
#             "ticker": ticker
#         })
#         if len(results) >= limit:
#             break
#     return results


# def try_finviz(ticker, limit):
#     print(f"📰 Scraping Finviz for {ticker}...")
#     url = f"https://finviz.com/quote.ashx?t={ticker}"
#     try:
#         r = requests.get(url, headers=get_headers(), timeout=10)
#         soup = BeautifulSoup(r.text, "lxml")
#         news_table = soup.find(id='news-table')
#     except Exception as e:
#         print(f"[finviz error] {e}")
#         return []

#     if not news_table:
#         return []

#     results, seen = [], set()
#     for row in news_table.find_all('tr'):
#         a_tag = row.a
#         td_tag = row.td
#         if a_tag and td_tag:
#             headline = a_tag.get_text(strip=True)
#             href = a_tag.get("href")

#             norm_href = (href or "").strip().lower()
#             if not norm_href or norm_href in seen or not headline:
#                 continue
#             seen.add(norm_href)

#             text, _ = extract_article_text_fallback(href)
#             if not text:
#                 continue

#             raw_date = td_tag.text.strip()
#             results.append({
#                 "date": format_date(raw_date, "finviz"),
#                 "title": headline,
#                 "text": text,
#                 "url": href,
#                 "ticker": ticker
#             })
#             if len(results) >= limit:
#                 break
#     return results

# def try_polygon(ticker, limit):
#     print(f"[polygon] Fetching news for {ticker}")
#     key = random.choice(POLYGON_KEYS)
#     url = f"https://api.polygon.io/v2/reference/news"
#     params = {
#         "ticker": ticker,
#         "limit": limit * 2,
#         "order": "desc",
#         "sort": "published_utc",
#         "apiKey": key
#     }

#     try:
#         r = requests.get(url, params=params, headers=get_headers(), timeout=10)
#         if r.status_code != 200:
#             print(f"[polygon error] Status code: {r.status_code}")
#             return []
#     except Exception as e:
#         print(f"[polygon exception] {e}")
#         return []

#     data = r.json().get("results", [])
#     results = []

#     for article in data:
#         url = article.get("article_url")
#         title = article.get("title")
#         published = article.get("published_utc")

#         if not url or not title:
#             continue

#         text, _ = extract_article_text_fallback(url)
#         if not text:
#             continue

#         results.append({
#             "date": format_date(published, "marketaux"),
#             "title": title,
#             "text": text,
#             "url": url,
#             "ticker": ticker
#         })

#         if len(results) >= limit:
#             break

#     if not results:
#         print(f"[polygon] No valid articles for {ticker}")
#     return results





# # ------------------ Master Function ------------------ #
# def get_articles(ticker, limit=2):
#     ticker = ticker.upper()
    
#     sources = [try_newsapi, try_marketaux, try_alphavantage, try_yahoo_rss, try_bing_news, try_finviz, try_polygon]
#     random.shuffle(sources)
#     for source_func in sources:
#         print(f"Trying {source_func.__name__}...")
#         articles = source_func(ticker, limit)
#         if articles:
#             return pd.DataFrame(articles)[["date", "title", "text", "url", "ticker"]]
#     print(f"❌ All sources failed for {ticker}.")
#     return pd.DataFrame(columns=["date", "title", "text", "url", "ticker"])
