# DATE MODIFIED 6/30/
import os
import re
import datetime
import numpy as np
import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from apscheduler.schedulers.background import BackgroundScheduler
from scraper import get_articles

# ------------------ Load ML Models ------------------ #
with open('lgbm_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as l:
    vectorizer = pickle.load(l)

# ------------------ Config ------------------ #
CACHE_TTL_MINUTES = 16
sentiment_cache = {}

 
PREDEFINED_TICKERS = [
    #DOW 30
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
    #ETFS
    "SPY"
]

# ------------------ Preprocessing & Scoring ------------------ #
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_sentiment(text: str) -> float:
    if not text:
        return 0.0
    txt = preprocess_text(text)
    c = analyzer.polarity_scores(txt)["compound"]
    return round(((c + 1) / 2) * 100, 2)

def is_cache_valid(cached_time: datetime.datetime) -> bool:
    if cached_time is None:
        return False
    return (datetime.datetime.utcnow() - cached_time).total_seconds() < CACHE_TTL_MINUTES * 60

# ------------------ Background Cache Refresh ------------------ #
def refresh_cache():
    now = datetime.datetime.utcnow()

    # Refresh only predefined tickers + recently queried ones
    tickers = set(PREDEFINED_TICKERS)
    for sym, data in sentiment_cache.items():
        if (now - data["timestamp"]).total_seconds() < 3 * 3600:  # last 3 hours
            tickers.add(sym)

    for sym in tickers:
        df = get_articles(sym, limit=2)
        if df.empty:
            blurb, score = f"No news articles found for {sym}.", None
        else:
            scores, links = [], []
            for _, row in df.iterrows():
                s = predict_sentiment(row["title"] + " " + row["text"][:15])
                scores.append(s)
                links.append(f"<a href='{row['url']}' target='_blank'>{row['title']}</a>")
            score = float(np.mean(scores)) if scores else None
            blurb = "<br>".join(links)

        sentiment_cache[sym] = {
            "article_blurb": blurb,
            "sentiment": score,
            "timestamp": now
        }

# ------------------ FastAPI Setup ------------------ #
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_cache, "interval", minutes=8, next_run_time=datetime.datetime.utcnow())
    scheduler.start()

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")

@app.get("/api/sentiment")
async def get_sentiment(ticker: str):
    user_ticker = ticker.upper().strip()
    tickers = [user_ticker] if user_ticker == "SPY" else [user_ticker, "SPY"]
    results = []

    for sym in tickers:
        cached = sentiment_cache.get(sym)
        if cached and is_cache_valid(cached["timestamp"]):
            results.append({
                "ticker": sym,
                "article_blurb": cached["article_blurb"],
                "sentiment": cached["sentiment"],
                "timestamp": cached["timestamp"].isoformat() + "Z"
            })
            continue

        # Cache miss: fetch and score
        df = get_articles(sym, limit=2)
        if df.empty:
            blurb, score = f"No news articles found for {sym}.", None
        else:
            scores, links = [], []
            for _, row in df.iterrows():
                s = predict_sentiment(row["title"] + " " + row["text"][:15])
                scores.append(s)
                links.append(f"<a href='{row['url']}' target='_blank'>{row['title']}</a>")
            score = float(np.mean(scores)) if scores else None
            blurb = "<br>".join(links)

        ts = datetime.datetime.utcnow()
        sentiment_cache[sym] = {
            "article_blurb": blurb,
            "sentiment": score,
            "timestamp": ts
        }

        results.append({
            "ticker": sym,
            "article_blurb": blurb,
            "sentiment": score,
            "timestamp": ts.isoformat() + "Z"
        })

    results.sort(key=lambda x: 0 if x["ticker"] == user_ticker else 1)
    return JSONResponse(content=results)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

# # Date Modified: 6/22/2025
# import os
# import re
# import datetime
# import torch
# import numpy as np
# import uvicorn
# import pickle
# import pandas as pd
# from fastapi import FastAPI
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from scraper import get_articles  

# # ------------------ Load Models ------------------ #
# with open('lgbm_model.pkl', 'rb') as f:
#     lgbm_model = pickle.load(f)
# with open('tfidf_vectorizer.pkl', 'rb') as l:
#     vectorizer = pickle.load(l)

# analyzer = SentimentIntensityAnalyzer()

# # ------------------ Preprocessing ------------------ #
# def preprocess_text(text):
#     text = str(text)
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # ------------------ Scoring ------------------ #
# def predict_sentiment(text: str) -> float:
#     if not text:
#         return 0.0
#     text = preprocess_text(text)
#     scores = analyzer.polarity_scores(text)
#     compound = scores["compound"]
#     normalized = (compound + 1) / 2  # scale [-1,1] to [0,1]
#     return round(normalized * 100, 2)  # return as percentage

# # ------------------ Caching ------------------ #
# sentiment_cache = {}

# def is_cache_valid(cached_time, max_age_minutes=10):
#     if cached_time is None:
#         return False
#     now = datetime.datetime.utcnow()
#     age = now - cached_time
#     return age.total_seconds() < max_age_minutes * 60

# # ------------------ FastAPI Setup ------------------ #
# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/", include_in_schema=False)
# async def root():
#     return FileResponse("static/index.html")

# @app.get("/api/sentiment")
# async def get_sentiment(ticker: str):
#     user_ticker = ticker.upper().strip()
#     tickers_to_check = [user_ticker, "SPY"] if user_ticker != "SPY" else ["SPY"]
#     results = []

#     for tk in tickers_to_check:
#         cached = sentiment_cache.get(tk)
#         if cached and is_cache_valid(cached.get("timestamp")):
#             results.append({
#                 "ticker": tk,
#                 "article_blurb": cached["article_blurb"],
#                 "sentiment": cached["sentiment"],
#                 "timestamp": cached["timestamp"].isoformat() + "Z",
#             })
#             continue

#         df = get_articles(tk, limit=2)
#         if df.empty:
#             blurb = f"No news articles found for {tk}."
#             sentiment_score = None
#         else:
#             sentiment_scores = []
#             links_html = []

#             for _, row in df.iterrows():
#                 combined_text = row["title"] + " " + row["text"][:15]
#                 score = predict_sentiment(combined_text)
#                 sentiment_scores.append(score)

#                 link = f"<a href='{row['url']}' target='_blank'>{row['title']}</a>"
#                 links_html.append(link)

#             sentiment_score = float(np.mean(sentiment_scores)) if sentiment_scores else None
#             blurb = "<br>".join(links_html)

#         timestamp = datetime.datetime.utcnow()
#         sentiment_cache[tk] = {
#             "article_blurb": blurb,
#             "sentiment": sentiment_score,
#             "timestamp": timestamp
#         }

#         results.append({
#             "ticker": tk,
#             "article_blurb": blurb,
#             "sentiment": sentiment_score,
#             "timestamp": timestamp.isoformat() + "Z"
#         })

#     results.sort(key=lambda x: 0 if x["ticker"] == user_ticker else 1)
#     return JSONResponse(content=results)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("app:app", host="0.0.0.0", port=port)



 
