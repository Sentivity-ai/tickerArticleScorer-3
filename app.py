import os
import re
import datetime
import requests
import torch
import numpy as np
import yfinance as yf

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from scipy.special import softmax

# === Model Setup ===
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# === Preprocessing ===
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Sentiment Scoring ===
def predict_sentiment(text: str) -> float:
    if not text:
        return 0.0
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded_input)
    scores = softmax(output[0][0].detach().numpy())
    
    negative_score = 0.0
    for idx, label in config.id2label.items():
        if label.lower() == 'negative':
            negative_score = scores[idx]

    return (1 - negative_score) * 100  # Higher is better (less negative)

# === Article Fetching ===
ALLOWED_TICKERS = {"AAPL", "GOOG", "AMZN", "NVDA", "META", "SPY"}
sentiment_cache = {ticker: {"article": None, "sentiment": None, "timestamp": None} for ticker in ALLOWED_TICKERS}

def fetch_latest_article(ticker: str) -> str:
    try:
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news or []

        for item in news_items:
            if not item:
                continue
            url = item.get("link")
            if not url:
                continue
            title = item.get("title", "")
            summary = item.get("summary", "")
            return f"{title} {summary}".strip()

    except Exception as e:
        print(f"[ERROR] Failed to fetch article for {ticker}: {e}")
    
    return ""

def is_cache_valid(ts, max_minutes=30):
    return ts and (datetime.datetime.utcnow() - ts).total_seconds() < max_minutes * 60

# === FastAPI App ===
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")

@app.get("/api/ticker")
def analyze_ticker(ticker: str):
    ticker = ticker.upper()

    if ticker not in ALLOWED_TICKERS:
        return JSONResponse(status_code=400, content={
            "error": f"Unsupported ticker '{ticker}'. Try: {', '.join(sorted(ALLOWED_TICKERS))}"
        })

    tickers_to_check = [ticker]
    if ticker != "SPY":
        tickers_to_check.append("SPY")  # Always compare to SPY

    results = []

    for tk in tickers_to_check:
        cache = sentiment_cache[tk]

        if is_cache_valid(cache["timestamp"]) and cache["article"]:
            results.append({
                "ticker": tk,
                "article": cache["article"],
                "sentiment": cache["sentiment"]
            })
            continue

        article = fetch_latest_article(tk)
        if not article:
            results.append({
                "ticker": tk,
                "article": "No recent news available.",
                "sentiment": 0.0
            })
            continue

        sentiment = predict_sentiment(article)
        sentiment_cache[tk] = {
            "article": article,
            "sentiment": float(sentiment),
            "timestamp": datetime.datetime.utcnow()
        }

        results.append({
            "ticker": tk,
            "article": article,
            "sentiment": float(sentiment)
        })

    return results
