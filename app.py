import os
import re
import datetime
import requests
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from scipy.special import softmax

# Load model & tokenizer
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Preprocess
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction
def predict_sentiment(text: str) -> float:
    if not text:
        return 0.0
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = softmax(output[0][0].detach().numpy())
    negative_score = 0.0
    for idx, label in config.id2label.items():
        if label.lower() == 'negative':
            negative_score = scores[idx]
    return (1 - negative_score) * 100

# Config
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "cMCv7jipVvV4qLBikgzllNmW_isiODRR")
ALLOWED_TICKERS = {"AAPL", "GOOG", "AMZN", "NVDA", "META"}
sentiment_cache = {ticker: {"article": None, "sentiment": None, "timestamp": None} for ticker in ALLOWED_TICKERS}

def fetch_latest_article(ticker: str) -> str:
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=1&apiKey={POLYGON_API_KEY}"
        res = requests.get(url).json()
        if res.get("results"):
            item = res["results"][0]
            return (item.get("title", "") + " " + item.get("description", "")).strip()
    except Exception as e:
        print(f"Error fetching news: {e}")
    return ""

def is_cache_valid(ts, max_minutes=30):
    return ts and (datetime.datetime.utcnow() - ts).total_seconds() < max_minutes * 60

# FastAPI app
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

    cache = sentiment_cache[ticker]
    if is_cache_valid(cache["timestamp"]) and cache["article"]:
        return {
            "ticker": ticker,
            "article": cache["article"],
            "sentiment": cache["sentiment"]
        }

    article = fetch_latest_article(ticker)
    if not article:
        return {"ticker": ticker, "article": "No recent news available.", "sentiment": 0.0}

    sentiment = predict_sentiment(article)
    sentiment_cache[ticker] = {
        "article": article,
        "sentiment": sentiment,
        "timestamp": datetime.datetime.utcnow()
    }

    return {
        "ticker": ticker,
        "article": article,
        "sentiment": sentiment
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
