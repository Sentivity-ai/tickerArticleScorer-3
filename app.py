import os
import re
import datetime
import requests
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import uvicorn

# Load model assets
MODEL_PATH = "score_predictor.pth"
VECTORIZER_PATH = "AutoVectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("score_predictor.pth not found")

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/xlm-twitter-politics-sentiment")
vectorizer = joblib.load(VECTORIZER_PATH)

class ScorePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

score_model = ScorePredictor(tokenizer.vocab_size)
score_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
score_model.eval()

# Utility functions
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text: str) -> float:
    if not text:
        return 0.0
    tokens = tokenizer(text.split(), return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = score_model(tokens["input_ids"], tokens["attention_mask"])
        return round(output[0].item() * 100, 2)  # percent score for UI

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

# FastAPI setup
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

    clean = preprocess(article)
    sentiment = predict_sentiment(clean)

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
