import os
import re
import datetime
import requests
import torch
import torch.nn as nn
import joblib
from transformers import AutoTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

MODEL_PATH = "score_predictor.pth"
MODEL_URL = "https://www.dropbox.com/scl/fi/atkis4c3x00so6qkdpyoc/score_predictor1.pth?rlkey=u73kyj4xby8aywb0hn67gfyw7&st=kn8x7sxn&dl=0"  # Replace with real link

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Dropbox...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

vectorizer = joblib.load("AutoVectorizer.pkl")
classifier = joblib.load("AutoClassifier.pkl")

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/xlm-twitter-politics-sentiment")

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
score_model.load_state_dict(torch.load(MODEL_PATH))
score_model.eval()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text):
    if not text:
        return 0.0
    encoded_input = tokenizer(
        text.split(),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    with torch.no_grad():
        score = score_model(input_ids, attention_mask)[0].item()
    return score

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "cMCv7jipVvV4qLBikgzllNmW_isiODRR")
ALLOWED_TICKERS = {"AAPL", "GOOG", "AMZN", "NVDA", "META"}
sentiment_cache = {ticker: {"article": None, "sentiment": None, "timestamp": None} for ticker in ALLOWED_TICKERS}

def fetch_articles(ticker):
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=1&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "results" in data and data["results"]:
            result = data["results"][0]
            return result.get("title", "") + " " + result.get("description", "")
    except Exception as e:
        print(f"Error fetching article for {ticker}: {e}")
    return ""

def is_cache_valid(ts, max_minutes=30):
    return ts and (datetime.datetime.utcnow() - ts).total_seconds() < max_minutes * 60

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            "sentiment": round(cache["sentiment"], 3)
        }

    article = fetch_articles(ticker)
    if not article:
        return {"ticker": ticker, "article": "No recent news available.", "sentiment": 0.0}

    clean_text = preprocess_text(article)
    sentiment = predict_sentiment(clean_text)

    sentiment_cache[ticker] = {
        "article": article,
        "sentiment": sentiment,
        "timestamp": datetime.datetime.utcnow()
    }

    return {
        "ticker": ticker,
        "article": article,
        "sentiment": round(sentiment, 3)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
