# Date Modified: 6/22/2025
import os
import re
import datetime
import torch
import numpy as np
import uvicorn
import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scraper import get_articles  # ✅ your updated scraper

# ------------------ Load Models ------------------ #
with open('lgbm_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as l:
    vectorizer = pickle.load(l)

analyzer = SentimentIntensityAnalyzer()

# ------------------ Preprocessing ------------------ #
def preprocess_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------ Scoring ------------------ #
def predict_sentiment(text: str) -> float:
    if not text:
        return 0.0
    text = preprocess_text(text)
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    normalized = (compound + 1) / 2  # scale [-1,1] to [0,1]
    return round(normalized * 100, 2)  # return as percentage

# ------------------ Caching ------------------ #
sentiment_cache = {}

def is_cache_valid(cached_time, max_age_minutes=10):
    if cached_time is None:
        return False
    now = datetime.datetime.utcnow()
    age = now - cached_time
    return age.total_seconds() < max_age_minutes * 60

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

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")

@app.get("/api/sentiment")
async def get_sentiment(ticker: str):
    user_ticker = ticker.upper().strip()
    tickers_to_check = [user_ticker, "SPY"] if user_ticker != "SPY" else ["SPY"]
    results = []

    for tk in tickers_to_check:
        cached = sentiment_cache.get(tk)
        if cached and is_cache_valid(cached.get("timestamp")):
            results.append({
                "ticker": tk,
                "article_blurb": cached["article_blurb"],
                "sentiment": cached["sentiment"],
                "timestamp": cached["timestamp"].isoformat() + "Z",
            })
            continue

        df = get_articles(tk, limit=2)
        if df.empty:
            blurb = f"No news articles found for {tk}."
            sentiment_score = None
        else:
            sentiment_scores = []
            links_html = []

            for _, row in df.iterrows():
                combined_text = row["title"] + " " + row["text"][:15]
                score = predict_sentiment(combined_text)
                sentiment_scores.append(score)

                link = f"<a href='{row['url']}' target='_blank'>{row['title']}</a>"
                links_html.append(link)

            sentiment_score = float(np.mean(sentiment_scores)) if sentiment_scores else None
            blurb = "<br>".join(links_html)

        timestamp = datetime.datetime.utcnow()
        sentiment_cache[tk] = {
            "article_blurb": blurb,
            "sentiment": sentiment_score,
            "timestamp": timestamp
        }

        results.append({
            "ticker": tk,
            "article_blurb": blurb,
            "sentiment": sentiment_score,
            "timestamp": timestamp.isoformat() + "Z"
        })

    results.sort(key=lambda x: 0 if x["ticker"] == user_ticker else 1)
    return JSONResponse(content=results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

# # Date Modified: 6/18/2025
# import os
# import re
# import datetime
# import torch
# import numpy as np
# import uvicorn
# from fastapi import FastAPI
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# from scipy.special import softmax
# from scraper import get_articles  # ✅ Make sure scraper.py is included

# # ------------------ Load Models ------------------ #
# MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# # Optional: Load custom LSTM model if exists
# class ScorePredictor(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
#         super(ScorePredictor, self).__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = torch.nn.Linear(hidden_dim, output_dim)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, input_ids, attention_mask):
#         embedded = self.embedding(input_ids)
#         lstm_out, _ = self.lstm(embedded)
#         final_hidden_state = lstm_out[:, -1, :]
#         output = self.fc(final_hidden_state)
#         return self.sigmoid(output)

# USE_LSTM = os.path.exists("score_predictor.pth")
# score_model = None
# if USE_LSTM:
#     score_model = ScorePredictor(tokenizer.vocab_size)
#     score_model.load_state_dict(torch.load("score_predictor.pth"))
#     score_model.eval()

# # ------------------ Preprocessing ------------------ #
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"\d{1,2}:\d{2}", "", text)
#     text = re.sub(r"speaker\s+[a-z]", "", text)
#     text = re.sub(r"\b[a-z]{2,20}\s+howley\b", "", text)
#     text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# # ------------------ Scoring ------------------ #
# def predict_sentiment(text: str) -> float:
#     if not text:
#         return 0.0
#     text = preprocess_text(text)
#     encoded_input = tokenizer(
#         text,
#         return_tensors='pt',
#         truncation=True,
#         padding=True,
#         max_length=512
#     )
#     output = model(**encoded_input)
#     scores = output[0][0].detach().numpy()
#     for idx, label in config.id2label.items():
#         if label.lower() == 'negative':
#             negative_score = softmax(scores)[idx]
#             break
#     return (1 - float(negative_score)) * 100

# # ------------------ Caching ------------------ #
# sentiment_cache = {}

# def is_cache_valid(cached_time, max_age_minutes=30):
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
#                 combined_text = row["title"] + " " + row["text"]
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

 
