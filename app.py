import gradio as gr
import requests
import torch
import torch.nn as nn
import re
import datetime
import yfinance as yf
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoConfig
from scipy.special import softmax
from newspaper import Article

# Load tokenizer and sentiment model
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)


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

# Load trained score predictor model
score_model = ScorePredictor(tokenizer.vocab_size)
score_model.load_state_dict(torch.load("score_predictor.pth"))
score_model.eval()

# preprocesses text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d{1,2}:\d{2}', '', text)  
    text = re.sub(r'speaker\s+[a-z]', '', text)  
    text = re.sub(r'\b[a-z]{2,20}\s+howley\b', '', text)  
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# predicts sentiment
def predict_sentiment(text):
    if not text:
        return 0.0
    # encoded_input = tokenizer(
    #     text.split(),
    #     return_tensors='pt',
    #     padding=True,
    #     truncation=True,
    #     max_length=512
    # )
    # input_ids, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]
    # with torch.no_grad():
    #     score = score_model(input_ids, attention_mask)[0].item()


    # k = 20   
    # midpoint = 0.7 

    # scaled_score = 1 / (1 + np.exp(-k * (score - midpoint)))
    # final_output = scaled_score * 100 

    # return 1-final_output
    text = preprocess_text(text)
    # encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = tokenizer(
    text,
    return_tensors='pt',
    truncation=True,
    padding=True,
    max_length=512
    )
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    negative_id = -1
    for idx, label in config.id2label.items():
        if label.lower() == 'negative':
            negative_id = idx
            negative_score = scores[negative_id]
    
    return (1-(float(negative_score)))*100


# extracts article text
def extract_article_text(url: str):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title or "",
            "text": article.text or "",
            "publish_date": article.publish_date,  
            "url": url
        }
    except Exception as e:
        print(f"[ERROR] newspaper3k failed for URL {url}: {e}")
        return None

# fetch article based on ticker
def fetch_article_for_ticker(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    news_items = ticker_obj.news or []

    if not news_items:
        return None

    for item in news_items:
        if item is None:
            continue
        # tries both fields where yfinance might store a URL
        url = item.get("link") or item.get("content", {}).get("clickThroughUrl", {}).get("url")
        if not url:
            continue

        parsed = extract_article_text(url)
        if parsed:
            return parsed

    return None

# initialize cache
sentiment_cache = {}

# checks if cache is valid 
def is_cache_valid(cached_time, max_age_minutes=10):
    if cached_time is None:
        return False
    now = datetime.datetime.utcnow()
    age = now - cached_time
    return age.total_seconds() < max_age_minutes * 60

# analyzes the tikcers
def analyze_ticker(user_ticker: str):
    user_ticker = user_ticker.upper().strip()
    tickers_to_check = [user_ticker, "SPY"] if user_ticker != "SPY" else ["SPY"]
    results = []

    for tk in tickers_to_check:
        cached = sentiment_cache.get(tk)
        if cached and is_cache_valid(cached.get("timestamp")):
            # reuse cached entry
            results.append({
                "ticker": tk,
                "article_blurb": cached["article_blurb"],
                "sentiment": cached["sentiment"],
                "timestamp": cached["timestamp"],
            })
            continue

        # fetch fresh article via yfinance + newspaper3k
        article_data = fetch_article_for_ticker(tk)
        if not article_data:
            blurb = f"No news articles found for {tk}."
            sentiment_score = None
        else:
            full_text = article_data["title"] + " " + article_data["text"]
            sentiment_score = predict_sentiment(full_text)

            cleaned_text = preprocess_text(article_data["text"])
            short_blurb = cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text
            blurb = f"{article_data['title']}\n\n{short_blurb}"

        timestamp = datetime.datetime.utcnow()
        cache_entry = {
            "article_blurb": blurb,
            "sentiment": sentiment_score,
            "timestamp": timestamp
        }
        sentiment_cache[tk] = cache_entry

        results.append({
            "ticker": tk,
            "article_blurb": blurb,
            "sentiment": sentiment_score,
            "timestamp": timestamp
        })

    # has user_ticker appears first in the list
    results.sort(key=lambda x: 0 if x["ticker"] == user_ticker else 1)
    return results


def display_sentiment(results):
    html = "<h2>Sentiment Analysis</h2><ul>"
    for r in results:
        ts_str = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        score_display = (
            f"{r['sentiment']:.2f}"
            if r['sentiment'] is not None else
            "—"
        )
        html += (
            f"<li><b>{r['ticker']}</b> &nbsp;({ts_str})<br>"
            f"{r['article_blurb']}<br>"
            f"<i>Sentiment score:</i> {score_display}</li>"
        )
    html += "</ul>"
    return html


    

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
