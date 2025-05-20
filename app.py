import gradio as gr
import requests
import torch
import torch.nn as nn
import re
import datetime
from transformers import AutoTokenizer
import joblib

# Load tokenizer and sentiment model
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

vectorizer = joblib.load("AutoVectorizer.pkl")
classifier = joblib.load("AutoClassifier.pkl")

# Global cache
sentiment_cache = {}

# FastAPI app
app = FastAPI()

# CORS for frontend or JS usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# predicts sentiment
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
    input_ids, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]
    with torch.no_grad():
        score = score_model(input_ids, attention_mask)[0].item()
    return score

# uses Polygon API to fetch article
def fetch_articles(ticker):
    POLYGON_API_KEY = "cMCv7jipVvV4qLBikgzllNmW_isiODRR"
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=1&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            article = data["results"][0]
            title = article.get("title", "")
            description = article.get("description", "")
            return [title + " " + description]
        else:
            return [f"No news articles found for {ticker}."]
    except Exception as e:
        return [f"Error fetching articles for {ticker}: {str(e)}"]

# allowed tickers
ALLOWED_TICKERS = {"AAPL", "GOOG", "AMZN", "NVDA", "META"}

# initialize cache
sentiment_cache = {ticker: {"article": None, "sentiment": None, "timestamp": None} for ticker in ALLOWED_TICKERS}

# checks if cache is valid 
def is_cache_valid(cached_time, max_age_minutes=30):
    if cached_time is None:
        return False
    now = datetime.datetime.utcnow()
    age = now - cached_time
    return age.total_seconds() < max_age_minutes * 60

# analyzes the tikcers
def analyze_ticker(ticker):
    ticker = ticker.upper()
    if ticker not in ALLOWED_TICKERS:
        return [{
            "article": f"Sorry, '{ticker}' is not supported. Please choose one of: {', '.join(sorted(ALLOWED_TICKERS))}.",
            "sentiment": 0.0
        }]

    cache_entry = sentiment_cache[ticker]

    # if cache is valid and article exists
    if is_cache_valid(cache_entry["timestamp"]) and cache_entry["article"] is not None:

        return [{
            "article": cache_entry["article"],
            "sentiment": cache_entry["sentiment"]
        }]

    # fetch new article and update cache if cache is invalid
    articles = fetch_articles(ticker)
    if not articles:
        return [{"article": "No articles found.", "sentiment": 0.0}]

    article = articles[0]  

    clean_text = preprocess_text(article)
    sentiment = predict_sentiment(clean_text)

    # update cache with current time
    sentiment_cache[ticker] = {
        "article": article,
        "sentiment": sentiment,
        "timestamp": datetime.datetime.utcnow()
    }

    return [{
        "article": article,
        "sentiment": sentiment
    }]

# display's sentiment
def display_sentiment(ticker):
    results = analyze_ticker(ticker)
    html_output = "<h2>Sentiment Analysis</h2><ul>"
    for r in results:
        html_output += f"<li><b>{r['article']}</b><br>Score: {r['sentiment']:.2f}</li>"
    html_output += "</ul>"
    return html_output

# search feature
with gr.Blocks() as demo:
    gr.Markdown("# Ticker Sentiment Analysis")
    ticker_input = gr.Textbox(label="Enter Ticker Symbol (e.g., AAPL)")
    output_html = gr.HTML()
    analyze_btn = gr.Button("Analyze")
    analyze_btn.click(fn=display_sentiment, inputs=[ticker_input], outputs=[output_html])

demo.launch()
