ArticleTickerScore: News-Based Ticker Sentiment Tool

A tool designed to measure short-term sentiment around major stock tickers using recent financial news headlines. It uses a combination of live article fetching, text preprocessing, and a PyTorch-based neural model to return a normalized sentiment score.


Goals

- Allow users to input a stock ticker and retrieve the most recent news article associated with it (via Polygon.io).
- Clean and tokenize the article for interpretation.
- Run the processed text through a trained LSTM sentiment model.
- Normalize and display the result in an interpretable format.
- Cache results for efficiency and refresh scores every 30 minutes.


Requirements

- gradio
- torch
- requests
- transformers
- datetime
- re
- os


Model Components

ScorePredictor class: A PyTorch-based LSTM classifier for sentiment scoring. It includes:
- An embedding layer (based on vocab size)
- A hidden LSTM layer for sequential understanding
- A linear + sigmoid output layer for binary-style scoring (normalized afterward)

AutoVectorizer: A trained vectorizer model that transforms input strings into vectors, capturing the strings’ textual features. The vector form can be interpreted by the AutoClassifier.

AutoClassifier: A binary classification model that labels vectorized Reddit posts as either sociopolitical (1) or not (0). It is used to filter out any irrelevant posts from the data set.



Main Script

1. Input Validation

- Converts the ticker to uppercase.
- Checks if it’s among the predefined tickers (AAPL, GOOG, AMZN, META, NVDA).
- If invalid, returns a friendly message and a default score.

2. Caching

Uses a global cache (sentiment_cache) to store:

- Last article
- Last sentiment score
- Timestamp

Uses is_cache_valid to determine if data is stale (older than 30 minutes).

3. Article Fetching

Uses Polygon.io’s /v2/reference/news API to fetch the most recent article for the ticker.

Extracts the title + description into a single string for model input.

4. Preprocessing

Cleans the article text using regex:

5. Sentiment Scoring

Tokenizes the cleaned text using the same tokenizer the model was trained with (cardiffnlp/xlm-twitter-politics-sentiment).

Passes the tokens into the ScorePredictor model.

Applies a custom normalization from [0.3, 0.9] → [0.0, 1.0].

6. Output

Returns a dictionary containing:
"article" – full text of the news snippet
"sentiment" – normalized score between 0.0 and 1.0


Helper Functions:

fetch_articles(ticker): Pulls a single article for the ticker via Polygon API.
preprocess_text(text):  Cleans and tokenizes the article text.
predict_sentiment(text): Runs the cleaned text through the LSTM model and returns a normalized sentiment score.
is_cache_valid(timestamp): Checks if cached data is less than 30 minutes old
analyze_ticker(ticker): Full logic for validating, caching, fetching, scoring, and returning sentiment results.
display_sentiment(ticker): Converts sentiment results into HTML format for rendering.


End Result

A web app that allows you to display the predicted sentiment of five major tickers: AAPL, GOOG, AMZN, NVDA, META.

