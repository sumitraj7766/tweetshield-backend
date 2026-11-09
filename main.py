
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords", quiet=True)

# Load trained model and vectorizer
model = joblib.load("twitter_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing setup (same as original app.py)
stop_words = set(stopwords.words("english"))

important_words = {
    "no", "nor", "not", "never",
    "don't", "doesn't", "didn't",
    "can't", "couldn't", "won't",
    "wouldn't", "shouldn't"
}

stop_words -= important_words

stemmer = PorterStemmer()

THRESHOLD = 0.65

app = FastAPI(title="TweetShield AI API")


def preprocess_tweet(tweet):
    text = tweet.lower()

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    text = re.sub(r"[^a-zA-Z0-9\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return ' '.join(tokens)


class TweetRequest(BaseModel):
    tweet: str


@app.get("/")
def home():
    return {"message": "TweetShield AI API is running"}


@app.post("/predict")
def predict(request: TweetRequest):
    tweet = request.tweet.strip()

    if not tweet:
        return {"error": "Tweet cannot be empty"}

    processed_text = preprocess_tweet(tweet)
    vector = tfidf.transform([processed_text])

    probability = float(model.predict_proba(vector)[0][1])
    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "label": (
            "Potentially Racist / Sexist"
            if prediction == 1
            else "Non-Racist / Non-Sexist"
        ),
        "confidence": round(probability * 100, 2),
        "threshold": THRESHOLD
    }
