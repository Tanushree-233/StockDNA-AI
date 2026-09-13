import re
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "ProsusAI/finbert"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_model.eval()

LABELS = {
    0: "Positive",
    1: "Negative",
    2: "Neutral"
}


NEGATIVE_PHRASES = {
    "miss estimates",
    "missed estimates",
    "losses widen",
    "loss widened",
    "net loss",
    "shares slide",
    "shares fall",
    "stock drops",
    "stock falls",
    "falls sharply",
    "drop sharply",
    "weaker outlook",
    "cut targets",
    "analysts cut",
    "job loss",
    "deflate value",
    "revenue decline",
    "profit decline",
    "losses increase",
    "worst week"
}


POSITIVE_PHRASES = {
    "beat estimates",
    "beats estimates",
    "exceeded estimates",
    "revenue surge",
    "revenue growth",
    "strong revenue",
    "profit growth",
    "strong growth",
    "record growth",
    "revenue jump",
    "revenue jumps",
    "more deals",
    "global expansion",
    "cost savings"
}


def finbert_sentiment(text):

    inputs = _tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    sentiment = LABELS[predicted_index]
    confidence = probabilities[predicted_index].item()

    return sentiment, round(confidence, 3)


def apply_financial_context(text, sentiment, score):

    text = str(text).lower()

    negative_matches = [
        phrase
        for phrase in NEGATIVE_PHRASES
        if re.search(r"\b" + re.escape(phrase) + r"\b", text)
    ]

    positive_matches = [
        phrase
        for phrase in POSITIVE_PHRASES
        if re.search(r"\b" + re.escape(phrase) + r"\b", text)
    ]

    if negative_matches and not positive_matches:
        return "Negative", max(score, 0.80)

    if positive_matches and not negative_matches:
        return "Positive", max(score, 0.80)

    return sentiment, score


def analyze_sentiment(text):

    sentiment, score = finbert_sentiment(text)

    sentiment, score = apply_financial_context(
        text,
        sentiment,
        score
    )

    return sentiment, score


def add_sentiment(df):

    df = df.copy()

    results = df["Clean_Text"].apply(
        analyze_sentiment
    )

    df["Sentiment"] = results.apply(
        lambda x: x[0]
    )

    df["Sentiment_Score"] = results.apply(
        lambda x: x[1]
    )

    df["Sentiment_Model"] = "FinBERT + Financial Context"

    return df