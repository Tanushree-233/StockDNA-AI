import re


TOPIC_KEYWORDS = {

    "Earnings": [
        "earnings",
        "quarterly results",
        "earnings call",
        "revenue",
        "profit",
        "eps",
        "quarter",
        "quarterly",
        "results",
        "estimates",
        "guidance",
        "outlook"
    ],

    "AI": [
        "artificial intelligence",
        "ai",
        "machine learning",
        "generative ai",
        "genai"
    ],

    "Technology": [
        "technology",
        "software",
        "cloud",
        "digital",
        "cybersecurity",
        "platform",
        "it services",
        "streaming"
    ],

    "Partnership": [
        "partnership",
        "partner",
        "partnered",
        "collaboration",
        "alliance",
        "joint venture"
    ],

    "Acquisition": [
        "acquisition",
        "acquire",
        "acquired",
        "merger",
        "takeover"
    ],

    "Product": [
        "product",
        "launch",
        "launched",
        "release",
        "released"
    ],

    "Regulation": [
        "regulation",
        "regulator",
        "government",
        "compliance",
        "policy",
        "ban",
        "visa",
        "h-1b"
    ],

    "Business": [
    "expansion",
    "expand",
    "expanded",
    "award",
    "exports",
    "export",
    "sale",
    "deal",
    "business"
],

    "Market": [
        "market",
        "stocks",
        "shares",
        "investors",
        "index",
        "sector",
        "stock",
        "valuation",
        "undervalued",
        "overvalued",
        "fair value"
    ]
}


def keyword_matches(text, keyword):

    text = str(text).lower()
    keyword = keyword.lower()

    pattern = r"\b" + re.escape(keyword) + r"\b"

    return bool(
        re.search(pattern, text)
    )


def classify_topic(text):

    text = str(text).lower()

    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword_matches(text, keyword):
                score += 1

        scores[topic] = score

    best_topic = max(
        scores,
        key=scores.get
    )

    if scores[best_topic] == 0:
        return "Other"

    return best_topic


def add_topics(df):

    df = df.copy()

    df["Topic"] = df["Clean_Text"].apply(
        classify_topic
    )

    return df