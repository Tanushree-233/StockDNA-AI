import pandas as pd
from nlp.nlp_cleaner import clean_news_data
from nlp.sentiment import add_sentiment
from nlp.topic_classifier import add_topics
from nlp.entity_extractor import add_entities
from nlp.event_classifier import add_events
from nlp.event_impact import add_event_impact


def run_nlp_pipeline(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        raise ValueError(
            "Cannot process an empty news dataset."
        )

    data = clean_news_data(df)

    print(
        f"[NLP] Cleaned {len(data)} news records."
    )

    data = add_sentiment(data)

    print(
        "[NLP] Sentiment analysis complete."
    )

    data = add_topics(data)

    print(
        "[NLP] Topic classification complete."
    )

    data = add_entities(data)

    print(
        "[NLP] Entity extraction complete."
    )

    data = add_events(data)

    print(
        "[NLP] Event detection complete."
    )

    data = add_event_impact(data)

    print(
        "[NLP] Event impact analysis complete."
    )

    return data