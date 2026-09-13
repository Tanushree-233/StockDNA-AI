import os
import pandas as pd

from nlp.nlp_pipeline import run_nlp_pipeline


INPUT_FILE = "data/raw/internal/news/news.csv"
OUTPUT_FILE = "data/processed/nlp_results.csv"


def main():

    print("=" * 50)
    print("STOCKDNA NLP - REAL NEWS TEST")
    print("=" * 50)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"News dataset not found: {INPUT_FILE}"
        )

    news = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded {len(news)} news records.")
    print("\nInput columns:")
    print(list(news.columns))

    results = run_nlp_pipeline(news)

    print("\n" + "=" * 50)
    print("NLP RESULTS")
    print("=" * 50)

    display_columns = [
        "Ticker",
        "Headline",
        "Sentiment",
        "Sentiment_Score",
        "Topic",
        "Entities",
        "Event",
        "Event_Impact"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    print(
        results[available_columns].to_string(
            index=False
        )
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(
        f"Total records processed: {len(results)}"
    )

    print("\nSentiment distribution:")
    print(
        results["Sentiment"].value_counts()
    )

    print("\nTopic distribution:")
    print(
        results["Topic"].value_counts()
    )

    print("\nEvent distribution:")
    print(
        results["Event"].value_counts()
    )

    print("\nEvent Impact distribution:")
    print(
        results["Event_Impact"].value_counts()
    )

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("=" * 50)

    print(
        f"\nResults saved to:\n{os.path.abspath(OUTPUT_FILE)}"
    )


if __name__ == "__main__":
    main()