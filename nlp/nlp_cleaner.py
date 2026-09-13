import re
import pandas as pd


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?&'%-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_text(text):
    text = str(text).lower()

    return re.findall(
        r"\b[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*\b",
        text
    )


def clean_news_data(df):
    df = df.copy()

    if "Headline" not in df.columns:
        raise ValueError("Headline column is required.")

    df["Clean_Text"] = df["Headline"].apply(clean_text)

    df = df[df["Clean_Text"].str.len() > 0]

    df["Tokens"] = df["Clean_Text"].apply(
        lambda text: " ".join(tokenize_text(text))
    )

    df["Token_Count"] = df["Clean_Text"].apply(
        lambda text: len(tokenize_text(text))
    )

    return df.reset_index(drop=True)