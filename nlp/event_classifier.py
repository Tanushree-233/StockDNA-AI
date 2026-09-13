import re


EVENT_KEYWORDS = {

    "Earnings": [
        "earnings",
        "quarterly results",
        "earnings call",
        "revenue",
        "profit",
        "eps",
        "results",
        "miss estimates",
        "missed estimates",
        "beat estimates",
        "beats estimates"
    ],

    "Partnership": [
        "partnership",
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

    "Product Launch": [
        "product launch",
        "launched",
        "launch",
        "new product",
        "release",
        "released"
    ],

    "Expansion": [
        "expansion",
        "expand",
        "expanded",
        "new facility",
        "new office",
        "goes global",
        "global expansion",
        "takes global",
        "to india",
        "in india"
    ],

    "Contract": [
        "contract",
        "wins contract",
        "signed contract",
        "deal",
        "order",
        "signed agreement"
    ],

    "Regulatory": [
        "regulation",
        "regulator",
        "government proposes",
        "government",
        "fine",
        "penalty",
        "investigation",
        "lawsuit",
        "visa",
        "h-1b",
        "compliance",
        "policy",
        "ban"
    ],

    "Leadership Change": [
        "appointed ceo",
        "appointed as ceo",
        "named ceo",
        "named as ceo",
        "new ceo",
        "became ceo",
        "resigned as ceo",
        "resignation",
        "resigned",
        "steps down",
        "stepped down",
        "replaced as ceo",
        "replacement as ceo"
    ]
}


def keyword_matches(text, keyword):

    text = str(text).lower()
    keyword = keyword.lower()

    pattern = r"\b" + re.escape(keyword) + r"\b"

    return bool(
        re.search(pattern, text)
    )


def detect_event(text):

    text = str(text).lower()

    detected_events = []

    for event, keywords in EVENT_KEYWORDS.items():

        for keyword in keywords:

            if keyword_matches(
                text,
                keyword
            ):
                detected_events.append(event)
                break

    if not detected_events:
        return "Other"

    return ", ".join(detected_events)


def add_events(df):

    df = df.copy()

    df["Event"] = df["Clean_Text"].apply(
        detect_event
    )

    return df