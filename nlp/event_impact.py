POSITIVE_EVENTS = {
    "Partnership",
    "Acquisition",
    "Product Launch",
    "Expansion",
    "Contract"
}

NEGATIVE_EVENTS = {
    "Regulatory",
    "Leadership Change"
}


def determine_event_impact(event, sentiment):

    if not event or event == "Other":
        return sentiment

    events = [
        item.strip()
        for item in str(event).split(",")
    ]

    positive_count = sum(
        item in POSITIVE_EVENTS
        for item in events
    )

    negative_count = sum(
        item in NEGATIVE_EVENTS
        for item in events
    )

    if negative_count > positive_count:
        return "Negative"

    if positive_count > negative_count:
        return "Positive"

    return sentiment


def add_event_impact(df):

    df = df.copy()

    df["Event_Impact"] = df.apply(
        lambda row: determine_event_impact(
            row["Event"],
            row["Sentiment"]
        ),
        axis=1
    )

    return df
