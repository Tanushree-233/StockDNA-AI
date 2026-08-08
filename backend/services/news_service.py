import feedparser


def get_latest_news(ticker: str):

    url = (
        f"https://news.google.com/rss/search?"
        f"q={ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)

    news = []

    for article in feed.entries[:5]:

        news.append({

            "title": article.title,

            "publisher": article.source.title
            if hasattr(article, "source")
            else "Unknown",

            "published": article.published,

            "link": article.link

        })

    return news