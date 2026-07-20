from scripts.collectors.download_prices import download_prices
from scripts.collectors.download_market import download_market
from scripts.collectors.download_fundamentals import download_fundamentals
from scripts.collectors.download_earnings import download_earnings
from scripts.collectors.download_macro import download_macro

def main():

    print("=" * 50)
    print("StockDNA AI")
    print("=" * 50)

    download_prices()

    print("Running Market Downloader...")

    download_market()

    print("Market Downloader Finished")

    print("Running Fundamentals Downloader...")

    download_fundamentals()

    print("Fundamentals Downloader Finished")

    print("Running Earnings Downloader...")

    download_earnings()

    print("Earnings Downloader Finished")

    print("\nDownload Completed")

    print("Running Macro Downloader...")

    download_macro()

    print("Macro Downloader Finished")


if __name__ == "__main__":
    main()