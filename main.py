from scripts.collectors.download_prices import download_prices
from scripts.collectors.download_market import download_market
from scripts.collectors.download_fundamentals import download_fundamentals
from scripts.collectors.download_earnings import download_earnings
from scripts.collectors.download_macro import download_macro
from scripts.processors.preprocess_prices import preprocess_prices
from scripts.processors.preprocess_market import preprocess_market
from scripts.processors.preprocess_fundamentals import preprocess_fundamentals
from scripts.processors.preprocess_earnings import preprocess_earnings
from scripts.processors.preprocess_macro import preprocess_macro
from scripts.processors.merge_dataset import merge_dataset

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

    print("Running Price Preprocessing...")

    preprocess_prices()

    print("Price Preprocessing Finished")
    
    print("Running Market Preprocessing...")

    preprocess_market()

    print("Market Preprocessing Finished")

    print("Running Fundamentals Preprocessing...")

    preprocess_fundamentals()

    print("Fundamentals Preprocessing Finished")

    print("Running Earnings Preprocessing...")

    preprocess_earnings()

    print("Earnings Preprocessing Finished")

    print("Running Macro Preprocessing...")

    preprocess_macro()

    print("Macro Preprocessing Finished")

    print("Running Dataset Merging...")

    merge_dataset()

    print("Dataset Merging Finished")

if __name__ == "__main__":
    main()