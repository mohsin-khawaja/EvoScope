from src.data.fetch_data import get_stock_data, get_crypto_data, get_news_sentiment

if __name__ == "__main__":
    print("=== STOCK DATA ===")
    print(get_stock_data("AAPL", "2022-01-01", "2022-01-05").head(), "\n")
    print("=== CRYPTO DATA ===")
    print(get_crypto_data("BTC/USDT", timeframe="1d").head(), "\n")
    print("=== NEWS SENTIMENT ===")
    print(get_news_sentiment("Bitcoin", "2022-01-01", "2022-01-05")) 