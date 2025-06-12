from src.data.fetch_data import get_stock_data, get_crypto_data, get_news_sentiment
from datetime import datetime, timedelta

if __name__ == "__main__":
    print("=== STOCK DATA ===")
    print(get_stock_data("AAPL", "2024-01-01", "2024-01-05").head(), "\n")
    
    print("=== CRYPTO DATA ===")
    print(get_crypto_data("BTC/USDT", timeframe="1d").head(), "\n")
    
    print("=== NEWS SENTIMENT ===")
    # Use recent dates for NewsAPI (free tier has limited historical data)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"Fetching news from {start_date} to {end_date}")
    print(get_news_sentiment("Bitcoin", start_date, end_date)) 