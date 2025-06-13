import pandas as pd
import numpy as np

def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data for a given symbol between start and end dates.
    Placeholder implementation returns a dummy DataFrame.
    """
    dates = pd.date_range(start, end)
    data = {
        'Open': np.random.uniform(150, 200, len(dates)),  # More realistic stock prices
        'High': np.random.uniform(160, 210, len(dates)),
        'Low': np.random.uniform(140, 190, len(dates)),
        'Close': np.random.uniform(150, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, size=len(dates))  # More realistic volume
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

def get_crypto_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical crypto data for a given symbol between start and end dates.
    Placeholder implementation returns a dummy DataFrame.
    """
    dates = pd.date_range(start, end)
    data = {
        'Open': np.random.uniform(40000, 60000, len(dates)),  # More realistic BTC prices
        'High': np.random.uniform(42000, 62000, len(dates)),
        'Low': np.random.uniform(38000, 58000, len(dates)),
        'Close': np.random.uniform(40000, 60000, len(dates)),
        'Volume': np.random.randint(100, 1000, size=len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

def get_news_sentiment(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch news sentiment data for a given symbol between start and end dates.
    Placeholder implementation returns a dummy DataFrame.
    """
    dates = pd.date_range(start, end)
    data = {
        'sentiment': np.random.uniform(-1, 1, len(dates))  # Sentiment between -1 and 1
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df 