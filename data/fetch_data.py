import pandas as pd
import numpy as np
import yfinance as yf

def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data for a given symbol between start and end dates.
    Uses yfinance for real stock data.
    """
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d")
        df.index = df.index.tz_localize(None)
        
        # Handle multi-level columns for single ticker
        if df.columns.nlevels > 1:
            # For single ticker, flatten the multi-level columns
            df.columns = df.columns.get_level_values(0)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        # Fallback to dummy data
        dates = pd.date_range(start, end)
        data = {
            'Open': np.random.uniform(150, 200, len(dates)),
            'High': np.random.uniform(160, 210, len(dates)),
            'Low': np.random.uniform(140, 190, len(dates)),
            'Close': np.random.uniform(150, 200, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, size=len(dates))
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