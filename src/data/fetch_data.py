# src/data/fetch_data.py

"""
Functions to fetch market and sentiment data from:
  • Yahoo Finance (stocks)
  • Binance.US via CCXT (crypto)
  • NewsAPI.org (headlines sentiment)
"""

import pandas as pd
import yfinance as yf
import ccxt
from newsapi import NewsApiClient
from datetime import datetime
try:
    from utils.config import (
        ALPHAVANTAGE_KEY,
        BINANCE_US_API_KEY,
        BINANCE_US_SECRET,
        NEWSAPI_KEY,
    )
except ImportError:
    # Fallback values for testing
    ALPHAVANTAGE_KEY = "demo"
    BINANCE_US_API_KEY = "demo"
    BINANCE_US_SECRET = "demo"
    NEWSAPI_KEY = "demo"


def get_stock_data(ticker: str,
                   start: str,
                   end: str,
                   interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance.
    Returns a DataFrame indexed by timezone-naive datetime.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.index = df.index.tz_localize(None)
    return df


def get_crypto_data(symbol: str,
                    timeframe: str = "1h",
                    since: int = None) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance.US via CCXT.
    `symbol` example: "BTC/USDT"
    """
    exchange = ccxt.binanceus({
        'apiKey': BINANCE_US_API_KEY,
        'secret': BINANCE_US_SECRET,
    })
    # default: start of today UTC
    since = since or exchange.parse8601(
        f"{datetime.utcnow().date().isoformat()}T00:00:00Z"
    )
    ohlcv = exchange.fetch_ohlcv(symbol,
                                 timeframe=timeframe,
                                 since=since)
    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')


def get_news_sentiment(query: str,
                       from_date: str,
                       to_date: str) -> pd.DataFrame:
    """
    Fetch headlines via NewsAPI.org and compute a simple sentiment score:
      +1 for each 'bull' in title, -1 for each 'bear'.
    Aggregates daily average.
    """
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    resp = client.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language='en',
        sort_by='relevancy',
        page_size=100
    )
    articles = resp.get('articles', [])
    records = []
    for art in articles:
        title = art.get('title') or ""
        score = title.lower().count('bull') - title.lower().count('bear')
        records.append({
            'date': art['publishedAt'][:10],
            'sentiment': score
        })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date').sentiment.mean().to_frame()
