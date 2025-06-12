# src/data/fetch_data.py
import pandas as pd
import yfinance as yf
import ccxt
from newsapi import NewsApiClient
from datetime import datetime
from ..utils.config import ALPHAVANTAGE_KEY, BINANCE_API_KEY, BINANCE_SECRET, NEWSAPI_KEY

def get_stock_data(ticker: str, start: str, end: str, interval: str="1d") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.index = df.index.tz_localize(None)
    return df

def get_crypto_data(symbol: str, timeframe: str="1h", since: int=None) -> pd.DataFrame:
    """Fetch OHLCV from Binance via CCXT."""
    exchange = ccxt.binance({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_SECRET})
    since = since or exchange.parse8601(f'{datetime.utcnow().date().isoformat()}T00:00:00Z')
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')

def get_news_sentiment(query: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch headlines via NewsAPI and compute simple sentiment score."""
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    res = client.get_everything(q=query, from_param=from_date, to_param=to_date, language='en')
    articles = res.get('articles', [])
    data = []
    for art in articles:
        title = art.get('title', "")
        score = title.lower().count('bull') - title.lower().count('bear')
        data.append({'date': art['publishedAt'][:10], 'sentiment': score})
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date').sentiment.mean().to_frame()
