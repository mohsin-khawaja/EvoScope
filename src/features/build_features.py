# src/features/build_features.py
import pandas as pd
from ..data.fetch_data import get_stock_data, get_crypto_data, get_news_sentiment

def build_dataset():
    stocks = get_stock_data("AAPL", "2021-01-01", "2021-12-31")
    crypto = get_crypto_data("BTC/USDT", timeframe="1d")
    news  = get_news_sentiment("Apple Inc", "2021-01-01", "2021-12-31")

    df = stocks[['Close']].rename(columns={'Close':'AAPL_Close'})
    df = df.join(crypto['close'].rename('BTC_Close'), how='outer')
    df = df.join(news, how='left')
    df = df.fillna(method='ffill').dropna()
    return df
