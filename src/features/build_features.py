# src/features/build_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..data.fetch_data import get_stock_data, get_crypto_data, get_news_sentiment

def build_dataset():
    try:
        # Use recent dates for better compatibility with free API tiers
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year of data
        
        print(f"Fetching stock data from {start_date} to {end_date}")
        stocks = get_stock_data("AAPL", start_date, end_date)
        
        # Simplify stock data handling
        if hasattr(stocks.columns, 'levels') and len(stocks.columns.levels) > 1:
            # Flatten multi-level columns
            stocks.columns = ['_'.join(col).strip() for col in stocks.columns.values]
            close_col = [col for col in stocks.columns if 'Close' in col][0]
            df = stocks[[close_col]].rename(columns={close_col: 'AAPL_Close'})
        else:
            df = stocks[['Close']].rename(columns={'Close': 'AAPL_Close'})
        
        print(f"Stock data shape: {df.shape}")
        
        # Add crypto data with error handling
        try:
            print("Fetching crypto data...")
            crypto = get_crypto_data("BTC/USDT", timeframe="1d")
            if len(crypto) > 0:
                # Resample crypto to match stock frequency and merge
                crypto_daily = crypto['close'].resample('D').last().rename('BTC_Close')
                df = df.merge(crypto_daily.to_frame(), left_index=True, right_index=True, how='left')
                print(f"After crypto merge: {df.shape}")
            else:
                df['BTC_Close'] = 50000.0  # Fallback price
                print("Using fallback BTC price")
        except Exception as e:
            print(f"Error fetching crypto data: {e}")
            df['BTC_Close'] = 50000.0  # Fallback price
        
        # Add news sentiment with error handling
        try:
            news_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            print(f"Fetching news sentiment from {news_start} to {end_date}")
            news = get_news_sentiment("Apple Inc", news_start, end_date)
            if len(news) > 0:
                df = df.merge(news, left_index=True, right_index=True, how='left')
                print(f"After news merge: {df.shape}")
            else:
                df['sentiment'] = 0.0
                print("Using neutral sentiment")
        except Exception as e:
            print(f"Error fetching news data: {e}")
            df['sentiment'] = 0.0  # Neutral sentiment
        
        # Ensure all required columns exist
        required_cols = ['AAPL_Close', 'BTC_Close', 'sentiment']
        for col in required_cols:
            if col not in df.columns:
                if col == 'sentiment':
                    df[col] = 0.0
                elif col == 'BTC_Close':
                    df[col] = 50000.0
        
        # Fill missing values for required columns only
        for col in required_cols:
            if col in df.columns:
                if col == 'sentiment':
                    df[col] = df[col].fillna(0.0)
                elif col == 'BTC_Close':
                    df[col] = df[col].ffill().bfill().fillna(50000.0)
                else:
                    df[col] = df[col].ffill().bfill()
        
        # Only drop rows where ALL required columns are NaN
        df = df.dropna(subset=required_cols, how='all')
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df[required_cols]  # Return only required columns
        
    except Exception as e:
        print(f"Error in build_dataset: {e}")
        # Return minimal fallback dataset
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        df = pd.DataFrame({
            'AAPL_Close': np.random.uniform(170, 190, len(dates)),
            'BTC_Close': np.random.uniform(40000, 60000, len(dates)),
            'sentiment': np.random.uniform(-0.5, 0.5, len(dates))
        }, index=dates)
        print(f"Using fallback dataset with shape: {df.shape}")
        return df
