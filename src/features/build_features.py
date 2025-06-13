# src/features/build_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.fetch_data import get_stock_data, get_crypto_data, get_news_sentiment

def build_features_from_data(data):
    """
    Build features from provided stock data
    """
    try:
        features = data.copy()
        
        # Technical indicators
        # Moving averages
        features['MA_5'] = features['Close'].rolling(window=5).mean()
        features['MA_20'] = features['Close'].rolling(window=20).mean()
        features['MA_50'] = features['Close'].rolling(window=50).mean()
        
        # RSI
        delta = features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features['Close'].ewm(span=12).mean()
        exp2 = features['Close'].ewm(span=26).mean()
        features['MACD'] = exp1 - exp2
        features['MACD_signal'] = features['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_window = 20
        bb_std = features['Close'].rolling(window=bb_window).std()
        bb_mean = features['Close'].rolling(window=bb_window).mean()
        features['BB_Upper'] = bb_mean + (bb_std * 2)
        features['BB_Lower'] = bb_mean - (bb_std * 2)
        features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / bb_mean
        
        # Price-based features
        features['Price_Change'] = features['Close'].pct_change()
        features['High_Low_Ratio'] = features['High'] / features['Low']
        features['Open_Close_Ratio'] = features['Open'] / features['Close']
        
        # Volume features
        if 'Volume' in features.columns:
            features['Volume_MA'] = features['Volume'].rolling(window=20).mean()
            features['Volume_Ratio'] = features['Volume'] / features['Volume_MA']
        else:
            features['Volume_Ratio'] = 1.0
        
        # Volatility
        features['Volatility'] = features['Close'].rolling(window=20).std()
        
        # Select final feature columns
        feature_columns = [
            'Close', 'Open', 'High', 'Low',
            'MA_5', 'MA_20', 'RSI', 'MACD', 'MACD_signal',
            'BB_Upper', 'BB_Lower', 'BB_Width',
            'Price_Change', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Volume_Ratio', 'Volatility'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in feature_columns if col in features.columns]
        
        return features[available_columns].dropna()
        
    except Exception as e:
        print(f"Error in build_features_from_data: {e}")
        # Return basic features
        basic_features = data[['Close', 'Open', 'High', 'Low']].copy()
        basic_features['Price_Change'] = basic_features['Close'].pct_change()
        return basic_features.dropna()

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
