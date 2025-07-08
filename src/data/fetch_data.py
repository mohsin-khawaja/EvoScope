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
import requests
from datetime import datetime
try:
    from utils.config import (
        ALPHAVANTAGE_KEY,
        BINANCE_US_API_KEY,
        BINANCE_US_SECRET,
        NEWSAPI_KEY,
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        ALPACA_BASE_URL,
        FRED_API_KEY,
    )
    # Use the working Alpha Vantage key if config loading fails
    if not ALPHAVANTAGE_KEY or ALPHAVANTAGE_KEY == "demo":
        ALPHAVANTAGE_KEY = "38RX2Y3EUK2CV7Y8"
except ImportError:
    # Fallback values for testing
    ALPHAVANTAGE_KEY = "38RX2Y3EUK2CV7Y8"  # Working key from other files
    BINANCE_US_API_KEY = "UVmgRMxKoetKkVgEEcuoPhmjGSBgtY3OfhA5Gl9jPFcDpD7LAcs7btnPVJTyqXnf"
    BINANCE_US_SECRET = "5JitR0QMrk8JcATQ1wgvu1jK1fEKwbzt0SDRUXAN4bG2ItvEilb3sFTEzg0aFq0N"
    NEWSAPI_KEY = "1d5bb349-6f72-4f83-860b-9c2fb3c220bd"  # Working key from other files
    ALPACA_API_KEY = "PKH6HJ2RBVZ20P8EJPNT"
    ALPACA_SECRET_KEY = "your_secret_key_here"
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
    FRED_API_KEY = "56JBx7QuGHquzDi6yzMd"

# Alpha Vantage base URL
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# FRED base URL
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Paper Trading (Start Here)
ALPACA_API_KEY=your_alpaca_paper_key
ALPACA_SECRET_KEY=your_alpaca_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# OpenAI for LLM Analysis
OPENAI_API_KEY=your_openai_key

# Market Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWSAPI_KEY=your_newsapi_key

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
    
    # Handle multi-level columns for single ticker
    if df.columns.nlevels > 1:
        # For single ticker, flatten the multi-level columns
        df.columns = df.columns.get_level_values(0)
    
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


def get_alpha_vantage_quote(symbol: str) -> pd.DataFrame:
    """
    Get real-time stock quote from Alpha Vantage.
    Returns a DataFrame with current price, change, volume, etc.
    """
    url = ALPHA_VANTAGE_BASE_URL
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            df = pd.DataFrame([{
                'symbol': symbol,
                'open': float(quote.get('02. open', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'close': float(quote.get('05. price', 0)),
                'previous_close': float(quote.get('08. previous close', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%').replace('%', ''),
                'volume': int(quote.get('06. volume', 0)),
                'timestamp': datetime.now()
            }])
            return df.set_index('timestamp')
        else:
            print(f"Alpha Vantage API Error: {data}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching Alpha Vantage quote for {symbol}: {e}")
        return pd.DataFrame()


def get_alpha_vantage_intraday(symbol: str, 
                               interval: str = "5min", 
                               outputsize: str = "compact") -> pd.DataFrame:
    """
    Get intraday stock data from Alpha Vantage.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        interval: Time interval (1min, 5min, 15min, 30min, 60min)
        outputsize: "compact" (last 100 data points) or "full" (all available)
    """
    url = ALPHA_VANTAGE_BASE_URL
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": ALPHAVANTAGE_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        time_series_key = f"Time Series ({interval})"
        if time_series_key in data:
            time_series = data[time_series_key]
            records = []
            
            for timestamp, values in time_series.items():
                records.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(records)
            return df.set_index('timestamp').sort_index()
        else:
            print(f"Alpha Vantage API Error: {data}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching Alpha Vantage intraday for {symbol}: {e}")
        return pd.DataFrame()


def get_alpha_vantage_technical_indicators(symbol: str, 
                                           indicator: str = "RSI", 
                                           interval: str = "daily",
                                           time_period: int = 14) -> pd.DataFrame:
    """
    Get technical indicators from Alpha Vantage.
    
    Args:
        symbol: Stock symbol
        indicator: Technical indicator (RSI, SMA, EMA, MACD, etc.)
        interval: Time interval (daily, weekly, monthly)
        time_period: Lookback period for calculation
    """
    url = ALPHA_VANTAGE_BASE_URL
    params = {
        "function": indicator,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close",
        "apikey": ALPHAVANTAGE_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        technical_key = f"Technical Analysis: {indicator}"
        if technical_key in data:
            technical_data = data[technical_key]
            records = []
            
            for date, values in technical_data.items():
                record = {'date': pd.to_datetime(date)}
                for key, value in values.items():
                    record[key.lower()] = float(value)
                records.append(record)
            
            df = pd.DataFrame(records)
            return df.set_index('date').sort_index()
        else:
            print(f"Alpha Vantage Technical Indicator Error: {data}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching {indicator} for {symbol}: {e}")
        return pd.DataFrame()


def get_alpha_vantage_company_overview(symbol: str) -> dict:
    """
    Get company fundamental data from Alpha Vantage.
    Returns company overview including financials, ratios, etc.
    """
    url = ALPHA_VANTAGE_BASE_URL
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Symbol" in data:
            # Convert numeric fields to appropriate types
            numeric_fields = [
                'MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio',
                'BookValue', 'DividendPerShare', 'DividendYield', 'EPS',
                'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM',
                'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM',
                'GrossProfitTTM', 'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY',
                'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE',
                'ForwardPE', 'PriceToSalesRatioTTM', 'PriceToBookRatio',
                'EVToRevenue', 'EVToEBITDA', 'Beta', '52WeekHigh', '52WeekLow',
                '50DayMovingAverage', '200DayMovingAverage', 'SharesOutstanding'
            ]
            
            for field in numeric_fields:
                if field in data and data[field] not in ['None', 'N/A', '-']:
                    try:
                        data[field] = float(data[field])
                    except ValueError:
                        data[field] = None
                        
            return data
        else:
            print(f"Alpha Vantage Company Overview Error: {data}")
            return {}
            
    except Exception as e:
        print(f"Error fetching company overview for {symbol}: {e}")
        return {}


def get_enhanced_stock_data(symbol: str, 
                           use_alpha_vantage: bool = True,
                           get_technicals: bool = True) -> dict:
    """
    Get enhanced stock data combining Yahoo Finance and Alpha Vantage.
    
    Args:
        symbol: Stock symbol
        use_alpha_vantage: Whether to include Alpha Vantage data
        get_technicals: Whether to fetch technical indicators
    
    Returns:
        Dictionary with combined data from multiple sources
    """
    result = {'symbol': symbol, 'timestamp': datetime.now()}
    
    # Yahoo Finance data (historical)
    try:
        yf_data = yf.download(symbol, period="1d", interval="1m", auto_adjust=True)
        if not yf_data.empty:
            result['yfinance'] = {
                'current_price': yf_data['Close'].iloc[-1],
                'volume': yf_data['Volume'].iloc[-1],
                'high': yf_data['High'].iloc[-1],
                'low': yf_data['Low'].iloc[-1]
            }
    except Exception as e:
        print(f"Yahoo Finance error for {symbol}: {e}")
        result['yfinance'] = {}
    
    # Alpha Vantage data
    if use_alpha_vantage:
        # Real-time quote
        av_quote = get_alpha_vantage_quote(symbol)
        if not av_quote.empty:
            result['alphavantage_quote'] = av_quote.iloc[-1].to_dict()
        
        # Company fundamentals
        company_overview = get_alpha_vantage_company_overview(symbol)
        if company_overview:
            result['company_overview'] = company_overview
        
        # Technical indicators
        if get_technicals:
            rsi = get_alpha_vantage_technical_indicators(symbol, "RSI", "daily", 14)
            if not rsi.empty:
                result['rsi'] = float(rsi.iloc[-1]['rsi'])
            
            macd = get_alpha_vantage_technical_indicators(symbol, "MACD", "daily")
            if not macd.empty:
                result['macd'] = {
                    'macd': float(macd.iloc[-1]['macd']),
                    'macd_signal': float(macd.iloc[-1]['macd_signal']),
                    'macd_hist': float(macd.iloc[-1]['macd_hist'])
                }
    
    return result


def get_alpaca_stock_data(symbol: str, 
                          timeframe: str = "1Day",
                          start_date: str = None,
                          end_date: str = None,
                          limit: int = 1000) -> pd.DataFrame:
    """
    Fetch stock data from Alpaca API.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        timeframe: Time frame for bars (1Min, 5Min, 15Min, 30Min, 1Hour, 1Day)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Maximum number of bars to return
    
    Returns:
        DataFrame with OHLCV data
    """
    url = f"{ALPACA_BASE_URL}/stocks/{symbol}/bars"
    
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    params = {
        "timeframe": timeframe,
        "limit": limit
    }
    
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "bars" in data:
            bars = data["bars"]
            records = []
            
            for bar in bars:
                records.append({
                    'timestamp': pd.to_datetime(bar['t']),
                    'open': float(bar['o']),
                    'high': float(bar['h']),
                    'low': float(bar['l']),
                    'close': float(bar['c']),
                    'volume': int(bar['v'])
                })
            
            df = pd.DataFrame(records)
            return df.set_index('timestamp').sort_index()
        else:
            print(f"Alpaca API Error: {data}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching Alpaca stock data for {symbol}: {e}")
        return pd.DataFrame()


def get_alpaca_account_info() -> dict:
    """
    Get account information from Alpaca API.
    
    Returns:
        Dictionary with account details
    """
    url = f"{ALPACA_BASE_URL}/account"
    
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        print(f"Error fetching Alpaca account info: {e}")
        return {}


def get_alpaca_positions() -> pd.DataFrame:
    """
    Get current positions from Alpaca API.
    
    Returns:
        DataFrame with position data
    """
    url = f"{ALPACA_BASE_URL}/positions"
    
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        positions = response.json()
        
        if positions:
            records = []
            for position in positions:
                records.append({
                    'symbol': position['symbol'],
                    'qty': float(position['qty']),
                    'side': position['side'],
                    'market_value': float(position['market_value']),
                    'cost_basis': float(position['cost_basis']),
                    'unrealized_pl': float(position['unrealized_pl']),
                    'unrealized_plpc': float(position['unrealized_plpc']),
                    'current_price': float(position['current_price']),
                    'lastday_price': float(position['lastday_price']),
                    'change_today': float(position['change_today'])
                })
            
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching Alpaca positions: {e}")
        return pd.DataFrame()


def get_fred_series(series_id: str, 
                    start_date: str = None, 
                    end_date: str = None, 
                    limit: int = 1000) -> pd.DataFrame:
    """
    Fetch economic data series from FRED API.
    
    Args:
        series_id: FRED series ID (e.g., "GDP", "UNRATE", "FEDFUNDS")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Maximum number of observations to return
    
    Returns:
        DataFrame with date index and series values
    """
    url = f"{FRED_BASE_URL}/series/observations"
    
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit
    }
    
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "observations" in data:
            observations = data["observations"]
            records = []
            
            for obs in observations:
                if obs["value"] != ".":  # FRED uses "." for missing values
                    records.append({
                        'date': pd.to_datetime(obs['date']),
                        'value': float(obs['value'])
                    })
            
            if records:
                df = pd.DataFrame(records)
                df = df.set_index('date').sort_index()
                df.columns = [series_id]
                return df
            else:
                return pd.DataFrame()
        else:
            print(f"FRED API Error: {data}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return pd.DataFrame()


def get_economic_indicators(indicators: list = None, 
                           start_date: str = None, 
                           end_date: str = None) -> pd.DataFrame:
    """
    Fetch multiple economic indicators from FRED API.
    
    Args:
        indicators: List of FRED series IDs. If None, uses default set.
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with economic indicators as columns
    """
    if indicators is None:
        indicators = [
            "FEDFUNDS",    # Federal Funds Rate
            "UNRATE",      # Unemployment Rate
            "CPIAUCSL",    # Consumer Price Index (Inflation)
            "GDP",         # Gross Domestic Product
            "DGS10",       # 10-Year Treasury Rate
            "DGS2",        # 2-Year Treasury Rate
            "UMCSENT",     # Consumer Sentiment
            "INDPRO",      # Industrial Production Index
            "HOUST",       # Housing Starts
            "PAYEMS"       # Nonfarm Payrolls
        ]
    
    combined_df = pd.DataFrame()
    
    for indicator in indicators:
        df = get_fred_series(indicator, start_date, end_date)
        if not df.empty:
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
    
    return combined_df


def get_fed_funds_rate(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get Federal Funds Rate from FRED."""
    return get_fred_series("FEDFUNDS", start_date, end_date)


def get_unemployment_rate(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get Unemployment Rate from FRED."""
    return get_fred_series("UNRATE", start_date, end_date)


def get_inflation_rate(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get Consumer Price Index (Inflation) from FRED."""
    return get_fred_series("CPIAUCSL", start_date, end_date)


def get_gdp_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get GDP data from FRED."""
    return get_fred_series("GDP", start_date, end_date)


def get_treasury_rates(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get Treasury yield rates from FRED."""
    rates = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS20", "DGS30"]
    combined_df = pd.DataFrame()
    
    for rate in rates:
        df = get_fred_series(rate, start_date, end_date)
        if not df.empty:
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
    
    return combined_df


def get_yield_curve() -> pd.DataFrame:
    """Get current yield curve data from FRED."""
    return get_treasury_rates()


def get_consumer_sentiment(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get Consumer Sentiment Index from FRED."""
    return get_fred_series("UMCSENT", start_date, end_date)


def get_fred_series_info(series_id: str) -> dict:
    """
    Get information about a FRED series.
    
    Args:
        series_id: FRED series ID
    
    Returns:
        Dictionary with series metadata
    """
    url = f"{FRED_BASE_URL}/series"
    
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "seriess" in data and len(data["seriess"]) > 0:
            return data["seriess"][0]
        else:
            print(f"FRED Series Info Error: {data}")
            return {}
            
    except Exception as e:
        print(f"Error fetching FRED series info for {series_id}: {e}")
        return {}

# Daily monitoring checklist
def daily_review():
    performance = get_performance_summary()
    
    if performance['total_return'] < -10:  # Stop at 10% portfolio loss
        stop_all_trading()
    
    if performance['volatility'] > 0.3:   # High volatility warning
        reduce_position_sizes()
