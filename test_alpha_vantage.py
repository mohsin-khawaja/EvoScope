#!/usr/bin/env python3
"""
Test Alpha Vantage API Integration
This script tests your Alpha Vantage API key and demonstrates its capabilities
"""

import os
import requests
import pandas as pd
from datetime import datetime
import json

# Your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "38RX2Y3EUK2CV7Y8"
BASE_URL = "https://www.alphavantage.co/query"

def test_api_connection():
    """Test basic API connection"""
    print("🔍 Testing Alpha Vantage API connection...")
    
    # Test with a simple quote request
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": "AAPL",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            print("✅ Alpha Vantage API connection successful!")
            print(f"   AAPL Price: ${quote.get('05. price', 'N/A')}")
            print(f"   Change: {quote.get('09. change', 'N/A')}")
            print(f"   Volume: {quote.get('06. volume', 'N/A')}")
            return True
        else:
            print(f"❌ API Error: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def get_technical_indicators(symbol="AAPL"):
    """Get technical indicators for a symbol"""
    print(f"\n📊 Getting technical indicators for {symbol}...")
    
    # Get RSI
    rsi_params = {
        "function": "RSI",
        "symbol": symbol,
        "interval": "daily",
        "time_period": 14,
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        rsi_response = requests.get(BASE_URL, params=rsi_params)
        rsi_data = rsi_response.json()
        
        if "Technical Analysis: RSI" in rsi_data:
            rsi_values = list(rsi_data["Technical Analysis: RSI"].values())
            latest_rsi = float(rsi_values[0]["RSI"])
            print(f"   RSI (14): {latest_rsi:.2f}")
        else:
            print(f"   RSI: Error - {rsi_data}")
            
    except Exception as e:
        print(f"   RSI Error: {e}")
    
    # Get MACD
    macd_params = {
        "function": "MACD",
        "symbol": symbol,
        "interval": "daily",
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        macd_response = requests.get(BASE_URL, params=macd_params)
        macd_data = macd_response.json()
        
        if "Technical Analysis: MACD" in macd_data:
            macd_values = list(macd_data["Technical Analysis: MACD"].values())
            latest_macd = macd_values[0]
            print(f"   MACD: {latest_macd['MACD']}")
            print(f"   MACD Signal: {latest_macd['MACD_Signal']}")
        else:
            print(f"   MACD: Error - {macd_data}")
            
    except Exception as e:
        print(f"   MACD Error: {e}")

def get_earnings_calendar():
    """Get upcoming earnings calendar"""
    print(f"\n📅 Getting earnings calendar...")
    
    params = {
        "function": "EARNINGS_CALENDAR",
        "horizon": "3month",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            # Parse CSV response
            lines = response.text.strip().split('\n')
            if len(lines) > 1:
                print(f"   Found {len(lines)-1} upcoming earnings")
                # Show first 3 earnings
                for i, line in enumerate(lines[1:4]):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        date = parts[1]
                        estimate = parts[2] if len(parts) > 2 else "N/A"
                        print(f"   {symbol}: {date} (Est: {estimate})")
            else:
                print("   No earnings data found")
        else:
            print(f"   Error: {response.status_code}")
            
    except Exception as e:
        print(f"   Earnings Error: {e}")

def get_economic_indicators():
    """Get economic indicators"""
    print(f"\n📈 Getting economic indicators...")
    
    # Get GDP data
    gdp_params = {
        "function": "REAL_GDP",
        "interval": "quarterly",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=gdp_params)
        data = response.json()
        
        if "data" in data:
            latest_gdp = data["data"][0]
            print(f"   Latest GDP: {latest_gdp['value']} ({latest_gdp['date']})")
        else:
            print(f"   GDP Error: {data}")
            
    except Exception as e:
        print(f"   GDP Error: {e}")

def create_enhanced_config():
    """Create enhanced configuration with Alpha Vantage"""
    print(f"\n⚙️ Creating enhanced configuration...")
    
    config = {
        "alpha_vantage": {
            "api_key": ALPHA_VANTAGE_API_KEY,
            "base_url": BASE_URL,
            "features": [
                "Real-time stock quotes",
                "Technical indicators (RSI, MACD, Bollinger Bands)",
                "Fundamental data (earnings, financials)",
                "Economic indicators (GDP, inflation)",
                "Forex and crypto data",
                "News sentiment"
            ]
        },
        "recommended_next_apis": [
            "Alpaca Markets (paper trading)",
            "NewsAPI (sentiment analysis)",
            "Finnhub (institutional data)"
        ]
    }
    
    # Save config
    with open("alpha_vantage_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to alpha_vantage_config.json")
    return config

def main():
    """Main test function"""
    print("🚀 Alpha Vantage API Test Suite")
    print("=" * 50)
    
    # Test basic connection
    if not test_api_connection():
        print("❌ API connection failed. Please check your key.")
        return
    
    # Test technical indicators
    get_technical_indicators("AAPL")
    
    # Test earnings calendar
    get_earnings_calendar()
    
    # Test economic indicators
    get_economic_indicators()
    
    # Create enhanced config
    config = create_enhanced_config()
    
    print("\n" + "=" * 50)
    print("🎉 Alpha Vantage API Integration Complete!")
    print("\n📋 Next Steps:")
    print("1. ✅ Alpha Vantage is working")
    print("2. 🔄 Integrate with your trading system")
    print("3. 📊 Add technical indicators to your AI models")
    print("4. 📰 Get NewsAPI for sentiment analysis")
    print("5. 💰 Get Alpaca for paper trading")
    
    print(f"\n🔑 Your API Key: {ALPHA_VANTAGE_API_KEY}")
    print("📊 Free Tier: 500 requests/day, 25 calls/minute")
    print("🌐 Dashboard: https://www.alphavantage.co/support/#api-key")

if __name__ == "__main__":
    main() 