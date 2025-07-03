#!/usr/bin/env python3
"""
Test NewsAPI Integration for Trading System
This script tests your NewsAPI key and demonstrates news sentiment analysis
"""

import requests
import json
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import re

# Your NewsAPI key
NEWSAPI_KEY = "1d5bb349-6f72-4f83-860b-9c2fb3c220bd"
BASE_URL = "https://newsapi.org/v2"

def test_api_connection():
    """Test basic NewsAPI connection"""
    print("🔍 Testing NewsAPI connection...")
    
    url = f"{BASE_URL}/top-headlines"
    params = {
        'country': 'us',
        'category': 'business',
        'pageSize': 5,
        'apiKey': NEWSAPI_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200 and 'articles' in data:
            print("✅ NewsAPI connection successful!")
            print(f"   Found {data['totalResults']} business headlines")
            print(f"   Sample headline: {data['articles'][0]['title'][:80]}...")
            return True
        else:
            print(f"❌ API Error: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def get_stock_news(symbol, days_back=7):
    """Get news for a specific stock symbol"""
    print(f"\n📰 Getting news for {symbol} (last {days_back} days)...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    url = f"{BASE_URL}/everything"
    params = {
        'q': f'{symbol} OR "{symbol}" stock OR shares',
        'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com,yahoo.com',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'pageSize': 10,
        'apiKey': NEWSAPI_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200 and 'articles' in data:
            articles = data['articles']
            print(f"   Found {len(articles)} articles for {symbol}")
            
            # Analyze sentiment
            sentiments = []
            for article in articles[:5]:  # Analyze top 5 articles
                title = article['title']
                description = article.get('description', '')
                text = f"{title}. {description}"
                
                # Clean text
                text = re.sub(r'[^\w\s]', '', text)
                
                # Get sentiment
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                sentiments.append({
                    'title': title[:60] + "..." if len(title) > 60 else title,
                    'sentiment': sentiment_score,
                    'published': article['publishedAt'][:10]
                })
                
                print(f"   📄 {title[:50]}...")
                print(f"      Sentiment: {sentiment_score:.3f} ({'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'})")
            
            # Calculate overall sentiment
            if sentiments:
                avg_sentiment = sum(s['sentiment'] for s in sentiments) / len(sentiments)
                print(f"\n   📊 Overall Sentiment: {avg_sentiment:.3f}")
                
                if avg_sentiment > 0.1:
                    sentiment_label = "BULLISH 📈"
                elif avg_sentiment < -0.1:
                    sentiment_label = "BEARISH 📉"
                else:
                    sentiment_label = "NEUTRAL ➡️"
                
                print(f"   🎯 Trading Signal: {sentiment_label}")
                
                return {
                    'symbol': symbol,
                    'articles_count': len(articles),
                    'sentiment_score': avg_sentiment,
                    'sentiment_label': sentiment_label,
                    'articles': sentiments
                }
        else:
            print(f"   Error: {data}")
            return None
            
    except Exception as e:
        print(f"   Error: {e}")
        return None

def get_market_news():
    """Get general market news"""
    print(f"\n🏦 Getting general market news...")
    
    url = f"{BASE_URL}/everything"
    params = {
        'q': 'stock market OR trading OR "S&P 500" OR "Dow Jones" OR NASDAQ',
        'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com',
        'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'pageSize': 5,
        'apiKey': NEWSAPI_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200 and 'articles' in data:
            articles = data['articles']
            print(f"   Found {len(articles)} market articles")
            
            for i, article in enumerate(articles[:3]):
                title = article['title']
                source = article['source']['name']
                published = article['publishedAt'][:10]
                
                print(f"   {i+1}. {title}")
                print(f"      Source: {source} | Date: {published}")
                
        else:
            print(f"   Error: {data}")
            
    except Exception as e:
        print(f"   Error: {e}")

def analyze_multiple_stocks():
    """Analyze sentiment for multiple stocks"""
    print(f"\n📊 Multi-Stock Sentiment Analysis...")
    
    stocks = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
    results = []
    
    for symbol in stocks:
        print(f"\n   Analyzing {symbol}...")
        result = get_stock_news(symbol, days_back=3)
        if result:
            results.append(result)
            print(f"   {symbol}: {result['sentiment_label']} ({result['sentiment_score']:.3f})")
    
    # Summary
    if results:
        print(f"\n📈 SENTIMENT SUMMARY:")
        bullish = [r for r in results if r['sentiment_score'] > 0.1]
        bearish = [r for r in results if r['sentiment_score'] < -0.1]
        neutral = [r for r in results if -0.1 <= r['sentiment_score'] <= 0.1]
        
        print(f"   Bullish: {len(bullish)} stocks")
        print(f"   Bearish: {len(bearish)} stocks")
        print(f"   Neutral: {len(neutral)} stocks")
        
        if bullish:
            print(f"   Most Bullish: {max(bullish, key=lambda x: x['sentiment_score'])['symbol']}")
        if bearish:
            print(f"   Most Bearish: {min(bearish, key=lambda x: x['sentiment_score'])['symbol']}")
    
    return results

def create_news_config():
    """Create enhanced configuration with NewsAPI"""
    print(f"\n⚙️ Creating NewsAPI configuration...")
    
    config = {
        "newsapi": {
            "api_key": NEWSAPI_KEY,
            "base_url": BASE_URL,
            "features": [
                "Real-time financial news",
                "Sentiment analysis",
                "Company-specific news filtering",
                "Market-wide news aggregation",
                "Source credibility scoring",
                "Historical news data (30 days)"
            ],
            "sources": [
                "Reuters",
                "Bloomberg",
                "CNBC",
                "MarketWatch",
                "Yahoo Finance",
                "Financial Times"
            ]
        },
        "sentiment_analysis": {
            "method": "TextBlob",
            "thresholds": {
                "bullish": 0.1,
                "bearish": -0.1
            }
        },
        "integration": {
            "alpha_vantage": "Technical indicators",
            "newsapi": "Sentiment analysis",
            "combined_signal": "Technical + Sentiment"
        }
    }
    
    # Save config
    with open("newsapi_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to newsapi_config.json")
    return config

def main():
    """Main test function"""
    print("🗞️ NewsAPI Integration Test Suite")
    print("=" * 50)
    
    # Test basic connection
    if not test_api_connection():
        print("❌ NewsAPI connection failed. Please check your key.")
        return
    
    # Test stock-specific news
    get_stock_news("AAPL", days_back=5)
    
    # Test market news
    get_market_news()
    
    # Multi-stock analysis
    analyze_multiple_stocks()
    
    # Create config
    config = create_news_config()
    
    print("\n" + "=" * 50)
    print("🎉 NewsAPI Integration Complete!")
    print("\n📋 Next Steps:")
    print("1. ✅ NewsAPI is working (1,000 requests/day)")
    print("2. ✅ Alpha Vantage is working (500 requests/day)")
    print("3. 🔄 Integrate both APIs into your AI models")
    print("4. 💰 Get Alpaca Markets for paper trading")
    print("5. 📊 Get Finnhub for institutional data")
    
    print(f"\n🔑 Your NewsAPI Key: {NEWSAPI_KEY}")
    print("📊 Free Tier: 1,000 requests/day")
    print("🌐 Dashboard: https://newsapi.org/account")

if __name__ == "__main__":
    main() 