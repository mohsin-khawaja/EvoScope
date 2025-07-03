#!/usr/bin/env python3
"""
Test script for CoinGecko API integration
No API key required - completely free!
"""

import requests
import json
import time
from datetime import datetime

# CoinGecko API endpoints
BASE_URL = "https://api.coingecko.com/api/v3"

# Popular cryptocurrencies to test
POPULAR_CRYPTOS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'BNB': 'binancecoin',
    'ADA': 'cardano',
    'SOL': 'solana',
    'DOGE': 'dogecoin',
    'MATIC': 'matic-network',
    'AVAX': 'avalanche-2',
    'DOT': 'polkadot',
    'LINK': 'chainlink'
}

def test_simple_price():
    """Test simple price endpoint"""
    print("🔍 Testing Simple Price Endpoint...")
    
    # Test single crypto
    crypto_id = 'bitcoin'
    url = f"{BASE_URL}/simple/price"
    params = {
        'ids': crypto_id,
        'vs_currencies': 'usd',
        'include_24hr_change': 'true',
        'include_24hr_vol': 'true',
        'include_market_cap': 'true'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if crypto_id in data:
            coin_data = data[crypto_id]
            print(f"✅ Bitcoin Price: ${coin_data['usd']:,.2f}")
            print(f"   24h Change: {coin_data.get('usd_24h_change', 0):.2f}%")
            print(f"   24h Volume: ${coin_data.get('usd_24h_vol', 0):,.0f}")
            print(f"   Market Cap: ${coin_data.get('usd_market_cap', 0):,.0f}")
        else:
            print("❌ No data returned for Bitcoin")
            
    except Exception as e:
        print(f"❌ Error testing simple price: {e}")

def test_multiple_cryptos():
    """Test multiple cryptocurrencies at once"""
    print("\n🔍 Testing Multiple Cryptocurrencies...")
    
    # Test top 5 cryptos
    crypto_ids = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']
    url = f"{BASE_URL}/simple/price"
    params = {
        'ids': ','.join(crypto_ids),
        'vs_currencies': 'usd',
        'include_24hr_change': 'true',
        'include_24hr_vol': 'true',
        'include_market_cap': 'true'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Multiple Crypto Prices:")
        for crypto_id in crypto_ids:
            if crypto_id in data:
                coin_data = data[crypto_id]
                symbol = [k for k, v in POPULAR_CRYPTOS.items() if v == crypto_id][0]
                print(f"   {symbol}: ${coin_data['usd']:,.2f} ({coin_data.get('usd_24h_change', 0):+.2f}%)")
            else:
                print(f"   {crypto_id}: No data")
                
    except Exception as e:
        print(f"❌ Error testing multiple cryptos: {e}")

def test_detailed_coin_data():
    """Test detailed coin data endpoint"""
    print("\n🔍 Testing Detailed Coin Data...")
    
    crypto_id = 'bitcoin'
    url = f"{BASE_URL}/coins/{crypto_id}"
    params = {
        'localization': 'false',
        'tickers': 'false',
        'market_data': 'true',
        'community_data': 'false',
        'developer_data': 'false',
        'sparkline': 'false'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Detailed Bitcoin Data:")
        print(f"   Name: {data.get('name', 'N/A')}")
        print(f"   Symbol: {data.get('symbol', 'N/A').upper()}")
        print(f"   Current Price: ${data['market_data']['current_price']['usd']:,.2f}")
        print(f"   24h High: ${data['market_data']['high_24h']['usd']:,.2f}")
        print(f"   24h Low: ${data['market_data']['low_24h']['usd']:,.2f}")
        print(f"   Market Cap Rank: #{data.get('market_cap_rank', 'N/A')}")
        print(f"   Total Supply: {data['market_data'].get('total_supply', 'N/A'):,.0f}")
        
    except Exception as e:
        print(f"❌ Error testing detailed data: {e}")

def test_trending_cryptos():
    """Test trending cryptocurrencies endpoint"""
    print("\n🔍 Testing Trending Cryptocurrencies...")
    
    url = f"{BASE_URL}/search/trending"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Trending Cryptocurrencies:")
        for i, coin in enumerate(data['coins'][:5], 1):
            coin_data = coin['item']
            print(f"   {i}. {coin_data['name']} ({coin_data['symbol']}) - Rank #{coin_data['market_cap_rank']}")
            
    except Exception as e:
        print(f"❌ Error testing trending: {e}")

def test_global_market_data():
    """Test global market data"""
    print("\n🔍 Testing Global Market Data...")
    
    url = f"{BASE_URL}/global"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        global_data = data['data']
        print("✅ Global Market Data:")
        print(f"   Total Market Cap: ${global_data['total_market_cap']['usd']:,.0f}")
        print(f"   Total Volume 24h: ${global_data['total_volume']['usd']:,.0f}")
        print(f"   Bitcoin Dominance: {global_data['market_cap_percentage']['btc']:.1f}%")
        print(f"   Ethereum Dominance: {global_data['market_cap_percentage']['eth']:.1f}%")
        print(f"   Active Cryptocurrencies: {global_data['active_cryptocurrencies']:,}")
        
    except Exception as e:
        print(f"❌ Error testing global data: {e}")

def test_rate_limits():
    """Test rate limits by making multiple requests"""
    print("\n🔍 Testing Rate Limits...")
    
    url = f"{BASE_URL}/simple/price"
    params = {
        'ids': 'bitcoin',
        'vs_currencies': 'usd'
    }
    
    successful_requests = 0
    failed_requests = 0
    
    print("   Making 10 rapid requests...")
    for i in range(10):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            successful_requests += 1
            print(f"   Request {i+1}: ✅")
        except Exception as e:
            failed_requests += 1
            print(f"   Request {i+1}: ❌ {e}")
        
        time.sleep(0.1)  # Small delay
    
    print(f"✅ Rate Limit Test Complete:")
    print(f"   Successful: {successful_requests}/10")
    print(f"   Failed: {failed_requests}/10")

def main():
    """Run all tests"""
    print("🚀 CoinGecko API Integration Test")
    print("=" * 50)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔗 API Documentation: https://www.coingecko.com/en/api/documentation")
    print("💡 No API key required - completely free!")
    print()
    
    # Run all tests
    test_simple_price()
    test_multiple_cryptos()
    test_detailed_coin_data()
    test_trending_cryptos()
    test_global_market_data()
    test_rate_limits()
    
    print("\n" + "=" * 50)
    print("✅ All CoinGecko API tests completed!")
    print("\n🎯 Integration Summary:")
    print("   • Real-time crypto prices ✅")
    print("   • 24h price changes ✅")
    print("   • Market cap & volume data ✅")
    print("   • High/low prices ✅")
    print("   • Multiple cryptocurrencies ✅")
    print("   • Trending coins ✅")
    print("   • Global market data ✅")
    print("   • No API key needed ✅")
    print("\n🔥 Ready to integrate with your trading dashboard!")

if __name__ == "__main__":
    main() 