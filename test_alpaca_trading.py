#!/usr/bin/env python3
"""
🚀 Alpaca Markets API Integration Test

This script demonstrates:
- Paper trading with real portfolio management
- Real-time order execution
- Position tracking and management
- Real market data integration
- Professional trading infrastructure

Setup Instructions:
1. Sign up at https://alpaca.markets/
2. Get your API keys from the dashboard
3. Replace the placeholder keys below
4. Run this script to test paper trading

Your current Alpha Vantage integration will work perfectly with Alpaca!
"""

import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd

# Alpaca API Configuration
ALPACA_API_KEY = "your_alpaca_api_key_here"
ALPACA_SECRET_KEY = "your_alpaca_secret_key_here"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading URL

# Headers for API requests
HEADERS = {
    'APCA-API-KEY-ID': ALPACA_API_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
    'Content-Type': 'application/json'
}

class AlpacaTradingClient:
    """Professional trading client for Alpaca Markets"""
    
    def __init__(self):
        self.base_url = ALPACA_BASE_URL
        self.headers = HEADERS
        self.account_info = None
        
    def get_account(self):
        """Get account information"""
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            if response.status_code == 200:
                self.account_info = response.json()
                return self.account_info
            else:
                print(f"Error getting account: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting positions: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def get_orders(self, status='all', limit=50):
        """Get orders"""
        try:
            params = {'status': status, 'limit': limit}
            response = requests.get(f"{self.base_url}/v2/orders", headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting orders: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def place_order(self, symbol, qty, side, order_type='market', limit_price=None, stop_price=None):
        """Place a trading order"""
        try:
            order_data = {
                'symbol': symbol,
                'qty': str(qty),
                'side': side,
                'type': order_type,
                'time_in_force': 'gtc'
            }
            
            if order_type == 'limit' and limit_price:
                order_data['limit_price'] = str(limit_price)
            elif order_type == 'stop' and stop_price:
                order_data['stop_price'] = str(stop_price)
            elif order_type == 'stop_limit' and limit_price and stop_price:
                order_data['limit_price'] = str(limit_price)
                order_data['stop_price'] = str(stop_price)
            
            response = requests.post(f"{self.base_url}/v2/orders", 
                                   headers=self.headers, 
                                   json=order_data)
            
            if response.status_code == 201:
                return response.json()
            else:
                print(f"Error placing order: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def cancel_order(self, order_id):
        """Cancel a specific order"""
        try:
            response = requests.delete(f"{self.base_url}/v2/orders/{order_id}", headers=self.headers)
            return response.status_code == 204
        except Exception as e:
            print(f"Error canceling order: {e}")
            return False
    
    def get_portfolio_history(self, period='1M', timeframe='1D'):
        """Get portfolio performance history"""
        try:
            params = {'period': period, 'timeframe': timeframe}
            response = requests.get(f"{self.base_url}/v2/account/portfolio/history", 
                                  headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting portfolio history: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_market_data(self, symbol):
        """Get latest market data for a symbol"""
        try:
            response = requests.get(f"{self.base_url}/v2/stocks/{symbol}/quotes/latest", 
                                  headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting market data: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None

def test_alpaca_integration():
    """Test Alpaca Markets API integration"""
    print("🚀 Testing Alpaca Markets API Integration")
    print("=" * 60)
    
    # Initialize client
    client = AlpacaTradingClient()
    
    # Test 1: Get Account Information
    print("\n📊 Account Information:")
    account = client.get_account()
    if account:
        print(f"   Account Status: {account.get('status', 'Unknown')}")
        print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"   Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"   Pattern Day Trader: {account.get('pattern_day_trader', False)}")
    else:
        print("   ❌ Could not retrieve account information")
        print("   Please check your API keys and try again")
        return False
    
    # Test 2: Get Current Positions
    print("\n📈 Current Positions:")
    positions = client.get_positions()
    if positions:
        for position in positions:
            print(f"   {position['symbol']}: {position['qty']} shares")
            print(f"   Current Value: ${float(position['market_value']):,.2f}")
            print(f"   Unrealized P&L: ${float(position['unrealized_pl']):,.2f}")
            print(f"   ---")
    else:
        print("   No current positions")
    
    # Test 3: Get Recent Orders
    print("\n📋 Recent Orders:")
    orders = client.get_orders(limit=10)
    if orders:
        for order in orders[-5:]:  # Show last 5 orders
            print(f"   {order['symbol']}: {order['side']} {order['qty']} shares")
            print(f"   Status: {order['status']}")
            print(f"   Order Type: {order['type']}")
            print(f"   Submitted: {order['submitted_at']}")
            print(f"   ---")
    else:
        print("   No recent orders")
    
    # Test 4: Get Market Data
    print("\n💰 Market Data Test:")
    test_symbols = ['AAPL', 'TSLA', 'NVDA']
    for symbol in test_symbols:
        market_data = client.get_market_data(symbol)
        if market_data:
            quote = market_data.get('quote', {})
            print(f"   {symbol}: ${quote.get('ap', 0):.2f} (Ask)")
            print(f"   Bid: ${quote.get('bp', 0):.2f}")
            print(f"   Timestamp: {quote.get('t', 'N/A')}")
        else:
            print(f"   {symbol}: Could not retrieve market data")
    
    # Test 5: Portfolio Performance
    print("\n📊 Portfolio Performance:")
    portfolio_history = client.get_portfolio_history()
    if portfolio_history:
        equity = portfolio_history.get('equity', [])
        if equity:
            initial_value = equity[0] if equity else 0
            current_value = equity[-1] if equity else 0
            total_return = ((current_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
            print(f"   Initial Value: ${initial_value:,.2f}")
            print(f"   Current Value: ${current_value:,.2f}")
            print(f"   Total Return: {total_return:+.2f}%")
        else:
            print("   No portfolio history available")
    else:
        print("   Could not retrieve portfolio history")
    
    # Test 6: Place a Demo Order (commented out for safety)
    print("\n🛡️  Demo Order (Safe Mode):")
    print("   Demo: Would place order for 1 share of AAPL")
    print("   Uncomment the code below to place actual trades")
    print("   Remember: This is paper trading - no real money at risk!")
    
    # Uncomment to place actual demo orders:
    # order_result = client.place_order(
    #     symbol='AAPL',
    #     qty=1,
    #     side='buy',
    #     order_type='market'
    # )
    # if order_result:
    #     print(f"   Order placed: {order_result['id']}")
    # else:
    #     print("   Failed to place order")
    
    print("\n" + "=" * 60)
    print("✅ Alpaca Markets API integration test completed!")
    print("\n🎯 Next Steps:")
    print("1. Sign up at https://alpaca.markets/")
    print("2. Get your API keys from the dashboard")
    print("3. Replace the placeholder keys in this script")
    print("4. Start with paper trading to test strategies")
    print("5. Integrate with your existing LSTM/RL models")
    print("\n💡 Benefits of adding Alpaca:")
    print("- Real portfolio management vs simulation")
    print("- Commission-free trading")
    print("- Professional-grade execution")
    print("- Real-time position tracking")
    print("- Advanced order types (stop loss, limit, etc.)")
    print("- Portfolio analytics and reporting")
    
    return True

if __name__ == "__main__":
    # Check if API keys are configured
    if ALPACA_API_KEY == "your_alpaca_api_key_here":
        print("🔑 Please configure your Alpaca API keys first!")
        print("\nTo get your API keys:")
        print("1. Sign up at https://alpaca.markets/")
        print("2. Go to your dashboard")
        print("3. Generate API keys")
        print("4. Replace the placeholder keys in this script")
        print("\nPaper trading is completely free and risk-free!")
    else:
        test_alpaca_integration() 