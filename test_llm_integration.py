#!/usr/bin/env python3
"""
Test LLM Integration with OpenAI API
This script tests the LLM market analyst functionality
"""

import os
import sys
sys.path.append('src')

# Set the API key as environment variable
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'

def test_openai_connection():
    """Test basic OpenAI API connection"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello from your AI trading system!' in exactly 5 words."}
            ],
            max_tokens=20
        )
        
        print("✅ OpenAI API Connection Successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return False

def test_llm_market_analyst():
    """Test the LLM Market Analyst class"""
    try:
        from trading.live_trading import LLMMarketAnalyst
        
        # Initialize with your API key
        analyst = LLMMarketAnalyst(
            api_key='your-openai-api-key-here',
            model="gpt-3.5-turbo"  # Use cheaper model for testing
        )
        
        # Test market analysis
        test_data = {
            'symbol': 'AAPL',
            'current_price': 150.00,
            'price_change': -2.5,
            'volume': 50000000,
            'technical_indicators': {
                'rsi': 45,
                'macd': -0.5,
                'moving_average': 152.0
            }
        }
        
        print("\n🧠 Testing LLM Market Analysis...")
        analysis = analyst.analyze_market_sentiment(test_data)
        
        print("✅ LLM Market Analyst Working!")
        print(f"Sentiment: {analysis.get('sentiment', 'N/A')}")
        print(f"Confidence: {analysis.get('confidence', 'N/A')}")
        print(f"Reasoning: {analysis.get('reasoning', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Market Analyst Error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing LLM Integration for RL-LSTM Trading System")
    print("=" * 60)
    
    # Test 1: Basic OpenAI connection
    print("\n1️⃣ Testing OpenAI API Connection...")
    api_works = test_openai_connection()
    
    if not api_works:
        print("\n❌ OpenAI API test failed. Please check your API key.")
        return False
    
    # Test 2: LLM Market Analyst
    print("\n2️⃣ Testing LLM Market Analyst...")
    analyst_works = test_llm_market_analyst()
    
    if analyst_works:
        print("\n🎉 All LLM tests passed!")
        print("\n🚀 Your system is ready for live trading with LLM integration!")
        print("\nNext steps:")
        print("1. Run: python demo_trading_system.py")
        print("2. Or run the live trading demo with LLM analysis")
        return True
    else:
        print("\n❌ LLM Market Analyst test failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 