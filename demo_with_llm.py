#!/usr/bin/env python3
"""
🚀 Enhanced RL-LSTM AI Trading Demo with Real LLM Integration

This script demonstrates the complete trading system with:
- LSTM price prediction
- Reinforcement Learning trading decisions  
- REAL LLM market analysis using OpenAI GPT
- Live market data integration

This version uses your actual OpenAI API key for real AI analysis!
"""

import sys
import os
sys.path.append('src')

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import asyncio

# Import our modules
from models.lstm_model import LSTMPricePredictor, create_lstm_model
from models.rl_agent import TradingAgent, DQNAgent, TradingEnvironment
from trading.live_trading import LLMMarketAnalyst, LiveTradingSystem, MarketDataProvider
from data.fetch_data import get_stock_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_enhanced_demo():
    """Run the enhanced demo with real LLM integration"""
    print("🚀 Enhanced RL-LSTM AI Trading Demo with Real LLM")
    print("=" * 60)
    
    # Configuration
    symbol = 'AAPL'
    initial_balance = 10000.0
    
    model_config = {
        'input_size': 10,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 60
    }
    
    print(f"📊 Analyzing {symbol} with REAL AI...")
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    
    try:
        # 1. Initialize LLM Analyst with your API key
        print("\n1️⃣ Initializing LLM Market Analyst...")
        llm_analyst = LLMMarketAnalyst(
            api_key=os.environ['OPENAI_API_KEY'],
            model="gpt-3.5-turbo"  # Using GPT-3.5 for cost efficiency
        )
        print("   ✅ LLM Analyst ready!")
        
        # 2. Load market data
        print("\n2️⃣ Loading live market data...")
        market_provider = MarketDataProvider()
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        recent_start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        historical_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        recent_data = get_stock_data(symbol, recent_start, end_date)
        historical_data = get_stock_data(symbol, historical_start, end_date)
        
        current_price = recent_data['Close'].iloc[-1]
        price_change_24h = ((current_price - recent_data['Close'].iloc[-2]) / 
                           recent_data['Close'].iloc[-2] * 100)
        
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   24h Change: {price_change_24h:+.2f}%")
        print(f"   Volume: {recent_data['Volume'].iloc[-1]:,}")
        
        # 3. Initialize AI models
        print("\n3️⃣ Initializing AI models...")
        lstm_model = create_lstm_model(model_config)
        environment = TradingEnvironment(initial_balance=initial_balance)
        dqn_agent = DQNAgent()
        trading_agent = TradingAgent(lstm_model, dqn_agent, environment)
        
        print(f"   LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
        print(f"   DQN parameters: {sum(p.numel() for p in dqn_agent.parameters()):,}")
        
        # 4. Feature engineering
        print("\n4️⃣ Building technical features...")
        features_data = historical_data.copy()
        
        # Add technical indicators
        features_data['MA_5'] = features_data['Close'].rolling(window=5).mean()
        features_data['MA_20'] = features_data['Close'].rolling(window=20).mean()
        
        # RSI
        delta = features_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Price change and volume ratio
        features_data['Price_Change'] = features_data['Close'].pct_change()
        features_data['Volume_Ratio'] = features_data['Volume'] / features_data['Volume'].rolling(window=20).mean()
        
        # Bollinger Bands
        bb_window = 20
        bb_std = features_data['Close'].rolling(window=bb_window).std()
        bb_mean = features_data['Close'].rolling(window=bb_window).mean()
        features_data['BB_Upper'] = bb_mean + (bb_std * 2)
        features_data['BB_Lower'] = bb_mean - (bb_std * 2)
        
        # Select features and clean data
        feature_columns = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Price_Change', 
                          'Volume_Ratio', 'BB_Upper', 'BB_Lower', 'Open']
        features_data = features_data[feature_columns].dropna()
        
        print(f"   Feature matrix: {features_data.shape}")
        
        # 5. LSTM Analysis
        print("\n5️⃣ Running LSTM price prediction...")
        sequence_length = model_config['sequence_length']
        
        if len(features_data) >= sequence_length:
            last_sequence = features_data.iloc[-sequence_length:].values
            lstm_input = torch.FloatTensor(last_sequence).unsqueeze(0)
            
            with torch.no_grad():
                lstm_prediction, lstm_features = lstm_model(lstm_input)
                lstm_features_np = lstm_features.numpy().flatten()
            
            predicted_price = lstm_prediction.item()
            price_change_pred = ((predicted_price - current_price) / current_price * 100)
            
            print(f"   LSTM Prediction: ${predicted_price:.2f} ({price_change_pred:+.1f}%)")
        else:
            print(f"   ⚠️ Using random features (insufficient data)")
            lstm_features_np = np.random.randn(128)
            predicted_price = current_price * (1 + np.random.uniform(-0.02, 0.02))
        
        # 6. RL Agent Analysis
        print("\n6️⃣ Running RL trading agent...")
        env_state = {
            'balance': initial_balance,
            'position': 0.0,
            'portfolio_value': initial_balance,
            'step_count': 0
        }
        
        rl_action, rl_action_name = trading_agent.predict_action(lstm_features_np, env_state)
        print(f"   RL Recommendation: {rl_action_name} (Action: {rl_action})")
        
        # 7. REAL LLM Market Analysis
        print("\n7️⃣ Running REAL LLM market analysis...")
        print("   🧠 Querying OpenAI GPT for market sentiment...")
        
        # Prepare data for LLM analysis
        market_data = {
            'symbol': symbol,
            'current_price': current_price,
            'price_change': price_change_24h,
            'volume': int(recent_data['Volume'].iloc[-1]),
            'predicted_price': predicted_price,
            'rl_recommendation': rl_action_name,
            'technical_indicators': {
                'rsi': float(features_data['RSI'].iloc[-1]) if not pd.isna(features_data['RSI'].iloc[-1]) else 50,
                'ma_5': float(features_data['MA_5'].iloc[-1]) if not pd.isna(features_data['MA_5'].iloc[-1]) else current_price,
                'ma_20': float(features_data['MA_20'].iloc[-1]) if not pd.isna(features_data['MA_20'].iloc[-1]) else current_price,
                'bb_upper': float(features_data['BB_Upper'].iloc[-1]) if not pd.isna(features_data['BB_Upper'].iloc[-1]) else current_price * 1.02,
                'bb_lower': float(features_data['BB_Lower'].iloc[-1]) if not pd.isna(features_data['BB_Lower'].iloc[-1]) else current_price * 0.98
            }
        }
        
        # Get real LLM analysis
        llm_analysis = await llm_analyst.analyze_market_sentiment(market_data)
        
        print(f"   ✅ LLM Analysis Complete!")
        print(f"   Sentiment: {llm_analysis.get('sentiment', 'NEUTRAL')}")
        print(f"   Confidence: {llm_analysis.get('confidence', 5)}/10")
        print(f"   Key Factors: {', '.join(llm_analysis.get('key_factors', ['Technical analysis']))}")
        
        # 8. Generate Final Trading Signal
        print("\n8️⃣ Generating final trading recommendation...")
        
        # Combine all AI insights
        sentiment_score = llm_analysis.get('confidence', 5)
        if llm_analysis.get('sentiment') == 'BULLISH':
            sentiment_weight = 1
        elif llm_analysis.get('sentiment') == 'BEARISH':
            sentiment_weight = -1
        else:
            sentiment_weight = 0
        
        # RL action weight
        rl_weight = 1 if rl_action == 1 else (-1 if rl_action == 2 else 0)
        
        # LSTM prediction weight
        lstm_weight = 1 if predicted_price > current_price else -1
        
        # Combined signal
        combined_signal = (sentiment_weight * 0.4 + rl_weight * 0.4 + lstm_weight * 0.2)
        
        if combined_signal > 0.3:
            final_action = "BUY"
            confidence = min(10, int(8 + combined_signal * 2))
        elif combined_signal < -0.3:
            final_action = "SELL"
            confidence = min(10, int(8 + abs(combined_signal) * 2))
        else:
            final_action = "HOLD"
            confidence = 6
        
        # 9. Display Results
        print("\n" + "="*60)
        print("🧠 FINAL AI TRADING ANALYSIS")
        print("="*60)
        print(f"📊 Symbol: {symbol}")
        print(f"💰 Current Price: ${current_price:.2f} ({price_change_24h:+.2f}%)")
        print(f"🔮 LSTM Prediction: ${predicted_price:.2f}")
        print(f"🤖 RL Recommendation: {rl_action_name}")
        print(f"💬 LLM Sentiment: {llm_analysis.get('sentiment', 'NEUTRAL')} ({llm_analysis.get('confidence', 5)}/10)")
        print(f"🎯 FINAL DECISION: {final_action} with {confidence}/10 confidence")
        print("="*60)
        
        # 10. Risk Management
        if final_action != "HOLD":
            position_size = min(0.25, confidence / 40)  # Max 25% position
            stop_loss = 0.02  # 2% stop loss
            take_profit = 0.05  # 5% take profit
            
            print(f"⚖️ Position Size: {position_size*100:.1f}% of portfolio")
            print(f"🛡️ Stop Loss: {stop_loss*100:.1f}%")
            print(f"🎯 Take Profit: {take_profit*100:.1f}%")
        
        # 11. LLM Reasoning
        if 'reasoning' in llm_analysis:
            print(f"\n💭 LLM Reasoning:")
            print(f"   {llm_analysis['reasoning']}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"💡 This analysis used REAL AI from OpenAI GPT!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    try:
        # Run the async demo
        result = asyncio.run(run_enhanced_demo())
        return result
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 