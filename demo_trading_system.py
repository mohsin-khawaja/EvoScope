#!/usr/bin/env python3
"""
🚀 RL-LSTM AI Trading Agent Demo

This script demonstrates the complete trading system with:
- LSTM price prediction
- Reinforcement Learning trading decisions
- LLM market analysis (simulated)
- Live market data integration

Run this to see the AI trading agent in action!
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Import our modules
from models.lstm_model import LSTMPricePredictor, create_lstm_model
from models.rl_agent import TradingAgent, DQNAgent, TradingEnvironment
from features.build_features import build_dataset
from data.fetch_data import get_stock_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_llm_analysis(symbol, price_data):
    """Simulate LLM market analysis"""
    current_price = price_data['Close'].iloc[-1]
    price_change = ((current_price - price_data['Close'].iloc[-2]) / price_data['Close'].iloc[-2] * 100)
    
    if price_change > 1:
        sentiment = "BULLISH"
        confidence = 8
    elif price_change < -1:
        sentiment = "BEARISH"
        confidence = 7
    else:
        sentiment = "NEUTRAL"
        confidence = 6
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "key_factors": ["Price momentum", "Volume analysis", "Technical indicators"],
        "price_prediction": "UP" if price_change > 0 else "DOWN",
        "risk_level": "MEDIUM",
        "reasoning": f"Based on {price_change:.1f}% price movement"
    }

def simulate_trading_signal(sentiment_analysis, rl_prediction):
    """Simulate LLM trading signal generation"""
    if sentiment_analysis['sentiment'] == 'BULLISH' and rl_prediction == 'BUY':
        action = 'BUY'
        confidence = 9
    elif sentiment_analysis['sentiment'] == 'BEARISH' and rl_prediction == 'SELL':
        action = 'SELL'
        confidence = 8
    else:
        action = 'HOLD'
        confidence = 6
    
    return {
        "action": action,
        "confidence": confidence,
        "position_size": 0.25,
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "reasoning": f"Combined analysis suggests {action}",
        "risk_reward_ratio": 2.5
    }

def main():
    """Main demo function"""
    print("🚀 RL-LSTM AI Trading Agent Demo")
    print("=" * 50)
    
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
    
    print(f"📊 Analyzing {symbol}...")
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    
    try:
        # 1. Load market data
        print("\n1️⃣ Loading market data...")
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
        
        # 2. Initialize models
        print("\n2️⃣ Initializing AI models...")
        lstm_model = create_lstm_model(model_config)
        environment = TradingEnvironment(initial_balance=initial_balance)
        dqn_agent = DQNAgent()
        trading_agent = TradingAgent(lstm_model, dqn_agent, environment)
        
        print(f"   LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
        print(f"   DQN parameters: {sum(p.numel() for p in dqn_agent.parameters()):,}")
        
        # 3. Feature engineering
        print("\n3️⃣ Building features...")
        
        # Create features from our historical data
        features_data = historical_data.copy()
        
        # Add technical indicators
        # Simple moving averages
        features_data['MA_5'] = features_data['Close'].rolling(window=5).mean()
        features_data['MA_20'] = features_data['Close'].rolling(window=20).mean()
        
        # RSI
        delta = features_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Price change
        features_data['Price_Change'] = features_data['Close'].pct_change()
        
        # Volume ratio
        features_data['Volume_Ratio'] = features_data['Volume'] / features_data['Volume'].rolling(window=20).mean()
        
        # Bollinger Bands
        bb_window = 20
        bb_std = features_data['Close'].rolling(window=bb_window).std()
        bb_mean = features_data['Close'].rolling(window=bb_window).mean()
        features_data['BB_Upper'] = bb_mean + (bb_std * 2)
        features_data['BB_Lower'] = bb_mean - (bb_std * 2)
        
        # Select final features and drop NaN
        feature_columns = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Price_Change', 
                          'Volume_Ratio', 'BB_Upper', 'BB_Lower', 'Open']
        features_data = features_data[feature_columns].dropna()
        
        print(f"   Feature matrix shape: {features_data.shape}")
        print(f"   Features: {list(features_data.columns)}")
        
        # 4. LSTM analysis
        print("\n4️⃣ Running LSTM analysis...")
        sequence_length = model_config['sequence_length']
        
        if len(features_data) >= sequence_length:
            last_sequence = features_data.iloc[-sequence_length:].values
            lstm_input = torch.FloatTensor(last_sequence).unsqueeze(0)
            
            with torch.no_grad():
                lstm_prediction, lstm_features = lstm_model(lstm_input)
                lstm_features_np = lstm_features.numpy().flatten()
            
            print(f"   LSTM Price Prediction: ${lstm_prediction.item():.2f}")
            print(f"   Feature vector size: {len(lstm_features_np)}")
        else:
            print(f"   ⚠️ Not enough data for LSTM (need {sequence_length}, have {len(features_data)})")
            lstm_features_np = np.random.randn(128)
            lstm_prediction = torch.tensor([current_price])
        
        # 5. RL agent analysis
        print("\n5️⃣ Running RL agent analysis...")
        env_state = {
            'balance': initial_balance,
            'position': 0.0,
            'portfolio_value': initial_balance,
            'step_count': 0
        }
        
        rl_action, rl_action_name = trading_agent.predict_action(lstm_features_np, env_state)
        
        # Get Q-values for visualization
        state_vector = trading_agent.prepare_state(lstm_features_np, env_state)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        with torch.no_grad():
            q_values = dqn_agent(state_tensor).numpy().flatten()
        
        print(f"   RL Recommendation: {rl_action_name}")
        print(f"   Q-values: HOLD={q_values[0]:.3f}, BUY={q_values[1]:.3f}, SELL={q_values[2]:.3f}")
        
        # 6. LLM analysis (simulated)
        print("\n6️⃣ Running market sentiment analysis...")
        sentiment_analysis = simulate_llm_analysis(symbol, recent_data)
        trading_signal = simulate_trading_signal(sentiment_analysis, rl_action_name)
        
        print(f"   Market Sentiment: {sentiment_analysis['sentiment']} ({sentiment_analysis['confidence']}/10)")
        print(f"   Key Factors: {', '.join(sentiment_analysis['key_factors'])}")
        
        # 7. Final recommendation
        print("\n🎯 FINAL TRADING RECOMMENDATION")
        print("=" * 40)
        print(f"Action: {trading_signal['action']}")
        print(f"Confidence: {trading_signal['confidence']}/10")
        print(f"Position Size: {trading_signal['position_size']*100:.1f}% of portfolio")
        print(f"Stop Loss: {trading_signal['stop_loss']*100:.1f}%")
        print(f"Take Profit: {trading_signal['take_profit']*100:.1f}%")
        print(f"Risk/Reward Ratio: {trading_signal['risk_reward_ratio']:.1f}")
        print(f"Reasoning: {trading_signal['reasoning']}")
        
        # 8. Trade simulation
        if trading_signal['action'] == 'BUY' and trading_signal['confidence'] >= 7:
            position_value = initial_balance * trading_signal['position_size']
            shares = position_value / current_price
            stop_loss_price = current_price * (1 - trading_signal['stop_loss'])
            take_profit_price = current_price * (1 + trading_signal['take_profit'])
            
            print(f"\n📋 SIMULATED TRADE DETAILS:")
            print(f"   Shares to buy: {shares:.2f}")
            print(f"   Investment: ${position_value:.2f}")
            print(f"   Entry price: ${current_price:.2f}")
            print(f"   Stop loss: ${stop_loss_price:.2f}")
            print(f"   Take profit: ${take_profit_price:.2f}")
            print(f"   Max loss: ${(current_price - stop_loss_price) * shares:.2f}")
            print(f"   Max profit: ${(take_profit_price - current_price) * shares:.2f}")
        
        # 9. Create visualization
        print("\n📊 Generating analysis chart...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price chart
        ax1.plot(recent_data.index, recent_data['Close'], 'b-', linewidth=2, label='Close Price')
        ax1.fill_between(recent_data.index, recent_data['Low'], recent_data['High'], alpha=0.3)
        ax1.set_title(f'{symbol} - Recent Price Action', fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-values
        actions = ['HOLD', 'BUY', 'SELL']
        colors = ['gray', 'green', 'red']
        bars = ax2.bar(actions, q_values, color=colors, alpha=0.7)
        bars[rl_action].set_alpha(1.0)
        bars[rl_action].set_edgecolor('black')
        bars[rl_action].set_linewidth(2)
        ax2.set_title('RL Agent Q-Values', fontweight='bold')
        ax2.set_ylabel('Q-Value')
        ax2.grid(True, alpha=0.3)
        
        # Sentiment
        sentiment_score = (sentiment_analysis['confidence'] if sentiment_analysis['sentiment'] == 'BULLISH' 
                          else -sentiment_analysis['confidence'] if sentiment_analysis['sentiment'] == 'BEARISH' 
                          else 0)
        color = 'green' if sentiment_score > 0 else 'red' if sentiment_score < 0 else 'gray'
        ax3.barh([0], [sentiment_score], color=color, alpha=0.7)
        ax3.set_xlim(-10, 10)
        ax3.set_title('Market Sentiment Score', fontweight='bold')
        ax3.set_xlabel('Sentiment Score')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Trading signal
        signal_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}[trading_signal['action']]
        ax4.bar([trading_signal['action']], [trading_signal['confidence']], 
                color=signal_color, alpha=0.7)
        ax4.set_ylim(0, 10)
        ax4.set_title('Trading Signal Confidence', fontweight='bold')
        ax4.set_ylabel('Confidence (1-10)')
        
        plt.suptitle(f'🤖 AI Trading Analysis for {symbol}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('trading_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Analysis complete! Chart saved as 'trading_analysis.png'")
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"❌ Demo failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("⚠️  This is for demonstration only - not for actual trading!")
    print("🚀 Ready to train models and implement live trading!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    main() 