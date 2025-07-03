#!/usr/bin/env python3
"""
🚀 Smart RL-LSTM AI Trading Demo with LLM Fallback

This script demonstrates the complete trading system with:
- LSTM price prediction
- Reinforcement Learning trading decisions  
- LLM market analysis (real OpenAI or intelligent fallback)
- Live market data integration

Your API key is configured - just add billing to OpenAI for real LLM analysis!
"""

import sys
import os
sys.path.append('src')

# Set your OpenAI API key (replace with your actual key or set as environment variable)
# os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'  
# Or set it in your environment: export OPENAI_API_KEY='your-key-here'
if 'OPENAI_API_KEY' not in os.environ:
    print("⚠️ Please set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("   Or uncomment and add your key in the script")

# Alpha Vantage API key (already configured)
ALPHA_VANTAGE_API_KEY = "38RX2Y3EUK2CV7Y8"

# NewsAPI key (already configured)
NEWSAPI_KEY = "1d5bb349-6f72-4f83-860b-9c2fb3c220bd"

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
from data.fetch_data import get_stock_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartLLMAnalyst:
    """Smart LLM Analyst with fallback to intelligent simulation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.use_real_llm = False
        
        # Try to initialize real LLM
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            
            # Test with a minimal request
            test_response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            self.use_real_llm = True
            print("✅ Real OpenAI LLM connected!")
            
        except Exception as e:
            print(f"⚠️ OpenAI API unavailable ({str(e)[:50]}...), using intelligent fallback")
            self.use_real_llm = False
    
    async def analyze_market_sentiment(self, market_data):
        """Analyze market sentiment with real LLM or intelligent fallback"""
        
        if self.use_real_llm:
            return await self._real_llm_analysis(market_data)
        else:
            return self._intelligent_fallback_analysis(market_data)
    
    async def _real_llm_analysis(self, market_data):
        """Real OpenAI LLM analysis"""
        try:
            prompt = f"""
            Analyze this stock market data and provide trading sentiment:
            
            Symbol: {market_data['symbol']}
            Current Price: ${market_data['current_price']:.2f}
            24h Change: {market_data['price_change']:.2f}%
            Volume: {market_data['volume']:,}
            RSI: {market_data['technical_indicators']['rsi']:.1f}
            MA5: ${market_data['technical_indicators']['ma_5']:.2f}
            MA20: ${market_data['technical_indicators']['ma_20']:.2f}
            
            Provide analysis in this JSON format:
            {{
                "sentiment": "BULLISH/BEARISH/NEUTRAL",
                "confidence": 1-10,
                "key_factors": ["factor1", "factor2", "factor3"],
                "reasoning": "Brief explanation"
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            # Parse response (simplified)
            content = response.choices[0].message.content
            
            # Extract sentiment and confidence (basic parsing)
            if "BULLISH" in content.upper():
                sentiment = "BULLISH"
            elif "BEARISH" in content.upper():
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            return {
                "sentiment": sentiment,
                "confidence": 8,
                "key_factors": ["Technical analysis", "Price momentum", "Volume analysis"],
                "reasoning": content[:200]
            }
            
        except Exception as e:
            print(f"⚠️ LLM API error, falling back: {e}")
            return self._intelligent_fallback_analysis(market_data)
    
    def _intelligent_fallback_analysis(self, market_data):
        """Intelligent fallback analysis based on technical indicators"""
        
        price_change = market_data['price_change']
        rsi = market_data['technical_indicators']['rsi']
        ma_5 = market_data['technical_indicators']['ma_5']
        ma_20 = market_data['technical_indicators']['ma_20']
        current_price = market_data['current_price']
        
        # Analyze technical indicators
        signals = []
        bullish_score = 0
        bearish_score = 0
        
        # Price momentum
        if price_change > 2:
            bullish_score += 2
            signals.append("Strong positive momentum")
        elif price_change < -2:
            bearish_score += 2
            signals.append("Strong negative momentum")
        elif price_change > 0:
            bullish_score += 1
            signals.append("Positive momentum")
        else:
            bearish_score += 1
            signals.append("Negative momentum")
        
        # RSI analysis
        if rsi > 70:
            bearish_score += 1
            signals.append("Overbought conditions (RSI)")
        elif rsi < 30:
            bullish_score += 1
            signals.append("Oversold conditions (RSI)")
        elif 40 <= rsi <= 60:
            signals.append("Neutral RSI")
        
        # Moving average analysis
        if current_price > ma_5 > ma_20:
            bullish_score += 2
            signals.append("Price above moving averages")
        elif current_price < ma_5 < ma_20:
            bearish_score += 2
            signals.append("Price below moving averages")
        elif current_price > ma_20:
            bullish_score += 1
            signals.append("Price above long-term average")
        
        # Volume analysis
        volume = market_data['volume']
        if volume > 50000000:  # High volume
            if price_change > 0:
                bullish_score += 1
                signals.append("High volume with price increase")
            else:
                bearish_score += 1
                signals.append("High volume with price decrease")
        
        # Determine sentiment
        if bullish_score > bearish_score + 1:
            sentiment = "BULLISH"
            confidence = min(9, 6 + bullish_score - bearish_score)
        elif bearish_score > bullish_score + 1:
            sentiment = "BEARISH"
            confidence = min(9, 6 + bearish_score - bullish_score)
        else:
            sentiment = "NEUTRAL"
            confidence = 5
        
        # Generate reasoning
        reasoning = f"Technical analysis shows {sentiment.lower()} signals. " + \
                   f"Key factors: {', '.join(signals[:3])}. " + \
                   f"Price momentum: {price_change:+.1f}%, RSI: {rsi:.1f}"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "key_factors": signals[:3],
            "reasoning": reasoning
        }

async def run_smart_demo():
    """Run the smart demo with LLM fallback"""
    print("🚀 Smart RL-LSTM AI Trading Demo")
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
        # 1. Initialize Smart LLM Analyst
        print("\n1️⃣ Initializing Smart LLM Analyst...")
        llm_analyst = SmartLLMAnalyst(
            api_key=os.environ['OPENAI_API_KEY'],
            model="gpt-3.5-turbo"
        )
        
        # 2. Load market data
        print("\n2️⃣ Loading live market data...")
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
        
        # 7. Smart LLM Market Analysis
        print("\n7️⃣ Running Smart LLM market analysis...")
        
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
        
        # Get LLM analysis (real or fallback)
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
            print(f"\n💭 AI Reasoning:")
            print(f"   {llm_analysis['reasoning']}")
        
        # 12. Next Steps
        print(f"\n🎉 Demo completed successfully!")
        if llm_analyst.use_real_llm:
            print(f"💡 This analysis used REAL OpenAI GPT!")
        else:
            print(f"💡 This used intelligent technical analysis fallback")
            print(f"🔑 Add billing to OpenAI account for real LLM analysis")
        
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
        result = asyncio.run(run_smart_demo())
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