#!/usr/bin/env python3
"""
Live Trading System with LLM Integration

This module provides real-time trading capabilities with:
- LSTM price prediction
- Reinforcement Learning trading decisions
- LLM market sentiment analysis
- Risk management and portfolio tracking
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from openai import OpenAI

# Import our models
from models.lstm_model import LSTMPricePredictor
from models.rl_agent import TradingAgent, DQNAgent, TradingEnvironment
from features.build_features import build_dataset, build_features_from_data


class MarketDataProvider:
    """Provides real-time market data"""
    
    def __init__(self, symbols: List[str] = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']):
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)
    
    def get_real_time_data(self, symbol: str, period: str = '1d', interval: str = '1m') -> pd.DataFrame:
        """Get real-time price data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_news(self, symbol: str) -> List[Dict]:
        """Get market news (placeholder - integrate with news API)"""
        # This would integrate with a news API like Alpha Vantage, NewsAPI, etc.
        return [
            {"headline": f"Market update for {symbol}", "sentiment": "neutral", "timestamp": datetime.now()},
            {"headline": f"Analyst upgrade for {symbol}", "sentiment": "positive", "timestamp": datetime.now()}
        ]


class LLMMarketAnalyst:
    """LLM-powered market analysis and trading signals"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    async def analyze_market_sentiment(self, symbol: str, news_data: List[Dict], 
                                     price_data: pd.DataFrame) -> Dict:
        """Analyze market sentiment using LLM"""
        try:
            # Prepare context for LLM
            current_price = price_data['Close'].iloc[-1]
            price_change = ((current_price - price_data['Close'].iloc[-2]) / 
                           price_data['Close'].iloc[-2] * 100)
            
            news_summary = "\n".join([item['headline'] for item in news_data[:3]])
            
            prompt = f"""
            Analyze the market sentiment for {symbol}:
            
            Current Price: ${current_price:.2f}
            24h Change: {price_change:.2f}%
            Recent News:
            {news_summary}
            
            Provide a sentiment analysis with:
            1. Overall sentiment (BULLISH/BEARISH/NEUTRAL)
            2. Confidence level (1-10)
            3. Key factors influencing sentiment
            4. Risk assessment
            
            Respond in JSON format:
            {{
                "sentiment": "BULLISH/BEARISH/NEUTRAL",
                "confidence": 1-10,
                "key_factors": ["factor1", "factor2", "factor3"],
                "risk_level": "LOW/MEDIUM/HIGH",
                "reasoning": "Brief explanation"
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            # Parse response (simplified - in production, use proper JSON parsing)
            content = response.choices[0].message.content
            
            # Basic parsing (you'd want more robust parsing in production)
            if "BULLISH" in content.upper():
                sentiment = "BULLISH"
            elif "BEARISH" in content.upper():
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            return {
                "sentiment": sentiment,
                "confidence": 7,  # Default confidence
                "key_factors": ["Price momentum", "Market news", "Technical indicators"],
                "risk_level": "MEDIUM",
                "reasoning": content[:200],
                "raw_response": content
            }
            
        except Exception as e:
            self.logger.error(f"Error in LLM sentiment analysis: {e}")
            return {
                "sentiment": "NEUTRAL",
                "confidence": 5,
                "key_factors": ["Analysis unavailable"],
                "risk_level": "MEDIUM",
                "reasoning": "LLM analysis failed"
            }
    
    async def generate_trading_signal(self, symbol: str, technical_analysis: Dict,
                                    sentiment_analysis: Dict, rl_prediction: str) -> Dict:
        """Generate trading signal using LLM"""
        try:
            prompt = f"""
            Generate a trading signal for {symbol} based on:
            
            Technical Analysis: {technical_analysis}
            Sentiment Analysis: {sentiment_analysis['sentiment']} (Confidence: {sentiment_analysis['confidence']}/10)
            RL Model Prediction: {rl_prediction}
            
            Provide a trading recommendation with:
            1. Action (BUY/SELL/HOLD)
            2. Confidence level (1-10)
            3. Position size (0.0-1.0)
            4. Stop loss percentage
            5. Take profit percentage
            6. Reasoning
            
            Respond in JSON format:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence": 1-10,
                "position_size": 0.0-1.0,
                "stop_loss": 0.01-0.10,
                "take_profit": 0.02-0.20,
                "reasoning": "Brief explanation",
                "risk_reward_ratio": 1.0-5.0
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert trading advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250
            )
            
            content = response.choices[0].message.content
            
            # Basic parsing (improve in production)
            if "BUY" in content.upper():
                action = "BUY"
            elif "SELL" in content.upper():
                action = "SELL"
            else:
                action = "HOLD"
            
            return {
                "action": action,
                "confidence": 7,
                "position_size": 0.25,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "reasoning": content[:150],
                "risk_reward_ratio": 2.5,
                "raw_response": content
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return {
                "action": "HOLD",
                "confidence": 5,
                "position_size": 0.0,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "reasoning": "Signal generation failed"
            }


class LiveTradingSystem:
    """Main live trading system orchestrator"""
    
    def __init__(
        self,
        lstm_model: LSTMPricePredictor,
        trading_agent: TradingAgent,
        llm_analyst: LLMMarketAnalyst,
        symbols: List[str] = ['AAPL'],
        initial_balance: float = 10000.0
    ):
        self.lstm_model = lstm_model
        self.trading_agent = trading_agent
        self.llm_analyst = llm_analyst
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Portfolio tracking
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.trade_history = []
        self.performance_metrics = []
        
        # Market data provider
        self.market_data = MarketDataProvider(symbols)
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            # Simple moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
            
            return {
                'SMA_20': data['SMA_20'].iloc[-1],
                'SMA_50': data['SMA_50'].iloc[-1],
                'RSI': data['RSI'].iloc[-1],
                'MACD': data['MACD'].iloc[-1],
                'MACD_signal': data['MACD_signal'].iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    async def analyze_symbol(self, symbol: str) -> Dict:
        """Comprehensive analysis of a symbol"""
        try:
            # Get market data
            price_data = self.market_data.get_real_time_data(symbol, period='5d', interval='1h')
            if price_data.empty:
                return None
            
            # Get news data
            news_data = self.market_data.get_market_news(symbol)
            
            # Technical analysis
            technical_analysis = self.calculate_technical_indicators(price_data)
            
            # Prepare LSTM features
            features_data = build_features_from_data(price_data)
            lstm_input = features_data.values[-60:].reshape(1, 60, -1)  # Last 60 time steps
            
            # LSTM prediction
            with torch.no_grad():
                lstm_features = self.lstm_model.extract_features(torch.FloatTensor(lstm_input))
                lstm_features_np = lstm_features.numpy().flatten()
            
            # RL prediction
            env_state = {
                'balance': self.balance,
                'position': self.positions[symbol],
                'portfolio_value': self.balance + sum(self.positions.values()),
                'step_count': 0
            }
            
            rl_action, rl_action_name = self.trading_agent.predict_action(lstm_features_np, env_state)
            
            # LLM sentiment analysis
            sentiment_analysis = await self.llm_analyst.analyze_market_sentiment(
                symbol, news_data, price_data
            )
            
            # LLM trading signal
            trading_signal = await self.llm_analyst.generate_trading_signal(
                symbol, technical_analysis, sentiment_analysis, rl_action_name
            )
            
            return {
                'symbol': symbol,
                'price_data': price_data,
                'technical_analysis': technical_analysis,
                'sentiment_analysis': sentiment_analysis,
                'rl_prediction': rl_action_name,
                'llm_signal': trading_signal,
                'lstm_features': lstm_features_np,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol: str, action: str, quantity: float, price: float) -> bool:
        """Execute a trade (simulation)"""
        try:
            if action == 'BUY':
                cost = quantity * price * 1.001  # 0.1% transaction cost
                if cost <= self.balance:
                    self.balance -= cost
                    self.positions[symbol] += quantity
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'timestamp': datetime.now(),
                        'balance': self.balance
                    })
                    return True
            
            elif action == 'SELL':
                if quantity <= self.positions[symbol]:
                    proceeds = quantity * price * 0.999  # 0.1% transaction cost
                    self.balance += proceeds
                    self.positions[symbol] -= quantity
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'timestamp': datetime.now(),
                        'balance': self.balance
                    })
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    async def trading_loop(self, interval_minutes: int = 5):
        """Main trading loop"""
        self.logger.info("Starting live trading system...")
        
        while True:
            try:
                for symbol in self.symbols:
                    analysis = await self.analyze_symbol(symbol)
                    
                    if analysis:
                        # Log analysis
                        self.logger.info(f"""
                        Analysis for {symbol}:
                        - Current Price: ${analysis['price_data']['Close'].iloc[-1]:.2f}
                        - Technical: {analysis['technical_analysis']}
                        - Sentiment: {analysis['sentiment_analysis']['sentiment']} 
                          (Confidence: {analysis['sentiment_analysis']['confidence']}/10)
                        - RL Prediction: {analysis['rl_prediction']}
                        - LLM Signal: {analysis['llm_signal']['action']} 
                          (Confidence: {analysis['llm_signal']['confidence']}/10)
                        """)
                        
                        # Make trading decision
                        llm_signal = analysis['llm_signal']
                        current_price = analysis['price_data']['Close'].iloc[-1]
                        
                        if llm_signal['confidence'] >= 7:  # High confidence threshold
                            if llm_signal['action'] == 'BUY':
                                quantity = (self.balance * llm_signal['position_size']) / current_price
                                if self.execute_trade(symbol, 'BUY', quantity, current_price):
                                    self.logger.info(f"Executed BUY: {quantity:.4f} {symbol} @ ${current_price:.2f}")
                            
                            elif llm_signal['action'] == 'SELL' and self.positions[symbol] > 0:
                                quantity = self.positions[symbol] * llm_signal['position_size']
                                if self.execute_trade(symbol, 'SELL', quantity, current_price):
                                    self.logger.info(f"Executed SELL: {quantity:.4f} {symbol} @ ${current_price:.2f}")
                        
                        # Update performance metrics
                        portfolio_value = self.balance + sum(
                            pos * current_price for pos in self.positions.values()
                        )
                        
                        self.performance_metrics.append({
                            'timestamp': datetime.now(),
                            'portfolio_value': portfolio_value,
                            'balance': self.balance,
                            'positions': self.positions.copy(),
                            'return': (portfolio_value - self.initial_balance) / self.initial_balance * 100
                        })
                
                # Wait for next iteration
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def get_performance_summary(self) -> Dict:
        """Get trading performance summary"""
        if not self.performance_metrics:
            return {}
        
        latest = self.performance_metrics[-1]
        total_return = latest['return']
        
        returns = [m['return'] for m in self.performance_metrics[-100:]]  # Last 100 data points
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        return {
            'total_return': total_return,
            'current_balance': self.balance,
            'portfolio_value': latest['portfolio_value'],
            'num_trades': len(self.trade_history),
            'volatility': volatility,
            'sharpe_ratio': total_return / volatility if volatility > 0 else 0,
            'positions': self.positions
        }


# Demo setup function
async def run_live_trading_demo():
    """
    Run a live trading demonstration
    """
    # Initialize models (you'll need to load pre-trained models)
    lstm_model = LSTMPricePredictor()
    dqn_agent = DQNAgent()
    environment = TradingEnvironment()
    trading_agent = TradingAgent(lstm_model, dqn_agent, environment)
    
    # Initialize LLM analyst (you'll need to provide your API key)
    llm_analyst = LLMMarketAnalyst(api_key="your-openai-api-key-here")
    
    # Initialize trading system
    trading_system = LiveTradingSystem(
        lstm_model=lstm_model,
        trading_agent=trading_agent,
        llm_analyst=llm_analyst,
        symbols=['AAPL', 'GOOGL'],
        initial_balance=10000.0
    )
    
    # Run trading loop
    await trading_system.trading_loop(interval_minutes=5)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the demo
    asyncio.run(run_live_trading_demo()) 