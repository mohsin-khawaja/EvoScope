#!/usr/bin/env python3
"""
🚀 Alpaca Tracker Setup Guide

This script helps you set up comprehensive tracking for your Alpaca paper trading account.
"""

import os
from pathlib import Path

def create_env_file():
    """Create or update .env file with Alpaca keys"""
    env_path = Path(".env")
    
    print("🔧 Setting up your .env file...")
    
    # Read existing .env if it exists
    existing_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    existing_vars[key] = value
    
    # Get Alpaca keys
    print("\n📋 Alpaca API Configuration:")
    print("You already have an API key: PKH6HJ2RBVZ20P8EJPNT")
    print("Now you need to get your SECRET key from Alpaca:")
    print("1. Go to https://alpaca.markets/")
    print("2. Log into your account")
    print("3. Go to Paper Trading → API Keys")
    print("4. Copy your SECRET key")
    
    # Update environment variables
    env_vars = {
        'ALPACA_API_KEY': 'PKH6HJ2RBVZ20P8EJPNT',
        'ALPACA_SECRET_KEY': 'your_alpaca_secret_key_here',
        'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets/v2',
        'ALPHAVANTAGE_KEY': existing_vars.get('ALPHAVANTAGE_KEY', 'your_alpha_vantage_key'),
        'NEWSAPI_KEY': existing_vars.get('NEWSAPI_KEY', 'your_newsapi_key'),
        'OPENAI_API_KEY': existing_vars.get('OPENAI_API_KEY', 'your_openai_api_key'),
        'FRED_API_KEY': existing_vars.get('FRED_API_KEY', '56JBx7QuGHquzDi6yzMd'),
    }
    
    # Write .env file
    with open(env_path, 'w') as f:
        f.write("# RL-LSTM AI Trading Agent Environment Variables\n")
        f.write("# Paper Trading Configuration\n\n")
        
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"\n✅ .env file created at {env_path}")
    print("⚠️  IMPORTANT: Update ALPACA_SECRET_KEY with your actual secret key!")
    
    return env_path

def test_connection():
    """Test the Alpaca connection"""
    print("\n🔗 Testing Alpaca Connection...")
    
    try:
        # Import after .env is created
        from src.trading.alpaca_tracker import AlpacaTracker
        
        tracker = AlpacaTracker()
        account_info = tracker.get_account_info()
        
        if account_info:
            print("✅ Connection successful!")
            print(f"   Account Status: {account_info.get('status', 'Unknown')}")
            print(f"   Portfolio Value: ${float(account_info.get('portfolio_value', 0)):,.2f}")
            return True
        else:
            print("❌ Connection failed!")
            print("   Please check your ALPACA_SECRET_KEY in the .env file")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_tracking_integration():
    """Create integration with the AI trading system"""
    integration_code = '''
# Add this to your AI trading system to enable tracking

from src.trading.alpaca_tracker import AlpacaTracker

class EnhancedTradingSystem:
    def __init__(self):
        # Initialize your existing components
        self.lstm_model = create_lstm_model()
        self.rl_agent = create_rl_agent()
        
        # Add Alpaca tracker
        self.tracker = AlpacaTracker()
        
    def execute_ai_trade(self, symbol, action, quantity, confidence, reasoning):
        """Execute trade and track with AI metadata"""
        
        # Place order through Alpaca
        order_data = {
            'symbol': symbol,
            'qty': quantity,
            'side': action.lower(),
            'type': 'market',
            'time_in_force': 'gtc'
        }
        
        # Execute trade (you'd use actual Alpaca API here)
        # order_result = self.place_alpaca_order(order_data)
        
        # Track the trade with AI metadata
        self.tracker.record_trade(
            order_data=order_data,
            ai_confidence=confidence,
            ai_reasoning=reasoning
        )
        
        # Record portfolio snapshot
        self.tracker.record_portfolio_snapshot()
        
        # Check for risk alerts
        self.tracker.check_risk_alerts()
        
        return order_result
    
    def daily_trading_routine(self):
        """Daily routine with tracking"""
        
        # Run your AI analysis
        ai_signals = self.generate_ai_signals()
        
        # Execute trades
        for signal in ai_signals:
            if signal['confidence'] > 0.7:  # High confidence threshold
                self.execute_ai_trade(
                    symbol=signal['symbol'],
                    action=signal['action'],
                    quantity=signal['quantity'],
                    confidence=signal['confidence'],
                    reasoning=signal['reasoning']
                )
        
        # Generate daily report
        report = self.tracker.generate_daily_report()
        print(report)
        
        # Create performance chart
        self.tracker.create_performance_chart()
        
        return report

# Usage example:
# trading_system = EnhancedTradingSystem()
# daily_report = trading_system.daily_trading_routine()
'''
    
    with open('ai_trading_integration.py', 'w') as f:
        f.write(integration_code)
    
    print("\n🤖 AI Trading Integration created!")
    print("   File: ai_trading_integration.py")
    print("   This shows how to integrate tracking with your AI system")

def create_daily_monitoring_script():
    """Create a daily monitoring script"""
    script_code = '''#!/usr/bin/env python3
"""
Daily Alpaca Trading Monitor

Run this script daily to track your trading performance.
"""

import sys
import os
sys.path.append('src')

from src.trading.alpaca_tracker import AlpacaTracker

def run_daily_monitoring():
    """Run daily monitoring and alerts"""
    print("🔄 Running Daily Alpaca Monitoring...")
    
    tracker = AlpacaTracker()
    
    # Update tracking data
    report = tracker.run_daily_update()
    
    # Check performance
    metrics = tracker.calculate_performance_metrics()
    
    # Alert conditions
    if metrics and metrics['total_return'] < -5:
        print("🚨 ALERT: Portfolio down more than 5%!")
        print("   Consider reviewing your strategy")
    
    if metrics and metrics['win_rate'] < 40:
        print("⚠️  WARNING: Low win rate detected")
        print("   Current win rate: {:.1f}%".format(metrics['win_rate']))
    
    return report

if __name__ == "__main__":
    run_daily_monitoring()
'''
    
    with open('daily_monitor.py', 'w') as f:
        f.write(script_code)
    
    # Make it executable
    os.chmod('daily_monitor.py', 0o755)
    
    print("\n📊 Daily monitoring script created!")
    print("   File: daily_monitor.py")
    print("   Run daily with: python daily_monitor.py")

def main():
    """Main setup function"""
    print("🚀 Alpaca Tracker Setup")
    print("=" * 50)
    
    # Step 1: Create .env file
    env_path = create_env_file()
    
    # Step 2: Create integration examples
    create_tracking_integration()
    create_daily_monitoring_script()
    
    print("\n" + "=" * 50)
    print("✅ Setup Complete!")
    print("\n🎯 Next Steps:")
    print("1. Update your ALPACA_SECRET_KEY in the .env file")
    print("2. Run: python test_alpaca_tracker.py")
    print("3. Start trading and run: python daily_monitor.py")
    print("4. Integrate with your AI system using ai_trading_integration.py")
    
    print("\n📋 Files Created:")
    print("✅ .env - Environment variables")
    print("✅ ai_trading_integration.py - AI integration example")
    print("✅ daily_monitor.py - Daily monitoring script")
    print("✅ src/trading/alpaca_tracker.py - Main tracker module")
    
    print("\n🔧 Manual Steps Required:")
    print("1. Get your Alpaca secret key from https://alpaca.markets/")
    print("2. Update ALPACA_SECRET_KEY in .env file")
    print("3. Test with: python test_alpaca_tracker.py")
    
    print("\n💡 Pro Tips:")
    print("• Use paper trading first to test your strategy")
    print("• Run daily_monitor.py as a cron job for automation")
    print("• Check the data/ folder for reports and charts")
    print("• Monitor alerts for risk management")

if __name__ == "__main__":
    main() 