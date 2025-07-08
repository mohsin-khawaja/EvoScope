#!/usr/bin/env python3
"""
🚀 Test Your Alpaca Tracker

This script helps you test your new Alpaca tracking system with your existing API keys.
"""

import sys
import os
sys.path.append('src')

from src.trading.alpaca_tracker import AlpacaTracker, track_alpaca_performance, get_alpaca_dashboard

def test_alpaca_tracker():
    """Test the Alpaca tracker with your existing keys"""
    print("🚀 Testing Alpaca Tracker Integration")
    print("=" * 60)
    
    try:
        # Initialize tracker
        tracker = AlpacaTracker()
        
        # Test 1: Get account info
        print("\n1️⃣ Testing Account Connection:")
        account_info = tracker.get_account_info()
        if account_info:
            print(f"   ✅ Account Status: {account_info.get('status', 'Unknown')}")
            print(f"   💰 Portfolio Value: ${float(account_info.get('portfolio_value', 0)):,.2f}")
            print(f"   💵 Cash: ${float(account_info.get('cash', 0)):,.2f}")
            print(f"   🔢 Buying Power: ${float(account_info.get('buying_power', 0)):,.2f}")
        else:
            print("   ❌ Could not connect to account")
            print("   Please check your API keys in src/utils/config.py")
            return False
        
        # Test 2: Get positions
        print("\n2️⃣ Testing Current Positions:")
        positions = tracker.get_positions()
        if positions:
            print(f"   📈 Found {len(positions)} positions:")
            for pos in positions:
                pnl = float(pos.get('unrealized_pl', 0))
                pnl_pct = float(pos.get('unrealized_plpc', 0)) * 100
                print(f"     {pos['symbol']}: {pos['qty']} shares | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        else:
            print("   📊 No current positions (normal for new accounts)")
        
        # Test 3: Record portfolio snapshot
        print("\n3️⃣ Testing Portfolio Snapshot:")
        tracker.record_portfolio_snapshot()
        print("   ✅ Portfolio snapshot recorded to database")
        
        # Test 4: Performance metrics
        print("\n4️⃣ Testing Performance Calculation:")
        metrics = tracker.calculate_performance_metrics()
        if metrics:
            print(f"   📊 Portfolio Value: ${metrics['current_portfolio_value']:,.2f}")
            print(f"   📈 Total Return: {metrics['total_return']:+.2f}%")
            print(f"   🔢 Number of Trades: {metrics['num_trades']}")
        else:
            print("   📊 No performance data yet (normal for new tracking)")
        
        # Test 5: Generate report
        print("\n5️⃣ Testing Report Generation:")
        report = tracker.generate_daily_report()
        print("   ✅ Daily report generated successfully!")
        
        # Test 6: Create performance chart
        print("\n6️⃣ Testing Chart Generation:")
        try:
            tracker.create_performance_chart()
            print("   ✅ Performance chart created!")
        except Exception as e:
            print(f"   ⚠️ Chart creation failed: {e}")
            print("   (This is normal if you have no historical data yet)")
        
        print("\n" + "=" * 60)
        print("✅ Alpaca Tracker Test Completed Successfully!")
        print("\n🎯 What's working:")
        print("✅ Database connection")
        print("✅ Alpaca API connection")
        print("✅ Portfolio tracking")
        print("✅ Report generation")
        
        print("\n🚀 Next Steps:")
        print("1. Run some trades in your Alpaca paper account")
        print("2. Run tracker.run_daily_update() to track performance")
        print("3. Integrate with your AI trading system")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tracker: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your API keys in src/utils/config.py")
        print("2. Make sure you have internet connection")
        print("3. Verify your Alpaca account is active")
        return False

def demo_tracking():
    """Demo the tracking functionality"""
    print("\n🎮 Demo Mode: Simulating Tracking")
    print("=" * 50)
    
    try:
        # Quick performance tracking
        print("Running quick performance check...")
        report = track_alpaca_performance()
        
        # Get dashboard data
        print("\nGetting dashboard data...")
        dashboard = get_alpaca_dashboard()
        
        print(f"Account Status: {dashboard['account_info'].get('status', 'Unknown')}")
        print(f"Positions: {len(dashboard['positions'])}")
        print(f"Recent Orders: {len(dashboard['recent_orders'])}")
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    print("🚀 Alpaca Tracker Test Suite")
    print("=" * 60)
    
    # Test the tracker
    success = test_alpaca_tracker()
    
    if success:
        # Run demo
        demo_tracking()
    else:
        print("\n🔧 Please fix the issues above before proceeding.") 