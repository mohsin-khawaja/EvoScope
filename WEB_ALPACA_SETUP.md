# 🚀 Web-Based Alpaca Tracker Setup Guide

Your AI trading system now has a **live web interface** for Alpaca paper trading tracking! Here's how to set it up and use it.

## 🎯 **What You Just Got**

### **1. Complete Web Interface**
- **Real-time dashboard** showing portfolio value, positions, performance
- **Interactive controls** for testing connection, recording snapshots, running updates
- **Live alerts** for risk management and system notifications
- **Performance metrics** with charts and analytics
- **Recent orders tracking** with status updates

### **2. API Integration**
- **RESTful API** connecting React frontend to Python Alpaca tracker
- **Automatic data refresh** every 30 seconds (optional)
- **Error handling** with helpful error messages
- **Real-time connection status** monitoring

### **3. New Navigation Tab**
- Added "**Alpaca Tracker**" to your website navigation
- Seamlessly integrated with existing demo system
- Professional UI matching your trading theme

## 🛠️ **Setup Instructions**

### **Step 1: Complete Basic Setup**
```bash
# 1. Run the tracker setup (if you haven't already)
python setup_alpaca_tracker.py

# 2. Update your .env file with your actual Alpaca secret key
# Edit .env file and replace:
ALPACA_SECRET_KEY=your_actual_alpaca_secret_key_here
```

### **Step 2: Install Web Dependencies**
```bash
# Install any missing Node.js dependencies
npm install
# or
yarn install
```

### **Step 3: Start Your Web Application**
```bash
# Start the Next.js development server
npm run dev
# or
yarn dev
```

### **Step 4: Access the Alpaca Tracker**
1. Open your web app: http://localhost:3000
2. Click on "**Alpaca Tracker**" in the navigation
3. Click "**Test Connection**" to verify your setup
4. Start using the live interface!

## 🎮 **How to Use the Web Interface**

### **Main Dashboard Features:**

#### **1. Control Panel**
- **Refresh**: Update all data manually
- **Test Connection**: Verify Alpaca API connectivity  
- **Record Snapshot**: Save current portfolio state
- **Daily Update**: Run comprehensive tracking update
- **Performance**: Fetch latest performance metrics
- **Auto-refresh**: Enable/disable automatic updates every 30 seconds

#### **2. Account Overview**
- **Portfolio Value**: Total account value
- **Cash Balance**: Available cash
- **Buying Power**: Available for new trades
- **Account Status**: Active/inactive status

#### **3. Performance Metrics**
- **Total Return**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest portfolio decline
- **Total Trades**: Number of executed trades

#### **4. Current Positions**
- **Real-time P&L** for each stock position
- **Share quantities** and current market values
- **Unrealized gains/losses** with percentage changes

#### **5. Recent Orders**
- **Trade history** with execution status
- **Order types** and quantities
- **Timestamps** for all trading activity

#### **6. System Alerts**
- **Risk alerts** when portfolio drops >10%
- **Connection status** notifications
- **Update confirmations** and error messages

## 📊 **Live Testing Process**

### **1. Initial Connection Test**
```
1. Click "Test Connection" button
2. Should see green "Connected" status
3. Account data should populate automatically
4. Any errors will show in red alert box
```

### **2. Portfolio Tracking Test**
```
1. Click "Record Snapshot" to save current state
2. Check that portfolio value is recorded correctly
3. Navigate to your Alpaca web dashboard
4. Make a small paper trade (e.g., buy 1 share of AAPL)
5. Return to web interface and click "Refresh"
6. New position should appear in "Current Positions"
```

### **3. Performance Monitoring Test**
```
1. Click "Daily Update" to run full analysis
2. Performance metrics should update
3. Check alerts panel for any notifications
4. Enable "Auto-refresh" to see live updates
```

## 🔧 **Troubleshooting**

### **Common Issues:**

#### **"Connection Failed" Error**
```
Problem: Red "Disconnected" status
Solution: 
1. Check ALPACA_SECRET_KEY in .env file
2. Verify internet connection
3. Confirm Alpaca account is active
4. Check console for detailed error messages
```

#### **"No Data Available" Messages**
```
Problem: Empty dashboard panels
Solution:
1. Click "Test Connection" first
2. Ensure your Alpaca account has paper trading enabled
3. Make at least one paper trade to generate data
4. Click "Daily Update" to sync all data
```

#### **Python Script Errors**
```
Problem: API endpoints returning errors
Solution:
1. Ensure Python dependencies are installed: pip install -r requirements.txt
2. Check that src/trading/alpaca_tracker.py exists
3. Verify Python path in terminal: python --version
4. Run: python test_alpaca_tracker.py to debug
```

#### **Auto-refresh Not Working**
```
Problem: Data not updating automatically
Solution:
1. Check that auto-refresh checkbox is enabled
2. Verify browser isn't blocking background requests
3. Refresh page and try again
4. Check browser console for JavaScript errors
```

## 🎯 **Testing with Real Paper Trades**

### **Step-by-Step Test Process:**

#### **1. Make Your First Paper Trade**
```
1. Go to https://app.alpaca.markets/
2. Log into your paper trading account
3. Buy 1-2 shares of a stable stock (AAPL, MSFT)
4. Wait for order to fill (usually instant)
```

#### **2. See It in Your Web Interface**
```
1. Return to your web app
2. Click "Refresh" or wait for auto-refresh
3. New position should appear in "Current Positions"
4. Portfolio value should update
5. Order should show in "Recent Orders"
```

#### **3. Monitor Performance**
```
1. Click "Record Snapshot" to save this state
2. Make another trade (sell or buy different stock)
3. Click "Daily Update" to run full analysis
4. Performance metrics should show your trading results
5. Alerts will notify you of any significant changes
```

## 🚨 **Risk Management Features**

The web interface automatically monitors:

- **Portfolio Loss >10%**: Shows red alert
- **High Volatility**: Yellow warning alert  
- **Low Win Rate**: Performance warning
- **Connection Issues**: Red disconnected status

## 💡 **Pro Tips for Testing**

### **1. Start Small**
- Use 1-5 shares per trade initially
- Test with stable stocks (AAPL, MSFT, GOOGL)
- Monitor performance daily

### **2. Use the Interface Effectively**
- Enable auto-refresh for live monitoring
- Record snapshots before major trades
- Run daily updates to see complete analytics
- Check alerts regularly for risk warnings

### **3. Track Your AI Performance**
- Compare web interface data with your AI predictions
- Use performance metrics to validate your models
- Monitor Sharpe ratio and drawdown for risk assessment

## 🎉 **You're Ready!**

Your **AI Trading System** now has:
- ✅ **Live web interface** for Alpaca tracking
- ✅ **Real-time portfolio monitoring**
- ✅ **Performance analytics dashboard**
- ✅ **Risk management alerts**
- ✅ **Professional trading interface**

**🚀 Start testing with paper trades and watch your AI system in action!**

---

**Need Help?** 
- Check the browser console for JavaScript errors
- Run `python test_alpaca_tracker.py` for Python debugging
- Ensure all environment variables are set correctly
- Verify your Alpaca paper trading account is active 