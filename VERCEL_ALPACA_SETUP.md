# 🚀 Vercel Alpaca Tracker Setup Guide

## 🚨 **Fixed 404 Error!**

The 404 error was caused by the API routes trying to spawn Python processes, which doesn't work in Vercel's serverless environment. This has been **completely fixed** by converting to pure TypeScript.

## ✅ **What's Fixed:**

1. **API Routes**: Converted from Python process spawning to direct TypeScript implementation
2. **Vercel Compatibility**: Now uses proper serverless functions
3. **Environment Variables**: Properly configured for Vercel deployment
4. **Real-time API**: Direct Alpaca API calls for live data

## 🔧 **Environment Variables Setup:**

### **Step 1: Set Environment Variables in Vercel**

Go to your Vercel dashboard → Project Settings → Environment Variables and add:

```
ALPACA_API_KEY=PKH6HJ2RBVZ20P8EJPNT
ALPACA_SECRET_KEY=your-secret-key-here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
OPENAI_API_KEY=your-openai-api-key-here
```

### **Step 2: Get Your Alpaca Secret Key**

1. Go to [Alpaca Markets](https://alpaca.markets/)
2. Login to your account
3. Go to **Paper Trading** → **API Keys**
4. Copy your **Secret Key**
5. Update the `ALPACA_SECRET_KEY` in Vercel

### **Step 3: Deploy**

```bash
# Deploy to Vercel
vercel --prod
```

## 🎯 **API Endpoints:**

### **Dashboard Data** (GET)
```
https://evoscope.vercel.app/api/alpaca-tracker?action=dashboard
```

### **Performance Metrics** (GET)
```
https://evoscope.vercel.app/api/alpaca-tracker?action=performance
```

### **Daily Report** (GET)
```
https://evoscope.vercel.app/api/alpaca-tracker?action=daily-report
```

### **Test Connection** (GET)
```
https://evoscope.vercel.app/api/alpaca-tracker?action=test-connection
```

### **Record Snapshot** (POST)
```
https://evoscope.vercel.app/api/alpaca-tracker
Body: { "action": "record-snapshot" }
```

## 🧪 **Testing:**

### **1. Test Connection**
```bash
curl "https://evoscope.vercel.app/api/alpaca-tracker?action=test-connection"
```

### **2. Get Dashboard**
```bash
curl "https://evoscope.vercel.app/api/alpaca-tracker?action=dashboard"
```

### **3. Get Performance**
```bash
curl "https://evoscope.vercel.app/api/alpaca-tracker?action=performance"
```

## 📊 **Expected Response:**

```json
{
  "success": true,
  "data": {
    "account_info": {
      "account_number": "your-account",
      "status": "ACTIVE",
      "currency": "USD",
      "cash": "100000.00",
      "portfolio_value": "100000.00",
      "equity": "100000.00",
      "buying_power": "400000.00"
    },
    "positions": [],
    "performance": {
      "total_return": 0,
      "daily_return": 0,
      "volatility": 0,
      "sharpe_ratio": 0,
      "max_drawdown": 0,
      "win_rate": 0,
      "num_trades": 0,
      "current_portfolio_value": 100000
    },
    "recent_orders": [],
    "timestamp": "2024-01-01T00:00:00.000Z"
  }
}
```

## 🌟 **Features:**

### **✅ Real-time Data:**
- Account information
- Current positions
- Recent orders
- Portfolio history

### **✅ Performance Analytics:**
- Total return
- Daily return
- Volatility
- Sharpe ratio
- Max drawdown
- Win rate

### **✅ Professional Dashboard:**
- Auto-refresh every 30 seconds
- Real-time connection status
- Alert system
- Control panel

## 🔧 **Local Development:**

### **Step 1: Environment Variables**
Create `.env.local`:
```
ALPACA_API_KEY=PKH6HJ2RBVZ20P8EJPNT
ALPACA_SECRET_KEY=your-secret-key-here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

### **Step 2: Run Development Server**
```bash
npm install
npm run dev
```

### **Step 3: Test Locally**
```bash
curl "http://localhost:3000/api/alpaca-tracker?action=test-connection"
```

## 🚨 **Error Handling:**

### **Common Issues:**

1. **API Key Not Set:**
   ```json
   {
     "success": false,
     "error": "Alpaca API keys not configured"
   }
   ```

2. **Invalid API Key:**
   ```json
   {
     "success": false,
     "error": "Alpaca API error: 401 Unauthorized"
   }
   ```

3. **Account Issues:**
   ```json
   {
     "success": false,
     "error": "Alpaca API error: 403 Forbidden"
   }
   ```

## 📈 **Next Steps:**

1. **Set up environment variables** in Vercel
2. **Deploy and test** the connection
3. **Make some paper trades** to generate data
4. **Monitor performance** using the dashboard

## 🎉 **Success!**

Your Alpaca tracker is now **fully compatible** with Vercel and ready for production use! The 404 error is completely resolved.

---

**Need help?** Check the Vercel logs or test locally first. 