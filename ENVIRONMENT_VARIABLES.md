# EvoScope Environment Variables Guide

This document outlines all the environment variables required for the EvoScope RL-LSTM AI Trading Agent.

## Required Environment Variables

### API Keys

#### Alpha Vantage (Stock Data)
```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```
- **Source**: https://www.alphavantage.co/support/#api-key
- **Purpose**: Real-time stock quotes and historical data
- **Free Tier**: 5 API requests per minute, 500 requests per day

#### NewsAPI (Market News)
```
NEWSAPI_KEY=your_newsapi_key_here
```
- **Source**: https://newsapi.org/register
- **Purpose**: Financial news and market sentiment analysis
- **Free Tier**: 1,000 requests per day

#### FRED (Economic Data)
```
FRED_API_KEY=your_fred_api_key_here
```
- **Source**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Purpose**: Economic indicators (GDP, unemployment, interest rates)
- **Free Tier**: No rate limits

#### Alpaca (Trading)
```
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```
- **Source**: https://alpaca.markets/docs/api-documentation/
- **Purpose**: Portfolio management and trading functionality
- **Free Tier**: Paper trading available

#### Binance US (Crypto Data)
```
BINANCE_US_API_KEY=your_binance_us_api_key_here
BINANCE_US_SECRET=your_binance_us_secret_here
```
- **Source**: https://www.binance.us/en/support/faq/360046786214
- **Purpose**: Cryptocurrency data and trading
- **Free Tier**: Market data available without authentication

### Application Configuration

```
NEXT_PUBLIC_APP_NAME=EvoScope RL-LSTM AI Trading Agent
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_API_BASE_URL=https://evoscope.vercel.app
NODE_ENV=production
```

## Setup Instructions

### For Local Development

1. Create a `.env.local` file in your project root
2. Add all the environment variables listed above
3. Replace placeholder values with your actual API keys
4. Restart your development server

### For Vercel Deployment

1. Go to your Vercel dashboard
2. Navigate to your project settings
3. Click on "Environment Variables"
4. Add each variable with its corresponding value
5. Make sure to set the environment to "Production"

### For Production Deployment

All sensitive keys should be stored securely in your hosting platform's environment variable system. Never commit API keys to your repository.

## Testing Your Setup

You can test if your environment variables are properly configured by:

1. Starting your application
2. Opening the browser developer tools
3. Checking the network tab for API calls
4. Verifying that real data is being returned (not mock data)

## Security Best Practices

1. **Never commit `.env.local` to version control**
2. **Use different API keys for development and production**
3. **Regularly rotate your API keys**
4. **Monitor API usage to detect unauthorized access**
5. **Use paper trading accounts for development**

## Troubleshooting

### Common Issues

1. **API calls returning 401 errors**: Check if your API keys are correct
2. **Rate limiting errors**: Verify you're not exceeding API limits
3. **CORS errors**: Ensure your domain is properly configured
4. **Missing data**: Check if the API service is operational

### Debug Commands

Add these to your API routes for debugging:

```javascript
console.log('Environment:', process.env.NODE_ENV)
console.log('API Keys loaded:', {
  alphaVantage: !!process.env.ALPHA_VANTAGE_API_KEY,
  newsApi: !!process.env.NEWSAPI_KEY,
  fred: !!process.env.FRED_API_KEY,
  alpaca: !!process.env.ALPACA_API_KEY
})
```

## API Usage Monitoring

Consider implementing API usage monitoring to track:
- Request counts per API
- Response times
- Error rates
- Rate limit status

This will help you optimize performance and avoid hitting rate limits.

## Support

If you encounter issues with environment variables:
1. Check Vercel deployment logs
2. Verify all variables are set correctly
3. Test individual API endpoints
4. Contact the respective API support if needed

Remember to keep your API keys secure and never share them publicly! 🔒 