# EvoScope Vercel Deployment Guide

## Prerequisites

- Vercel account
- GitHub repository connected to Vercel
- Domain: `evoscope.vercel.app`

## Environment Variables Setup

In your Vercel dashboard, navigate to your project settings and add the following environment variables:

### API Keys
```
ALPHA_VANTAGE_API_KEY=38RX2Y3EUK2CV7Y8
NEWSAPI_KEY=1d5bb349-6f72-4f83-860b-9c2fb3c220bd
FRED_API_KEY=56JBx7QuGHquzDi6yzMd
ALPACA_API_KEY=PKH6HJ2RBVZ20P8EJPNT
ALPACA_SECRET_KEY=your_actual_secret_key_here
```

### Application Configuration
```
NEXT_PUBLIC_APP_NAME=EvoScope RL-LSTM AI Trading Agent
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_API_BASE_URL=https://evoscope.vercel.app
```

### Trading API Configuration
```
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
BINANCE_US_API_KEY=UVmgRMxKoetKkVgEEcuoPhmjGSBgtY3OfhA5Gl9jPFcDpD7LAcs7btnPVJTyqXnf
BINANCE_US_SECRET=5JitR0QMrk8JcATQ1wgvu1jK1fEKwbzt0SDRUXAN4bG2ItvEilb3sFTEzg0aFq0N
```

## Deployment Steps

### 1. Update API Route for Environment Variables

Update your API route to use environment variables instead of hardcoded values:

```typescript
// app/api/market/route.ts
const ALPHA_VANTAGE_KEY = process.env.ALPHA_VANTAGE_API_KEY || "38RX2Y3EUK2CV7Y8"
const NEWSAPI_KEY = process.env.NEWSAPI_KEY || "1d5bb349-6f72-4f83-860b-9c2fb3c220bd"
const FRED_API_KEY = process.env.FRED_API_KEY || "56JBx7QuGHquzDi6yzMd"
const ALPACA_API_KEY = process.env.ALPACA_API_KEY || "PKH6HJ2RBVZ20P8EJPNT"
const ALPACA_SECRET = process.env.ALPACA_SECRET_KEY || "your_secret_key"
```

### 2. Deploy Commands

1. **Push to GitHub**: Make sure all your latest changes are committed and pushed to your GitHub repository.

2. **Vercel CLI Deployment** (if using CLI):
   ```bash
   npm install -g vercel
   vercel --prod
   ```

3. **Automatic Deployment**: If connected to GitHub, Vercel will automatically deploy when you push to your main branch.

### 3. Custom Domain Setup

1. Go to your Vercel dashboard
2. Navigate to your project settings
3. Go to "Domains" tab
4. Add `evoscope.vercel.app` as your custom domain

### 4. Build Settings

Vercel should automatically detect your Next.js configuration, but ensure:

- **Framework**: Next.js
- **Build Command**: `npm run build`
- **Output Directory**: `.next`
- **Install Command**: `npm install`

## Verification Steps

After deployment, verify the following:

1. **Homepage**: `https://evoscope.vercel.app` loads correctly
2. **API Endpoints**: Test the market data API endpoints
3. **Real-time Data**: Check if stock quotes and market data are loading
4. **Interactive Features**: Test the trading demo and charts
5. **Mobile Responsiveness**: Ensure the app works on mobile devices

## Performance Optimization

### 1. Image Optimization
Your `next.config.js` already has:
```javascript
images: {
  unoptimized: true
}
```

### 2. Static Generation
Consider adding static generation for better performance:
```javascript
export const dynamic = 'force-static'
```

### 3. Caching
Implement API response caching for better performance:
```javascript
export const revalidate = 60 // Revalidate every 60 seconds
```

## Monitoring and Analytics

### 1. Vercel Analytics
Enable Vercel Analytics in your dashboard for:
- Performance monitoring
- User analytics
- Error tracking

### 2. Environment-Specific Configurations

Create different configurations for staging vs production:

```javascript
// next.config.js
const isProduction = process.env.NODE_ENV === 'production'

const nextConfig = {
  images: {
    unoptimized: true
  },
  env: {
    NEXT_PUBLIC_APP_NAME: 'EvoScope RL-LSTM AI Trading Agent',
    NEXT_PUBLIC_APP_VERSION: '1.0.0',
    NEXT_PUBLIC_API_BASE_URL: isProduction 
      ? 'https://evoscope.vercel.app' 
      : 'http://localhost:3000'
  }
}
```

## Security Considerations

1. **API Keys**: Never commit API keys to your repository
2. **Environment Variables**: Use Vercel's secure environment variable storage
3. **CORS**: Configure CORS properly for your domain
4. **Rate Limiting**: Implement rate limiting for API endpoints

## Troubleshooting

### Common Issues:

1. **Build Failures**: Check your dependencies and TypeScript errors
2. **API Timeouts**: Increase timeout limits in your API routes
3. **Environment Variables**: Ensure all required variables are set
4. **Import Errors**: Check for missing dependencies or incorrect imports

### Debug Mode:
Add debug logging to your API routes:
```javascript
console.log('Environment:', process.env.NODE_ENV)
console.log('API Keys loaded:', !!process.env.ALPHA_VANTAGE_API_KEY)
```

## Post-Deployment Checklist

- [ ] All environment variables are set
- [ ] Custom domain is configured
- [ ] SSL certificate is active
- [ ] API endpoints are responding
- [ ] Charts and data are loading
- [ ] Mobile responsiveness is tested
- [ ] Error tracking is enabled
- [ ] Analytics are configured

## Support

If you encounter any issues during deployment:
1. Check Vercel deployment logs
2. Test API endpoints individually
3. Verify environment variables are correctly set
4. Check for any CORS issues
5. Ensure all dependencies are included in package.json

Your EvoScope RL-LSTM AI Trading Agent is now ready for production deployment! 🚀 