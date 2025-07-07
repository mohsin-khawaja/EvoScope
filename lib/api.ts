// Comprehensive API integration for RL-LSTM AI Trading System
// Uses Next.js API routes to handle backend data fetching

// TypeScript Interfaces
export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  changePercent: string;
  volume: number;
  high: number;
  low: number;
  previousClose: number;
  timestamp: Date;
}

export interface CryptoQuote {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  timestamp: Date;
}

export interface NewsArticle {
  title: string;
  description: string;
  source: string;
  publishedAt: string;
  url: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  sentimentScore: number;
}

export interface EconomicIndicator {
  name: string;
  value: number;
  date: string;
  change?: number;
  changePercent?: number;
}

export interface MarketAnalysis {
  symbol: string;
  technicalSignal: number;
  sentimentSignal: number;
  economicSignal: number;
  combinedSignal: number;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  riskScore: number;
}

export interface PortfolioData {
  totalValue: number;
  dayChange: number;
  dayChangePercent: number;
  positions: Position[];
  cashBalance: number;
  buying_power: number;
}

export interface Position {
  symbol: string;
  qty: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPC: number;
  currentPrice: number;
  side: 'long' | 'short';
}

export interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  reasoning: string;
  timestamp: Date;
}

// ===================
// STOCK DATA FUNCTIONS
// ===================

export async function getStockQuote(symbol: string): Promise<StockQuote | null> {
  try {
    const response = await fetch(`/api/market?action=stock_quote&symbol=${symbol}`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    
    if (data.error) {
      console.error('API Error:', data.error)
      return null
    }
    
    return data
  } catch (error) {
    console.error('Error fetching stock quote:', error)
    return null
  }
}

export async function getStockIntraday(symbol: string, interval: string = "5min"): Promise<any[]> {
  // For now, return mock data since Alpha Vantage intraday requires more complex handling
  const mockData = []
  const basePrice = 150 + Math.random() * 100
  
  for (let i = 0; i < 50; i++) {
    const time = new Date(Date.now() - (50 - i) * 5 * 60 * 1000)
    mockData.push({
      time: time.toISOString(),
      open: basePrice + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      high: basePrice + Math.sin(i * 0.1) * 5 + Math.random() * 3,
      low: basePrice + Math.sin(i * 0.1) * 5 - Math.random() * 2,
      close: basePrice + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      volume: 10000 + Math.random() * 5000
    })
  }
  
  return mockData
}

export async function getMultipleStockQuotes(symbols: string[]): Promise<StockQuote[]> {
  try {
    const response = await fetch(`/api/market?action=multiple_quotes&symbols=${symbols.join(',')}`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    
    if (data.error) {
      console.error('API Error:', data.error)
      return []
    }
    
    return data
  } catch (error) {
    console.error('Error fetching multiple quotes:', error)
    return []
  }
}

// ===================
// CRYPTO DATA FUNCTIONS
// ===================

export async function getCryptoQuote(symbol: string): Promise<CryptoQuote | null> {
  // Mock crypto data for now
  const cryptoPrices: Record<string, number> = {
    'BTC/USDT': 43250,
    'ETH/USDT': 2650,
    'BNB/USDT': 315,
    'ADA/USDT': 0.65,
    'SOL/USDT': 105
  }
  
  const basePrice = cryptoPrices[symbol] || 100
  const change24h = (Math.random() - 0.5) * basePrice * 0.1
  
  return {
    symbol,
    price: basePrice + change24h,
    change24h,
    changePercent24h: (change24h / basePrice) * 100,
    volume24h: 1000000000 + Math.random() * 500000000,
    high24h: basePrice + Math.abs(change24h) + Math.random() * 20,
    low24h: basePrice - Math.abs(change24h) - Math.random() * 20,
    timestamp: new Date()
  }
}

export async function getCryptoKlines(symbol: string, interval: string = "1h", limit: number = 100): Promise<any[]> {
  const mockData = []
  const basePrice = symbol === 'BTC/USDT' ? 43250 : 2650
  
  for (let i = 0; i < limit; i++) {
    const time = new Date(Date.now() - (limit - i) * 60 * 60 * 1000)
    mockData.push({
      time,
      open: basePrice + Math.sin(i * 0.1) * 100 + Math.random() * 50,
      high: basePrice + Math.sin(i * 0.1) * 100 + Math.random() * 100,
      low: basePrice + Math.sin(i * 0.1) * 100 - Math.random() * 50,
      close: basePrice + Math.sin(i * 0.1) * 100 + Math.random() * 50,
      volume: 1000000 + Math.random() * 500000
    })
  }
  
  return mockData
}

export async function getMultipleCryptoQuotes(symbols: string[]): Promise<CryptoQuote[]> {
  const quotes = await Promise.all(
    symbols.map(symbol => getCryptoQuote(symbol))
  )
  
  return quotes.filter(quote => quote !== null) as CryptoQuote[]
}

// Popular crypto symbols mapping
export const POPULAR_CRYPTOS = {
  BTC: 'bitcoin',
  ETH: 'ethereum',
  BNB: 'binancecoin',
  ADA: 'cardano',
  SOL: 'solana',
  DOT: 'polkadot',
  AVAX: 'avalanche-2',
  MATIC: 'matic-network',
  ATOM: 'cosmos',
  LINK: 'chainlink'
}

export function getFallbackCryptoQuote(symbol: string): CryptoQuote {
  const cryptoPrices: Record<string, number> = {
    'BTC': 43250,
    'ETH': 2650,
    'BNB': 315,
    'ADA': 0.65,
    'SOL': 105,
    'DOT': 7.5,
    'AVAX': 38,
    'MATIC': 0.85,
    'ATOM': 12,
    'LINK': 15
  }
  
  const basePrice = cryptoPrices[symbol] || 100
  const change24h = (Math.random() - 0.5) * basePrice * 0.08
  
  return {
    symbol,
    price: basePrice + change24h,
    change24h,
    changePercent24h: (change24h / basePrice) * 100,
    volume24h: 1000000000 + Math.random() * 500000000,
    high24h: basePrice + Math.abs(change24h) + Math.random() * 20,
    low24h: basePrice - Math.abs(change24h) - Math.random() * 20,
    timestamp: new Date()
  }
}

export async function getCryptoMarketAnalysis(cryptoId: string): Promise<MarketAnalysis | null> {
  try {
    // For now, return simulated analysis since we don't have a real crypto analysis API
    const symbol = Object.keys(POPULAR_CRYPTOS).find(k => POPULAR_CRYPTOS[k as keyof typeof POPULAR_CRYPTOS] === cryptoId) || 'CRYPTO'
    
    const technicalSignal = Math.random() - 0.5
    const sentimentSignal = Math.random() - 0.5
    const economicSignal = Math.random() - 0.5
    const combinedSignal = (technicalSignal + sentimentSignal + economicSignal) / 3
    
    let action: 'BUY' | 'SELL' | 'HOLD'
    let confidence: number
    
    if (combinedSignal > 0.2) {
      action = 'BUY'
      confidence = 70 + Math.random() * 25
    } else if (combinedSignal < -0.2) {
      action = 'SELL'
      confidence = 70 + Math.random() * 25
    } else {
      action = 'HOLD'
      confidence = 50 + Math.random() * 30
    }
    
    return {
      symbol,
      technicalSignal,
      sentimentSignal,
      economicSignal,
      combinedSignal,
      action,
      confidence,
      reasoning: `Crypto analysis for ${symbol}: ${action} recommended with ${confidence.toFixed(1)}% confidence based on technical indicators (${technicalSignal.toFixed(2)}), market sentiment (${sentimentSignal.toFixed(2)}), and economic factors (${economicSignal.toFixed(2)}).`,
      riskScore: Math.random() * 10
    }
  } catch (error) {
    console.error('Error in crypto market analysis:', error)
    return null
  }
}

// ===================
// NEWS & SENTIMENT FUNCTIONS
// ===================

export async function getMarketNews(query: string = "stock market", pageSize: number = 10): Promise<NewsArticle[]> {
  try {
    const response = await fetch(`/api/market?action=market_news&query=${encodeURIComponent(query)}&pageSize=${pageSize}`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    
    if (data.error) {
      console.error('API Error:', data.error)
      return []
    }
    
    return data
  } catch (error) {
    console.error('Error fetching news:', error)
    return []
  }
}

export async function getStockNews(symbol: string, daysBack: number = 3): Promise<NewsArticle[]> {
  return getMarketNews(symbol, 10)
}

// ===================
// ECONOMIC DATA FUNCTIONS
// ===================

export async function getEconomicIndicators(): Promise<EconomicIndicator[]> {
  try {
    const response = await fetch(`/api/market?action=economic_indicators`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    
    if (data.error) {
      console.error('API Error:', data.error)
      return []
    }
    
    return data
  } catch (error) {
    console.error('Error fetching economic indicators:', error)
    return []
  }
}

export async function getFEDFundsRate(): Promise<number | null> {
  try {
    const indicators = await getEconomicIndicators()
    const fedFunds = indicators.find(ind => ind.name === 'Federal Funds Rate')
    return fedFunds ? fedFunds.value : null
  } catch (error) {
    console.error('Error fetching Fed Funds Rate:', error)
    return null
  }
}

// ===================
// ALPACA TRADING FUNCTIONS
// ===================

export async function getAlpacaAccount(): Promise<any> {
  // Mock account data
  return {
    portfolio_value: '12485.67',
    cash: '2500.00',
    buying_power: '5000.00',
    equity: '12485.67',
    day_trade_buying_power: '5000.00'
  }
}

export async function getAlpacaPositions(): Promise<Position[]> {
  // Mock positions
  return [
    { symbol: 'AAPL', qty: 10, marketValue: 1753.20, unrealizedPL: 23.40, unrealizedPLPC: 1.35, currentPrice: 175.32, side: 'long' },
    { symbol: 'TSLA', qty: 5, marketValue: 1244.50, unrealizedPL: 67.50, unrealizedPLPC: 5.73, currentPrice: 248.90, side: 'long' },
    { symbol: 'NVDA', qty: 8, marketValue: 3654.24, unrealizedPL: 89.20, unrealizedPLPC: 2.50, currentPrice: 456.78, side: 'long' }
  ]
}

export async function getPortfolioData(): Promise<PortfolioData | null> {
  try {
    const response = await fetch(`/api/market?action=portfolio`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    
    if (data.error) {
      console.error('API Error:', data.error)
      return null
    }
    
    return data
  } catch (error) {
    console.error('Error fetching portfolio data:', error)
    return null
  }
}

// ===================
// ANALYSIS FUNCTIONS
// ===================

export async function getComprehensiveMarketAnalysis(symbol: string): Promise<MarketAnalysis | null> {
  try {
    const response = await fetch(`/api/market?action=market_analysis&symbol=${symbol}`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    
    if (data.error) {
      console.error('API Error:', data.error)
      return null
    }
    
    return data
  } catch (error) {
    console.error('Error fetching market analysis:', error)
    return null
  }
}

// ===================
// REAL-TIME DATA FUNCTIONS
// ===================

export async function getRealTimeMarketData(symbols: string[]): Promise<any> {
  const stockSymbols = symbols.filter(s => !s.includes('/'))
  const cryptoSymbols = symbols.filter(s => s.includes('/'))
  
  const [stocks, crypto, news, economicData] = await Promise.all([
    getMultipleStockQuotes(stockSymbols.slice(0, 5)),
    getMultipleCryptoQuotes(cryptoSymbols.slice(0, 3)),
    getMarketNews('stock market', 5),
    getEconomicIndicators()
  ])
  
  return {
    stocks,
    crypto,
    news,
    economicData,
    timestamp: new Date()
  }
}

export async function getWatchlistData(symbols: string[]): Promise<any[]> {
  const stockSymbols = symbols.filter(s => !s.includes('/'))
  const cryptoSymbols = symbols.filter(s => s.includes('/'))
  
  const [stockData, cryptoData] = await Promise.all([
    getMultipleStockQuotes(stockSymbols),
    getMultipleCryptoQuotes(cryptoSymbols)
  ])
  
  return [...stockData, ...cryptoData]
}

// ===================
// FALLBACK DATA
// ===================

export function getFallbackData() {
  return {
    stocks: [
      { symbol: 'AAPL', price: 175.32, change: 2.45, changePercent: '1.42%', volume: 45678900, high: 176.80, low: 173.20, previousClose: 172.87, timestamp: new Date() },
      { symbol: 'TSLA', price: 248.90, change: -3.20, changePercent: '-1.27%', volume: 32145600, high: 252.10, low: 247.50, previousClose: 252.10, timestamp: new Date() },
      { symbol: 'NVDA', price: 456.78, change: 12.34, changePercent: '2.78%', volume: 28934500, high: 461.20, low: 452.30, previousClose: 444.44, timestamp: new Date() },
      { symbol: 'MSFT', price: 378.85, change: 5.67, changePercent: '1.52%', volume: 23567800, high: 380.20, low: 375.40, previousClose: 373.18, timestamp: new Date() },
      { symbol: 'GOOGL', price: 142.65, change: -1.23, changePercent: '-0.85%', volume: 19876500, high: 144.20, low: 141.90, previousClose: 143.88, timestamp: new Date() }
    ],
    crypto: [
      { symbol: 'BTC/USDT', price: 43250.00, change24h: 1250.00, changePercent24h: 2.98, volume24h: 1234567890, high24h: 43800.00, low24h: 42100.00, timestamp: new Date() },
      { symbol: 'ETH/USDT', price: 2650.50, change24h: -45.20, changePercent24h: -1.68, volume24h: 987654321, high24h: 2720.00, low24h: 2630.00, timestamp: new Date() },
      { symbol: 'BNB/USDT', price: 315.75, change24h: 8.90, changePercent24h: 2.90, volume24h: 456789123, high24h: 318.50, low24h: 310.20, timestamp: new Date() }
    ],
    news: [
      { 
        title: 'Market rallies on strong economic data', 
        description: 'Stock market gains momentum as latest employment figures exceed expectations...', 
        source: 'Reuters', 
        publishedAt: new Date().toISOString(), 
        url: '#',
        sentiment: 'positive' as const, 
        sentimentScore: 0.6 
      },
      { 
        title: 'Fed considers rate adjustment amid inflation concerns', 
        description: 'Federal Reserve officials signal potential policy changes in upcoming meeting...', 
        source: 'Bloomberg', 
        publishedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), 
        url: '#',
        sentiment: 'neutral' as const, 
        sentimentScore: 0.1 
      },
      { 
        title: 'Tech sector shows resilience despite market volatility', 
        description: 'Major technology companies report stronger than expected quarterly results...', 
        source: 'CNBC', 
        publishedAt: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(), 
        url: '#',
        sentiment: 'positive' as const, 
        sentimentScore: 0.4 
      }
    ],
    economicData: [
      { name: 'Federal Funds Rate', value: 5.25, date: new Date().toISOString().split('T')[0], change: 0.25, changePercent: 5.0 },
      { name: 'Unemployment Rate', value: 3.7, date: new Date().toISOString().split('T')[0], change: -0.1, changePercent: -2.6 },
      { name: 'Inflation Rate', value: 3.2, date: new Date().toISOString().split('T')[0], change: -0.3, changePercent: -8.6 },
      { name: '10-Year Treasury', value: 4.35, date: new Date().toISOString().split('T')[0], change: 0.05, changePercent: 1.2 },
      { name: '2-Year Treasury', value: 4.85, date: new Date().toISOString().split('T')[0], change: 0.10, changePercent: 2.1 },
      { name: 'Consumer Sentiment', value: 67.4, date: new Date().toISOString().split('T')[0], change: 2.8, changePercent: 4.3 }
    ]
  }
} 