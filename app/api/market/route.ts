import { NextRequest, NextResponse } from 'next/server'

// API Keys (in production, use environment variables)
const ALPHA_VANTAGE_KEY = "38RX2Y3EUK2CV7Y8"
const NEWSAPI_KEY = "1d5bb349-6f72-4f83-860b-9c2fb3c220bd"
const FRED_API_KEY = "56JBx7QuGHquzDi6yzMd"
const ALPACA_API_KEY = "PKH6HJ2RBVZ20P8EJPNT"
const ALPACA_SECRET = "your_secret_key" // You need to provide this

// Base URLs
const ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
const NEWSAPI_BASE = "https://newsapi.org/v2"
const FRED_BASE = "https://api.stlouisfed.org/fred"
const ALPACA_BASE = "https://paper-api.alpaca.markets/v2"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action')
  const symbol = searchParams.get('symbol')

  try {
    switch (action) {
      case 'stock_quote':
        if (!symbol) {
          return NextResponse.json({ error: 'Symbol required' }, { status: 400 })
        }
        return await getStockQuote(symbol)

      case 'multiple_quotes':
        const symbols = searchParams.get('symbols')?.split(',') || []
        return await getMultipleQuotes(symbols)

      case 'economic_indicators':
        return await getEconomicIndicators()

      case 'market_news':
        const query = searchParams.get('query') || 'stock market'
        const pageSize = parseInt(searchParams.get('pageSize') || '10')
        return await getMarketNews(query, pageSize)

      case 'portfolio':
        return await getPortfolioData()

      case 'market_analysis':
        if (!symbol) {
          return NextResponse.json({ error: 'Symbol required' }, { status: 400 })
        }
        return await getMarketAnalysis(symbol)

      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
    }
  } catch (error) {
    console.error('API Error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

async function getStockQuote(symbol: string) {
  try {
    const response = await fetch(
      `${ALPHA_VANTAGE_BASE}?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${ALPHA_VANTAGE_KEY}`
    )
    
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    
    const data = await response.json()
    
    if (data['Global Quote']) {
      const quote = data['Global Quote']
      const result = {
        symbol,
        price: parseFloat(quote['05. price']),
        change: parseFloat(quote['09. change']),
        changePercent: quote['10. change percent'].replace('%', ''),
        volume: parseInt(quote['06. volume']),
        high: parseFloat(quote['03. high']),
        low: parseFloat(quote['04. low']),
        previousClose: parseFloat(quote['08. previous close']),
        timestamp: new Date()
      }
      return NextResponse.json(result)
    }
    
    return NextResponse.json({ error: 'No data found' }, { status: 404 })
  } catch (error) {
    console.error('Error fetching stock quote:', error)
    return NextResponse.json({ error: 'Failed to fetch stock quote' }, { status: 500 })
  }
}

async function getMultipleQuotes(symbols: string[]) {
  try {
    const quotes = await Promise.all(
      symbols.slice(0, 10).map(async (symbol) => {
        try {
          const response = await fetch(
            `${ALPHA_VANTAGE_BASE}?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${ALPHA_VANTAGE_KEY}`
          )
          
          if (!response.ok) return null
          
          const data = await response.json()
          
          if (data['Global Quote']) {
            const quote = data['Global Quote']
            return {
              symbol,
              price: parseFloat(quote['05. price']),
              change: parseFloat(quote['09. change']),
              changePercent: quote['10. change percent'].replace('%', ''),
              volume: parseInt(quote['06. volume']),
              high: parseFloat(quote['03. high']),
              low: parseFloat(quote['04. low']),
              previousClose: parseFloat(quote['08. previous close']),
              timestamp: new Date()
            }
          }
          
          return null
        } catch (error) {
          console.error(`Error fetching quote for ${symbol}:`, error)
          return null
        }
      })
    )
    
    const validQuotes = quotes.filter(quote => quote !== null)
    return NextResponse.json(validQuotes)
  } catch (error) {
    console.error('Error fetching multiple quotes:', error)
    return NextResponse.json({ error: 'Failed to fetch quotes' }, { status: 500 })
  }
}

async function getEconomicIndicators() {
  const indicators = [
    { id: 'FEDFUNDS', name: 'Federal Funds Rate' },
    { id: 'UNRATE', name: 'Unemployment Rate' },
    { id: 'CPIAUCSL', name: 'Inflation Rate' },
    { id: 'DGS10', name: '10-Year Treasury' },
    { id: 'DGS2', name: '2-Year Treasury' },
    { id: 'UMCSENT', name: 'Consumer Sentiment' }
  ]
  
  try {
    const results = await Promise.all(
      indicators.map(async (indicator) => {
        try {
          const response = await fetch(
            `${FRED_BASE}/series/observations?series_id=${indicator.id}&api_key=${FRED_API_KEY}&file_type=json&limit=2&sort_order=desc`
          )
          
          if (!response.ok) return null
          
          const data = await response.json()
          
          if (data.observations && data.observations.length > 0) {
            const latest = data.observations[0]
            const previous = data.observations[1]
            
            if (latest.value !== '.') {
              const currentValue = parseFloat(latest.value)
              const previousValue = previous && previous.value !== '.' ? parseFloat(previous.value) : currentValue
              const change = currentValue - previousValue
              const changePercent = previousValue !== 0 ? (change / previousValue) * 100 : 0
              
              return {
                name: indicator.name,
                value: currentValue,
                date: latest.date,
                change,
                changePercent
              }
            }
          }
          
          return null
        } catch (error) {
          console.error(`Error fetching ${indicator.name}:`, error)
          return null
        }
      })
    )
    
    const validResults = results.filter(result => result !== null)
    return NextResponse.json(validResults)
  } catch (error) {
    console.error('Error fetching economic indicators:', error)
    return NextResponse.json({ error: 'Failed to fetch economic indicators' }, { status: 500 })
  }
}

async function getMarketNews(query: string, pageSize: number) {
  try {
    const response = await fetch(
      `${NEWSAPI_BASE}/everything?q=${encodeURIComponent(query)}&domains=reuters.com,bloomberg.com,cnbc.com,marketwatch.com&language=en&sortBy=publishedAt&pageSize=${pageSize}&apiKey=${NEWSAPI_KEY}`
    )
    
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    
    const data = await response.json()
    
    if (data.articles) {
      const articles = data.articles.map((article: any) => ({
        title: article.title,
        description: article.description || '',
        source: article.source.name,
        publishedAt: article.publishedAt,
        url: article.url,
        sentiment: analyzeSentiment(article.title + ' ' + (article.description || '')),
        sentimentScore: getSentimentScore(article.title + ' ' + (article.description || ''))
      }))
      
      return NextResponse.json(articles)
    }
    
    return NextResponse.json([])
  } catch (error) {
    console.error('Error fetching news:', error)
    return NextResponse.json({ error: 'Failed to fetch news' }, { status: 500 })
  }
}

async function getPortfolioData() {
  try {
    // Mock portfolio data since Alpaca requires secret key
    const mockData = {
      totalValue: 12485.67,
      dayChange: 234.56,
      dayChangePercent: 1.88,
      positions: [
        { symbol: 'AAPL', qty: 10, marketValue: 1753.20, unrealizedPL: 23.40, unrealizedPLPC: 1.35, currentPrice: 175.32, side: 'long' },
        { symbol: 'TSLA', qty: 5, marketValue: 1244.50, unrealizedPL: 67.50, unrealizedPLPC: 5.73, currentPrice: 248.90, side: 'long' },
        { symbol: 'NVDA', qty: 8, marketValue: 3654.24, unrealizedPL: 89.20, unrealizedPLPC: 2.50, currentPrice: 456.78, side: 'long' }
      ],
      cashBalance: 5833.73,
      buying_power: 11667.46
    }
    
    return NextResponse.json(mockData)
  } catch (error) {
    console.error('Error fetching portfolio data:', error)
    return NextResponse.json({ error: 'Failed to fetch portfolio data' }, { status: 500 })
  }
}

async function getMarketAnalysis(symbol: string) {
  try {
    // Get stock quote and news for analysis
    const [quoteResponse, newsResponse] = await Promise.all([
      fetch(`${ALPHA_VANTAGE_BASE}?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${ALPHA_VANTAGE_KEY}`),
      fetch(`${NEWSAPI_BASE}/everything?q=${symbol}&domains=reuters.com,bloomberg.com,cnbc.com,marketwatch.com&language=en&sortBy=relevancy&pageSize=5&apiKey=${NEWSAPI_KEY}`)
    ])
    
    let technicalSignal = 0
    let sentimentSignal = 0
    
    // Technical analysis
    if (quoteResponse.ok) {
      const quoteData = await quoteResponse.json()
      if (quoteData['Global Quote']) {
        const quote = quoteData['Global Quote']
        const changePercent = parseFloat(quote['10. change percent'].replace('%', ''))
        
        if (changePercent > 2) technicalSignal = 0.8
        else if (changePercent > 0.5) technicalSignal = 0.4
        else if (changePercent > 0) technicalSignal = 0.2
        else if (changePercent < -2) technicalSignal = -0.8
        else if (changePercent < -0.5) technicalSignal = -0.4
        else technicalSignal = -0.2
      }
    }
    
    // Sentiment analysis
    if (newsResponse.ok) {
      const newsData = await newsResponse.json()
      if (newsData.articles && newsData.articles.length > 0) {
        const avgSentiment = newsData.articles.reduce((sum: number, article: any) => {
          return sum + getSentimentScore(article.title + ' ' + (article.description || ''))
        }, 0) / newsData.articles.length
        sentimentSignal = Math.max(-1, Math.min(1, avgSentiment))
      }
    }
    
    // Economic signal (simplified)
    const economicSignal = Math.random() * 0.4 - 0.2 // Random between -0.2 and 0.2
    
    // Combined signal
    const combinedSignal = (technicalSignal * 0.4 + sentimentSignal * 0.3 + economicSignal * 0.3)
    
    // Determine action and confidence
    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD'
    let confidence = Math.abs(combinedSignal)
    
    if (combinedSignal > 0.3) action = 'BUY'
    else if (combinedSignal < -0.3) action = 'SELL'
    
    // Risk Score
    const riskScore = Math.abs(technicalSignal) * 0.3 + Math.random() * 0.3
    
    const analysis = {
      symbol,
      technicalSignal,
      sentimentSignal,
      economicSignal,
      combinedSignal,
      action,
      confidence,
      reasoning: `Technical: ${technicalSignal.toFixed(2)}, Sentiment: ${sentimentSignal.toFixed(2)}, Economic: ${economicSignal.toFixed(2)}`,
      riskScore
    }
    
    return NextResponse.json(analysis)
  } catch (error) {
    console.error('Error generating market analysis:', error)
    return NextResponse.json({ error: 'Failed to generate analysis' }, { status: 500 })
  }
}

// Helper functions
function analyzeSentiment(text: string): 'positive' | 'negative' | 'neutral' {
  const positiveWords = ['gain', 'rise', 'up', 'bullish', 'positive', 'growth', 'strong', 'surge', 'rally', 'boost', 'soar']
  const negativeWords = ['fall', 'drop', 'down', 'bearish', 'negative', 'decline', 'weak', 'crash', 'plunge', 'tumble', 'slump']
  
  const lowerText = text.toLowerCase()
  let positiveCount = 0
  let negativeCount = 0
  
  positiveWords.forEach(word => {
    if (lowerText.includes(word)) positiveCount++
  })
  
  negativeWords.forEach(word => {
    if (lowerText.includes(word)) negativeCount++
  })
  
  if (positiveCount > negativeCount) return 'positive'
  if (negativeCount > positiveCount) return 'negative'
  return 'neutral'
}

function getSentimentScore(text: string): number {
  const positiveWords = ['gain', 'rise', 'up', 'bullish', 'positive', 'growth', 'strong', 'surge', 'rally', 'boost', 'soar']
  const negativeWords = ['fall', 'drop', 'down', 'bearish', 'negative', 'decline', 'weak', 'crash', 'plunge', 'tumble', 'slump']
  
  const lowerText = text.toLowerCase()
  let score = 0
  
  positiveWords.forEach(word => {
    const matches = (lowerText.match(new RegExp(word, 'g')) || []).length
    score += matches * 0.1
  })
  
  negativeWords.forEach(word => {
    const matches = (lowerText.match(new RegExp(word, 'g')) || []).length
    score -= matches * 0.1
  })
  
  return Math.max(-1, Math.min(1, score))
} 