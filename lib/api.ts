// API integration for real market data
const ALPHA_VANTAGE_KEY = "38RX2Y3EUK2CV7Y8";
const NEWSAPI_KEY = "1d5bb349-6f72-4f83-860b-9c2fb3c220bd";
const COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3";

export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  changePercent: string;
  volume: number;
  high: number;
  low: number;
  timestamp: Date;
}

export interface CryptoQuote {
  id: string;
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  marketCap: number;
  high24h: number;
  low24h: number;
  timestamp: Date;
}

export interface NewsArticle {
  title: string;
  description: string;
  source: string;
  publishedAt: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  sentimentScore: number;
}

export interface MarketAnalysis {
  symbol: string;
  technicalSignal: number;
  sentimentSignal: number;
  combinedSignal: number;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
}

// Alpha Vantage API functions
export async function getStockQuote(symbol: string): Promise<StockQuote | null> {
  try {
    const response = await fetch(
      `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${ALPHA_VANTAGE_KEY}`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data['Global Quote']) {
      const quote = data['Global Quote'];
      return {
        symbol,
        price: parseFloat(quote['05. price']),
        change: parseFloat(quote['09. change']),
        changePercent: quote['10. change percent'].replace('%', ''),
        volume: parseInt(quote['06. volume']),
        high: parseFloat(quote['03. high']),
        low: parseFloat(quote['04. low']),
        timestamp: new Date()
      };
    }
    
    return null;
  } catch (error) {
    console.error('Error fetching stock quote:', error);
    return null;
  }
}

export async function getMultipleQuotes(symbols: string[]): Promise<StockQuote[]> {
  const quotes = await Promise.all(
    symbols.map(symbol => getStockQuote(symbol))
  );
  
  return quotes.filter(quote => quote !== null) as StockQuote[];
}

// NewsAPI functions
export async function getStockNews(symbol: string, daysBack: number = 7): Promise<NewsArticle[]> {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - daysBack * 24 * 60 * 60 * 1000);
    
    const response = await fetch(
      `https://newsapi.org/v2/everything?q=${symbol}&domains=reuters.com,bloomberg.com,cnbc.com,marketwatch.com&from=${startDate.toISOString().split('T')[0]}&to=${endDate.toISOString().split('T')[0]}&sortBy=relevancy&pageSize=10&apiKey=${NEWSAPI_KEY}`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.articles) {
      return data.articles.map((article: any) => ({
        title: article.title,
        description: article.description || '',
        source: article.source.name,
        publishedAt: article.publishedAt,
        sentiment: analyzeSentiment(article.title + ' ' + (article.description || '')),
        sentimentScore: getSentimentScore(article.title + ' ' + (article.description || ''))
      }));
    }
    
    return [];
  } catch (error) {
    console.error('Error fetching news:', error);
    return [];
  }
}

// Simple sentiment analysis
function analyzeSentiment(text: string): 'positive' | 'negative' | 'neutral' {
  const positiveWords = ['gain', 'rise', 'up', 'bullish', 'positive', 'growth', 'strong', 'surge', 'rally'];
  const negativeWords = ['fall', 'drop', 'down', 'bearish', 'negative', 'decline', 'weak', 'crash', 'plunge'];
  
  const lowerText = text.toLowerCase();
  let positiveCount = 0;
  let negativeCount = 0;
  
  positiveWords.forEach(word => {
    if (lowerText.includes(word)) positiveCount++;
  });
  
  negativeWords.forEach(word => {
    if (lowerText.includes(word)) negativeCount++;
  });
  
  if (positiveCount > negativeCount) return 'positive';
  if (negativeCount > positiveCount) return 'negative';
  return 'neutral';
}

function getSentimentScore(text: string): number {
  const positiveWords = ['gain', 'rise', 'up', 'bullish', 'positive', 'growth', 'strong', 'surge', 'rally'];
  const negativeWords = ['fall', 'drop', 'down', 'bearish', 'negative', 'decline', 'weak', 'crash', 'plunge'];
  
  const lowerText = text.toLowerCase();
  let score = 0;
  
  positiveWords.forEach(word => {
    const matches = (lowerText.match(new RegExp(word, 'g')) || []).length;
    score += matches * 0.1;
  });
  
  negativeWords.forEach(word => {
    const matches = (lowerText.match(new RegExp(word, 'g')) || []).length;
    score -= matches * 0.1;
  });
  
  return Math.max(-1, Math.min(1, score));
}

// Combined market analysis
export async function getMarketAnalysis(symbol: string): Promise<MarketAnalysis | null> {
  try {
    // Get stock quote and news in parallel
    const [quote, news] = await Promise.all([
      getStockQuote(symbol),
      getStockNews(symbol, 3)
    ]);
    
    if (!quote) return null;
    
    // Calculate technical signal
    let technicalSignal = 0;
    if (quote.change > 0) technicalSignal += 0.5;
    if (parseFloat(quote.changePercent) > 2) technicalSignal += 0.3;
    else if (parseFloat(quote.changePercent) < -2) technicalSignal -= 0.3;
    
    // Calculate sentiment signal
    let sentimentSignal = 0;
    if (news.length > 0) {
      const avgSentiment = news.reduce((sum, article) => sum + article.sentimentScore, 0) / news.length;
      sentimentSignal = avgSentiment;
    } else {
      // Fallback to technical sentiment
      const changePercent = parseFloat(quote.changePercent);
      if (changePercent > 2) sentimentSignal = 0.3;
      else if (changePercent < -2) sentimentSignal = -0.3;
      else if (changePercent > 0) sentimentSignal = 0.1;
      else sentimentSignal = -0.1;
    }
    
    // Combined signal
    const combinedSignal = (technicalSignal + sentimentSignal) / 2;
    
    // Determine action and confidence
    let action: 'BUY' | 'SELL' | 'HOLD';
    let confidence: number;
    let reasoning: string;
    
    if (combinedSignal > 0.2) {
      action = 'BUY';
      confidence = Math.min(95, 60 + Math.abs(combinedSignal) * 100);
      reasoning = `Strong bullish signals: Price up ${quote.changePercent}%, ${news.length > 0 ? 'positive news sentiment' : 'technical momentum'}`;
    } else if (combinedSignal < -0.2) {
      action = 'SELL';
      confidence = Math.min(95, 60 + Math.abs(combinedSignal) * 100);
      reasoning = `Bearish signals detected: Price down ${quote.changePercent}%, ${news.length > 0 ? 'negative news sentiment' : 'technical weakness'}`;
    } else {
      action = 'HOLD';
      confidence = 50;
      reasoning = `Mixed signals: Price ${quote.changePercent}%, neutral sentiment`;
    }
    
    return {
      symbol,
      technicalSignal,
      sentimentSignal,
      combinedSignal,
      action,
      confidence,
      reasoning
    };
  } catch (error) {
    console.error('Error in market analysis:', error);
    return null;
  }
}

// Fallback data for when APIs are not available
export function getFallbackQuote(symbol: string): StockQuote {
  const basePrice = 150;
  const randomChange = (Math.random() - 0.5) * 10;
  const price = basePrice + randomChange;
  
  return {
    symbol,
    price,
    change: randomChange,
    changePercent: ((randomChange / basePrice) * 100).toFixed(2),
    volume: Math.floor(Math.random() * 10000000),
    high: price + Math.random() * 5,
    low: price - Math.random() * 5,
    timestamp: new Date()
  };
}

export function getFallbackAnalysis(symbol: string): MarketAnalysis {
  const actions: ('BUY' | 'SELL' | 'HOLD')[] = ['BUY', 'SELL', 'HOLD'];
  const action = actions[Math.floor(Math.random() * actions.length)];
  const confidence = 60 + Math.random() * 30;
  
  return {
    symbol,
    technicalSignal: Math.random() - 0.5,
    sentimentSignal: Math.random() - 0.5,
    combinedSignal: Math.random() - 0.5,
    action,
    confidence,
    reasoning: `Simulated ${action} signal with ${confidence.toFixed(1)}% confidence`
  };
}

// CoinGecko API functions (No API key required!)
export async function getCryptoQuote(cryptoId: string): Promise<CryptoQuote | null> {
  try {
    const response = await fetch(
      `${COINGECKO_BASE_URL}/simple/price?ids=${cryptoId}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_market_cap=true&include_last_updated_at=true`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    const coinData = data[cryptoId];
    
    if (coinData) {
      // Get additional data for high/low
      const detailResponse = await fetch(
        `${COINGECKO_BASE_URL}/coins/${cryptoId}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false`
      );
      
      let high24h = coinData.usd * 1.05; // Fallback
      let low24h = coinData.usd * 0.95;  // Fallback
      let name = cryptoId;
      let symbol = cryptoId;
      
      if (detailResponse.ok) {
        const detailData = await detailResponse.json();
        high24h = detailData.market_data?.high_24h?.usd || high24h;
        low24h = detailData.market_data?.low_24h?.usd || low24h;
        name = detailData.name || cryptoId;
        symbol = detailData.symbol?.toUpperCase() || cryptoId;
      }
      
      return {
        id: cryptoId,
        symbol,
        name,
        price: coinData.usd,
        change24h: coinData.usd_24h_change || 0,
        changePercent24h: coinData.usd_24h_change || 0,
        volume24h: coinData.usd_24h_vol || 0,
        marketCap: coinData.usd_market_cap || 0,
        high24h,
        low24h,
        timestamp: new Date()
      };
    }
    
    return null;
  } catch (error) {
    console.error('Error fetching crypto quote:', error);
    return null;
  }
}

export async function getMultipleCryptoQuotes(cryptoIds: string[]): Promise<CryptoQuote[]> {
  try {
    const idsString = cryptoIds.join(',');
    const response = await fetch(
      `${COINGECKO_BASE_URL}/simple/price?ids=${idsString}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_market_cap=true&include_last_updated_at=true`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    const quotes: CryptoQuote[] = [];
    
    for (const cryptoId of cryptoIds) {
      const coinData = data[cryptoId];
      if (coinData) {
        quotes.push({
          id: cryptoId,
          symbol: cryptoId.toUpperCase(),
          name: cryptoId,
          price: coinData.usd,
          change24h: coinData.usd_24h_change || 0,
          changePercent24h: coinData.usd_24h_change || 0,
          volume24h: coinData.usd_24h_vol || 0,
          marketCap: coinData.usd_market_cap || 0,
          high24h: coinData.usd * 1.05, // Approximation
          low24h: coinData.usd * 0.95,  // Approximation
          timestamp: new Date()
        });
      }
    }
    
    return quotes;
  } catch (error) {
    console.error('Error fetching multiple crypto quotes:', error);
    return [];
  }
}

export async function getCryptoMarketAnalysis(cryptoId: string): Promise<MarketAnalysis | null> {
  try {
    const quote = await getCryptoQuote(cryptoId);
    if (!quote) return null;
    
    // Calculate technical signal based on price movement
    let technicalSignal = 0;
    if (quote.changePercent24h > 5) technicalSignal += 0.5;
    else if (quote.changePercent24h > 2) technicalSignal += 0.3;
    else if (quote.changePercent24h < -5) technicalSignal -= 0.5;
    else if (quote.changePercent24h < -2) technicalSignal -= 0.3;
    
    // Volume analysis
    if (quote.volume24h > quote.marketCap * 0.1) { // High volume relative to market cap
      if (quote.changePercent24h > 0) technicalSignal += 0.2;
      else technicalSignal -= 0.2;
    }
    
    // Simple sentiment based on momentum
    let sentimentSignal = 0;
    if (quote.changePercent24h > 10) sentimentSignal = 0.4;
    else if (quote.changePercent24h > 5) sentimentSignal = 0.2;
    else if (quote.changePercent24h < -10) sentimentSignal = -0.4;
    else if (quote.changePercent24h < -5) sentimentSignal = -0.2;
    else sentimentSignal = quote.changePercent24h / 100; // Normalize small changes
    
    // Combined signal
    const combinedSignal = (technicalSignal + sentimentSignal) / 2;
    
    // Determine action and confidence
    let action: 'BUY' | 'SELL' | 'HOLD';
    let confidence: number;
    let reasoning: string;
    
    if (combinedSignal > 0.2) {
      action = 'BUY';
      confidence = Math.min(95, 65 + Math.abs(combinedSignal) * 100);
      reasoning = `Strong bullish momentum: ${quote.symbol} up ${quote.changePercent24h.toFixed(2)}% with high volume`;
    } else if (combinedSignal < -0.2) {
      action = 'SELL';
      confidence = Math.min(95, 65 + Math.abs(combinedSignal) * 100);
      reasoning = `Bearish trend detected: ${quote.symbol} down ${Math.abs(quote.changePercent24h).toFixed(2)}% with selling pressure`;
    } else {
      action = 'HOLD';
      confidence = 50 + Math.random() * 20;
      reasoning = `Consolidation phase: ${quote.symbol} showing ${quote.changePercent24h.toFixed(2)}% change, waiting for clearer signals`;
    }
    
    return {
      symbol: quote.symbol,
      technicalSignal,
      sentimentSignal,
      combinedSignal,
      action,
      confidence,
      reasoning
    };
  } catch (error) {
    console.error('Error in crypto market analysis:', error);
    return null;
  }
}

// Popular crypto mappings (CoinGecko IDs)
export const POPULAR_CRYPTOS = {
  'BTC': 'bitcoin',
  'ETH': 'ethereum',
  'BNB': 'binancecoin',
  'ADA': 'cardano',
  'SOL': 'solana',
  'DOGE': 'dogecoin',
  'MATIC': 'matic-network',
  'AVAX': 'avalanche-2',
  'DOT': 'polkadot',
  'LINK': 'chainlink'
};

export function getFallbackCryptoQuote(symbol: string): CryptoQuote {
  const basePrice = symbol === 'BTC' ? 45000 : symbol === 'ETH' ? 2500 : 100;
  const randomChange = (Math.random() - 0.5) * 10;
  const price = basePrice + (basePrice * randomChange / 100);
  
  return {
    id: symbol.toLowerCase(),
    symbol,
    name: symbol,
    price,
    change24h: randomChange,
    changePercent24h: randomChange,
    volume24h: Math.floor(Math.random() * 1000000000),
    marketCap: price * Math.floor(Math.random() * 100000000),
    high24h: price + Math.random() * price * 0.05,
    low24h: price - Math.random() * price * 0.05,
    timestamp: new Date()
  };
} 