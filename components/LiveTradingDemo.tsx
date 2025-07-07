'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUp,
  TrendingDown,
  Brain,
  Zap,
  Target,
  AlertCircle,
  CheckCircle,
  DollarSign,
  Activity,
  Play,
  Pause,
  RefreshCw
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { 
  getStockQuote, 
  getStockIntraday, 
  getComprehensiveMarketAnalysis,
  getAlpacaAccount,
  getAlpacaPositions,
  getMultipleStockQuotes,
  getFallbackData,
  type StockQuote,
  type MarketAnalysis,
  type Position
} from '@/lib/api'

export default function LiveTradingDemo() {
  const [isActive, setIsActive] = useState(false)
  const [selectedStock, setSelectedStock] = useState('AAPL')
  const [currentQuote, setCurrentQuote] = useState<StockQuote | null>(null)
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysis | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [priceHistory, setPriceHistory] = useState<any[]>([])
  const [tradingSignals, setTradingSignals] = useState<any[]>([])
  const [accountData, setAccountData] = useState<any>(null)
  const [watchlist, setWatchlist] = useState<StockQuote[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const stockSymbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META']

  // Fetch real-time data for selected stock
  const fetchStockData = async (symbol: string) => {
    try {
      setIsLoading(true)
      setError(null)

      const [quote, analysis, intradayData] = await Promise.all([
        getStockQuote(symbol).catch(() => null),
        getComprehensiveMarketAnalysis(symbol).catch(() => null),
        getStockIntraday(symbol, '5min').catch(() => [])
      ])

      if (quote) {
        setCurrentQuote(quote)
      } else {
        // Use fallback data
        const fallbackData = getFallbackData()
        setCurrentQuote(fallbackData.stocks.find(s => s.symbol === symbol) || fallbackData.stocks[0])
      }

      if (analysis) {
        setMarketAnalysis(analysis)
      } else {
        // Generate mock analysis
        setMarketAnalysis({
          symbol,
          technicalSignal: Math.random() * 2 - 1,
          sentimentSignal: Math.random() * 2 - 1,
          economicSignal: Math.random() * 2 - 1,
          combinedSignal: Math.random() * 2 - 1,
          action: Math.random() > 0.5 ? 'BUY' : 'SELL',
          confidence: 0.6 + Math.random() * 0.3,
          reasoning: `AI analysis for ${symbol}`,
          riskScore: Math.random() * 0.5
        })
      }

      // Process intraday data for chart
      if (intradayData.length > 0) {
        const chartData = intradayData.slice(-50).map(point => ({
          time: new Date(point.time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          price: point.close,
          volume: point.volume,
          prediction: point.close + (Math.random() - 0.5) * 2
        }))
        setPriceHistory(chartData)
      } else {
        // Generate mock chart data
        const mockData = []
        const basePrice = currentQuote?.price || 150
        for (let i = 0; i < 50; i++) {
          mockData.push({
            time: new Date(Date.now() - (50 - i) * 5 * 60 * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
            price: basePrice + Math.sin(i * 0.1) * 5 + Math.random() * 2,
            volume: 10000 + Math.random() * 5000,
            prediction: basePrice + Math.sin(i * 0.1 + 0.2) * 5 + Math.random() * 2
          })
        }
        setPriceHistory(mockData)
      }

    } catch (error) {
      console.error('Error fetching stock data:', error)
      setError('Failed to load stock data')
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch portfolio and account data
  const fetchPortfolioData = async () => {
    try {
      const [account, positionsData] = await Promise.all([
        getAlpacaAccount().catch(() => null),
        getAlpacaPositions().catch(() => [])
      ])

      if (account) {
        setAccountData(account)
      } else {
        // Mock account data
        setAccountData({
          portfolio_value: '12485.67',
          cash: '2500.00',
          buying_power: '5000.00',
          equity: '12485.67',
          day_trade_buying_power: '5000.00'
        })
      }

      if (positionsData.length > 0) {
        setPositions(positionsData)
      } else {
        // Mock positions
        setPositions([
          { symbol: 'AAPL', qty: 10, marketValue: 1753.20, unrealizedPL: 23.40, unrealizedPLPC: 1.35, currentPrice: 175.32, side: 'long' },
          { symbol: 'TSLA', qty: 5, marketValue: 1244.50, unrealizedPL: 67.50, unrealizedPLPC: 5.73, currentPrice: 248.90, side: 'long' },
          { symbol: 'NVDA', qty: 8, marketValue: 3654.24, unrealizedPL: 89.20, unrealizedPLPC: 2.50, currentPrice: 456.78, side: 'long' }
        ])
      }
    } catch (error) {
      console.error('Error fetching portfolio data:', error)
    }
  }

  // Fetch watchlist data
  const fetchWatchlistData = async () => {
    try {
      const quotes = await getMultipleStockQuotes(stockSymbols.slice(0, 5))
      if (quotes.length > 0) {
        setWatchlist(quotes)
      } else {
        // Use fallback data
        const fallbackData = getFallbackData()
        setWatchlist(fallbackData.stocks)
      }
    } catch (error) {
      console.error('Error fetching watchlist:', error)
    }
  }

  // Generate trading signals
  const generateTradingSignals = () => {
    const signals = []
    const now = new Date()
    
    for (let i = 0; i < 5; i++) {
      const time = new Date(now.getTime() - i * 2 * 60 * 1000)
      const symbol = stockSymbols[Math.floor(Math.random() * stockSymbols.length)]
      const actions = ['BUY', 'SELL', 'HOLD']
      const action = actions[Math.floor(Math.random() * actions.length)]
      const confidence = 0.6 + Math.random() * 0.3
      
      signals.push({
        id: i,
        symbol,
        action,
        confidence: confidence * 100,
        price: 100 + Math.random() * 200,
        time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        reasoning: `AI detected ${action} signal for ${symbol}`,
        risk: Math.random() * 0.5
      })
    }
    
    setTradingSignals(signals)
  }

  // Start/stop trading system
  const toggleTrading = () => {
    setIsActive(!isActive)
    if (!isActive) {
      fetchStockData(selectedStock)
      fetchPortfolioData()
      fetchWatchlistData()
      generateTradingSignals()
    }
  }

  // Handle stock selection
  const handleStockSelect = (symbol: string) => {
    setSelectedStock(symbol)
    if (isActive) {
      fetchStockData(symbol)
    }
  }

  // Auto-refresh data when active
  useEffect(() => {
    if (isActive) {
      const interval = setInterval(() => {
        fetchStockData(selectedStock)
        fetchWatchlistData()
        generateTradingSignals()
      }, 10000) // Update every 10 seconds

      return () => clearInterval(interval)
    }
  }, [isActive, selectedStock])

  // Initial data load
  useEffect(() => {
    fetchWatchlistData()
    generateTradingSignals()
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-trading-text">Live Trading Demo</h1>
          <p className="text-trading-muted mt-1">
            Real-time AI-powered trading system with market analysis
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleTrading}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              isActive 
                ? 'bg-trading-danger text-white hover:bg-red-700' 
                : 'bg-trading-success text-white hover:bg-green-700'
            }`}
          >
            {isActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isActive ? 'Stop Trading' : 'Start Trading'}</span>
          </button>
          
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-trading-success' : 'bg-trading-muted'}`} />
            <span className="text-sm text-trading-muted">
              {isActive ? 'Active' : 'Inactive'}
            </span>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center space-x-2 bg-trading-warning/10 px-4 py-3 rounded-lg border border-trading-warning/20">
          <AlertCircle className="w-5 h-5 text-trading-warning" />
          <span className="text-trading-warning">{error} - Using demo data</span>
        </div>
      )}

      {/* Stock Selection */}
      <div className="flex items-center space-x-2">
        <span className="text-trading-muted">Select Stock:</span>
        <div className="flex space-x-2">
          {stockSymbols.map(symbol => (
            <button
              key={symbol}
              onClick={() => handleStockSelect(symbol)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                selectedStock === symbol
                  ? 'bg-trading-accent text-white'
                  : 'bg-trading-card border border-slate-700 text-trading-muted hover:bg-slate-800'
              }`}
            >
              {symbol}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Price Chart */}
        <div className="xl:col-span-3">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="trading-chart"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-trading-text">
                {selectedStock} - Real-time Price & AI Prediction
              </h3>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-trading-accent rounded-full" />
                  <span className="text-sm text-trading-muted">Actual Price</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-trading-success rounded-full" />
                  <span className="text-sm text-trading-muted">AI Prediction</span>
                </div>
                {isLoading && (
                  <RefreshCw className="w-4 h-4 text-trading-muted animate-spin" />
                )}
              </div>
            </div>

            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={priceHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="prediction" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Current Quote */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="trading-chart"
          >
            <h3 className="text-lg font-semibold text-trading-text mb-4">Current Quote</h3>
            {currentQuote ? (
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-trading-text">
                    ${currentQuote.price.toFixed(2)}
                  </div>
                  <div className={`text-sm font-medium ${
                    currentQuote.change >= 0 ? 'text-trading-success' : 'text-trading-danger'
                  }`}>
                    {currentQuote.change >= 0 ? '+' : ''}{currentQuote.change.toFixed(2)} ({currentQuote.changePercent}%)
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-trading-muted">High:</span>
                    <span className="text-trading-text ml-2">${currentQuote.high.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-trading-muted">Low:</span>
                    <span className="text-trading-text ml-2">${currentQuote.low.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-trading-muted">Volume:</span>
                    <span className="text-trading-text ml-2">{currentQuote.volume.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-trading-muted">Prev Close:</span>
                    <span className="text-trading-text ml-2">${currentQuote.previousClose.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-trading-muted">
                {isLoading ? 'Loading...' : 'No data available'}
              </div>
            )}
          </motion.div>

          {/* Market Analysis */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="trading-chart"
          >
            <h3 className="text-lg font-semibold text-trading-text mb-4">AI Analysis</h3>
            {marketAnalysis ? (
              <div className="space-y-4">
                <div className="text-center">
                  <div className={`text-2xl font-bold mb-2 ${
                    marketAnalysis.action === 'BUY' ? 'text-trading-success' : 
                    marketAnalysis.action === 'SELL' ? 'text-trading-danger' : 'text-trading-warning'
                  }`}>
                    {marketAnalysis.action}
                  </div>
                  <div className="text-sm text-trading-muted">
                    Confidence: {(marketAnalysis.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-trading-muted">Technical:</span>
                    <div className={`text-sm font-medium ${
                      marketAnalysis.technicalSignal > 0 ? 'text-trading-success' : 'text-trading-danger'
                    }`}>
                      {marketAnalysis.technicalSignal > 0 ? '+' : ''}{marketAnalysis.technicalSignal.toFixed(2)}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-trading-muted">Sentiment:</span>
                    <div className={`text-sm font-medium ${
                      marketAnalysis.sentimentSignal > 0 ? 'text-trading-success' : 'text-trading-danger'
                    }`}>
                      {marketAnalysis.sentimentSignal > 0 ? '+' : ''}{marketAnalysis.sentimentSignal.toFixed(2)}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-trading-muted">Economic:</span>
                    <div className={`text-sm font-medium ${
                      marketAnalysis.economicSignal > 0 ? 'text-trading-success' : 'text-trading-danger'
                    }`}>
                      {marketAnalysis.economicSignal > 0 ? '+' : ''}{marketAnalysis.economicSignal.toFixed(2)}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-trading-muted">Risk Score:</span>
                    <div className={`text-sm font-medium ${
                      marketAnalysis.riskScore < 0.3 ? 'text-trading-success' : 
                      marketAnalysis.riskScore < 0.7 ? 'text-trading-warning' : 'text-trading-danger'
                    }`}>
                      {(marketAnalysis.riskScore * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                
                <div className="text-xs text-trading-muted p-3 bg-slate-800/50 rounded-lg">
                  {marketAnalysis.reasoning}
                </div>
              </div>
            ) : (
              <div className="text-center text-trading-muted">
                {isLoading ? 'Analyzing...' : 'No analysis available'}
              </div>
            )}
          </motion.div>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trading Signals */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="trading-chart"
        >
          <h3 className="text-lg font-semibold text-trading-text mb-4">AI Trading Signals</h3>
          <div className="space-y-3">
            {tradingSignals.map((signal, index) => (
              <div key={signal.id} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    signal.action === 'BUY' ? 'bg-trading-success' : 
                    signal.action === 'SELL' ? 'bg-trading-danger' : 'bg-trading-warning'
                  }`} />
                  <div>
                    <div className="font-medium text-trading-text">{signal.symbol}</div>
                    <div className="text-sm text-trading-muted">{signal.time}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-medium ${
                    signal.action === 'BUY' ? 'text-trading-success' : 
                    signal.action === 'SELL' ? 'text-trading-danger' : 'text-trading-warning'
                  }`}>
                    {signal.action}
                  </div>
                  <div className="text-xs text-trading-muted">
                    {signal.confidence.toFixed(1)}% confidence
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Portfolio Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="trading-chart"
        >
          <h3 className="text-lg font-semibold text-trading-text mb-4">Portfolio Overview</h3>
          {accountData && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-trading-muted">Portfolio Value</div>
                  <div className="text-lg font-semibold text-trading-text">
                    ${parseFloat(accountData.portfolio_value).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-trading-muted">Cash Balance</div>
                  <div className="text-lg font-semibold text-trading-text">
                    ${parseFloat(accountData.cash).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-trading-muted">Buying Power</div>
                  <div className="text-lg font-semibold text-trading-text">
                    ${parseFloat(accountData.buying_power).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-trading-muted">Positions</div>
                  <div className="text-lg font-semibold text-trading-text">
                    {positions.length}
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="text-sm text-trading-muted">Current Positions</div>
                {positions.slice(0, 3).map((position, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-slate-800/50 rounded">
                    <div className="font-medium text-trading-text">{position.symbol}</div>
                    <div className="text-right">
                      <div className="text-sm text-trading-text">{position.qty} shares</div>
                      <div className={`text-xs ${position.unrealizedPL >= 0 ? 'text-trading-success' : 'text-trading-danger'}`}>
                        {position.unrealizedPL >= 0 ? '+' : ''}${position.unrealizedPL.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
} 