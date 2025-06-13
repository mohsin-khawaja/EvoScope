'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Play,
  Pause,
  Square,
  TrendingUp,
  TrendingDown,
  Brain,
  Zap,
  Target,
  DollarSign,
  Activity,
  RefreshCw
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import toast from 'react-hot-toast'

interface TradeSignal {
  action: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  price: number
  reasoning: string
  timestamp: Date
}

export default function LiveTradingDemo() {
  const [isRunning, setIsRunning] = useState(false)
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL')
  const [portfolio, setPortfolio] = useState({
    cash: 10000,
    positions: {} as Record<string, number>,
    totalValue: 10000
  })
  
  const [priceData, setPriceData] = useState<any[]>([])
  const [currentSignal, setCurrentSignal] = useState<TradeSignal | null>(null)
  const [tradeHistory, setTradeHistory] = useState<any[]>([])
  const [aiThinking, setAiThinking] = useState(false)

  const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'BTC-USD']
  const basePrice = 150

  // Simulate real-time price updates
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      const now = new Date()
      const newPrice = basePrice + Math.sin(Date.now() / 10000) * 20 + (Math.random() - 0.5) * 10
      
      setPriceData(prev => {
        const newData = [...prev, {
          time: now.toLocaleTimeString(),
          price: newPrice,
          volume: Math.floor(Math.random() * 1000000),
          prediction: newPrice + (Math.random() - 0.5) * 5
        }].slice(-50) // Keep last 50 points
        
        return newData
      })

      // Trigger AI analysis every 3-5 seconds
      if (Math.random() < 0.3) {
        analyzeAndTrade(newPrice)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [isRunning, selectedSymbol])

  const analyzeAndTrade = async (currentPrice: number) => {
    setAiThinking(true)
    
    // Simulate AI thinking time
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    const actions = ['BUY', 'SELL', 'HOLD'] as const
    const action = actions[Math.floor(Math.random() * actions.length)]
    const confidence = 65 + Math.random() * 30 // 65-95%
    
    const reasoningOptions = {
      BUY: [
        'LSTM predicts upward trend with strong momentum indicators',
        'RL agent identified favorable risk-reward ratio',
        'Technical analysis shows bullish divergence',
        'Market sentiment analysis suggests positive outlook'
      ],
      SELL: [
        'LSTM model detects potential price reversal',
        'RL agent recommends profit-taking at current levels',
        'Technical indicators show overbought conditions',
        'Risk management suggests reducing exposure'
      ],
      HOLD: [
        'Market conditions remain uncertain',
        'Current position size is optimal',
        'Waiting for stronger technical signals',
        'Risk-reward ratio not favorable for new positions'
      ]
    }
    
    const signal: TradeSignal = {
      action,
      confidence,
      price: currentPrice,
      reasoning: reasoningOptions[action][Math.floor(Math.random() * reasoningOptions[action].length)],
      timestamp: new Date()
    }
    
    setCurrentSignal(signal)
    setAiThinking(false)
    
    // Execute trade if confidence is high enough
    if (confidence > 80 && action !== 'HOLD') {
      executeTrade(signal)
    }
  }

  const executeTrade = (signal: TradeSignal) => {
    const quantity = Math.floor(portfolio.cash / signal.price / 10) // Conservative position sizing
    
    if (quantity > 0) {
      const trade = {
        symbol: selectedSymbol,
        action: signal.action,
        quantity,
        price: signal.price,
        timestamp: signal.timestamp,
        confidence: signal.confidence,
        reasoning: signal.reasoning
      }
      
      setTradeHistory(prev => [trade, ...prev.slice(0, 9)]) // Keep last 10 trades
      
      // Update portfolio
      if (signal.action === 'BUY') {
        const cost = quantity * signal.price
        setPortfolio(prev => ({
          ...prev,
          cash: prev.cash - cost,
          positions: {
            ...prev.positions,
            [selectedSymbol]: (prev.positions[selectedSymbol] || 0) + quantity
          }
        }))
        
        toast.success(`🟢 Bought ${quantity} shares of ${selectedSymbol} at $${signal.price.toFixed(2)}`)
      } else if (signal.action === 'SELL') {
        const revenue = quantity * signal.price
        setPortfolio(prev => ({
          ...prev,
          cash: prev.cash + revenue,
          positions: {
            ...prev.positions,
            [selectedSymbol]: Math.max(0, (prev.positions[selectedSymbol] || 0) - quantity)
          }
        }))
        
        toast.success(`🔴 Sold ${quantity} shares of ${selectedSymbol} at $${signal.price.toFixed(2)}`)
      }
    }
  }

  const startDemo = () => {
    setIsRunning(true)
    setPriceData([])
    setTradeHistory([])
    setCurrentSignal(null)
    toast.success('🚀 Live trading demo started!')
  }

  const stopDemo = () => {
    setIsRunning(false)
    toast.success('⏹️ Demo stopped')
  }

  const resetDemo = () => {
    setIsRunning(false)
    setPriceData([])
    setTradeHistory([])
    setCurrentSignal(null)
    setPortfolio({
      cash: 10000,
      positions: {},
      totalValue: 10000
    })
    toast.success('🔄 Demo reset')
  }

  const currentPrice = priceData.length > 0 ? priceData[priceData.length - 1]?.price : basePrice

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-trading-text">Live Trading Demo</h1>
          <p className="text-trading-muted mt-1">
            Real-time AI trading simulation with LSTM predictions and RL decisions
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="bg-trading-card border border-slate-700 rounded-lg px-3 py-2 text-trading-text"
          >
            {symbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
          
          <button
            onClick={isRunning ? stopDemo : startDemo}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              isRunning ? 'bg-trading-danger hover:bg-red-600' : 'bg-trading-success hover:bg-green-600'
            } text-white`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isRunning ? 'Stop' : 'Start'} Demo</span>
          </button>
          
          <button
            onClick={resetDemo}
            className="flex items-center space-x-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="metric-card"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-slate-800">
              <DollarSign className="w-5 h-5 text-trading-success" />
            </div>
          </div>
          <div className="metric-value">${portfolio.cash.toFixed(2)}</div>
          <div className="metric-label">Available Cash</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="metric-card"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-slate-800">
              <Activity className="w-5 h-5 text-trading-accent" />
            </div>
          </div>
          <div className="metric-value">{Object.keys(portfolio.positions).length}</div>
          <div className="metric-label">Active Positions</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="metric-card"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-slate-800">
              <Target className="w-5 h-5 text-trading-warning" />
            </div>
          </div>
          <div className="metric-value">${currentPrice.toFixed(2)}</div>
          <div className="metric-label">Current Price</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="metric-card"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-slate-800">
              <TrendingUp className="w-5 h-5 text-trading-success" />
            </div>
          </div>
          <div className="metric-value">{tradeHistory.length}</div>
          <div className="metric-label">Trades Executed</div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Price Chart */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="xl:col-span-2 trading-chart"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-trading-text">Live Price Feed - {selectedSymbol}</h3>
            <div className="flex items-center space-x-2">
              {isRunning && (
                <>
                  <div className="status-online" />
                  <span className="text-sm text-trading-success">Streaming Live</span>
                </>
              )}
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={priceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#64748b" />
              <YAxis stroke="#64748b" domain={['dataMin - 5', 'dataMax + 5']} />
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
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* AI Analysis Panel */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-trading-text mb-4">AI Analysis</h3>
          
          {aiThinking ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-center">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="w-8 h-8 border-3 border-trading-accent border-t-transparent rounded-full mx-auto mb-3"
                />
                <p className="text-trading-muted">AI analyzing market data...</p>
              </div>
            </div>
          ) : currentSignal ? (
            <div className="space-y-4">
              <div className={`p-4 rounded-lg border ${
                currentSignal.action === 'BUY' 
                  ? 'bg-trading-success/10 border-trading-success/20' 
                  : currentSignal.action === 'SELL'
                  ? 'bg-trading-danger/10 border-trading-danger/20'
                  : 'bg-trading-warning/10 border-trading-warning/20'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-lg">
                    {currentSignal.action === 'BUY' ? '🟢 BUY' : currentSignal.action === 'SELL' ? '🔴 SELL' : '🟡 HOLD'}
                  </span>
                  <span className="text-sm text-trading-muted">
                    {currentSignal.confidence.toFixed(1)}% confidence
                  </span>
                </div>
                
                <div className="text-sm text-trading-muted mb-2">
                  Price: ${currentSignal.price.toFixed(2)}
                </div>
                
                <div className="text-sm">
                  {currentSignal.reasoning}
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-trading-muted">LSTM Prediction</span>
                  <span className="text-trading-success">Bullish</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-trading-muted">RL Agent</span>
                  <span className="text-trading-accent">Active</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-trading-muted">Risk Level</span>
                  <span className="text-trading-warning">Medium</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-trading-muted">
              {isRunning ? 'Waiting for AI analysis...' : 'Start demo to see AI analysis'}
            </div>
          )}
        </motion.div>
      </div>

      {/* Trade History */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <h3 className="text-lg font-semibold text-trading-text mb-4">Recent Trades</h3>
        
        {tradeHistory.length > 0 ? (
          <div className="space-y-3">
            {tradeHistory.map((trade, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center justify-between p-3 bg-slate-800 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    trade.action === 'BUY' ? 'bg-trading-success' : 'bg-trading-danger'
                  }`} />
                  <div>
                    <div className="font-medium text-trading-text">
                      {trade.action} {trade.quantity} {trade.symbol}
                    </div>
                    <div className="text-sm text-trading-muted">
                      @ ${trade.price.toFixed(2)} • {trade.confidence.toFixed(1)}% confidence
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm text-trading-text">
                    {trade.timestamp.toLocaleTimeString()}
                  </div>
                  <div className="text-xs text-trading-muted truncate max-w-48">
                    {trade.reasoning}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-trading-muted">
            No trades executed yet. Start the demo to begin trading!
          </div>
        )}
      </motion.div>
    </div>
  )
} 