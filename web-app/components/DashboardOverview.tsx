'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Brain,
  Zap,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts'

export default function DashboardOverview() {
  const [realTimeData, setRealTimeData] = useState<any[]>([])
  const [currentTime, setCurrentTime] = useState(new Date())

  // Simulate real-time data updates
  useEffect(() => {
    const generateData = () => {
      const now = new Date()
      const data = []
      
      for (let i = 23; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60 * 60 * 1000)
        data.push({
          time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          price: 150 + Math.sin(i * 0.3) * 10 + Math.random() * 5,
          volume: 1000 + Math.random() * 500,
          prediction: 150 + Math.sin(i * 0.3 + 0.5) * 10 + Math.random() * 3,
          portfolio: 10000 + (i * 50) + Math.random() * 100
        })
      }
      return data
    }

    setRealTimeData(generateData())
    
    const interval = setInterval(() => {
      setCurrentTime(new Date())
      setRealTimeData(generateData())
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const metrics = [
    {
      title: 'Portfolio Value',
      value: '$12,485.67',
      change: '+5.23%',
      trend: 'up',
      icon: DollarSign,
      color: 'text-trading-success'
    },
    {
      title: 'Daily P&L',
      value: '+$234.56',
      change: '+1.88%',
      trend: 'up',
      icon: TrendingUp,
      color: 'text-trading-success'
    },
    {
      title: 'Accuracy',
      value: '94.2%',
      change: '+2.1%',
      trend: 'up',
      icon: Target,
      color: 'text-trading-accent'
    },
    {
      title: 'Active Trades',
      value: '3',
      change: 'AAPL, TSLA, BTC',
      trend: 'neutral',
      icon: Activity,
      color: 'text-trading-warning'
    }
  ]

  const recentTrades = [
    { symbol: 'AAPL', action: 'BUY', quantity: 10, price: 175.32, time: '10:30 AM', pnl: '+$23.40' },
    { symbol: 'TSLA', action: 'SELL', quantity: 5, price: 248.90, time: '11:15 AM', pnl: '+$67.50' },
    { symbol: 'BTC-USD', action: 'BUY', quantity: 0.1, price: 43250.00, time: '12:45 PM', pnl: '+$142.30' },
    { symbol: 'NVDA', action: 'SELL', quantity: 8, price: 456.78, time: '02:20 PM', pnl: '+$89.20' }
  ]

  const modelStats = [
    { name: 'LSTM Accuracy', value: 94.2, color: '#3b82f6' },
    { name: 'RL Win Rate', value: 78.5, color: '#10b981' },
    { name: 'Risk Score', value: 23.1, color: '#f59e0b' },
    { name: 'Confidence', value: 87.9, color: '#8b5cf6' }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-trading-text">System Dashboard</h1>
          <p className="text-trading-muted mt-1">
            Real-time AI trading system overview • Last updated: {currentTime.toLocaleTimeString()}
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 bg-trading-success/10 px-3 py-2 rounded-lg">
            <CheckCircle className="w-4 h-4 text-trading-success" />
            <span className="text-sm text-trading-success font-medium">All Systems Operational</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        {metrics.map((metric, index) => {
          const Icon = metric.icon
          return (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="metric-card"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-2 rounded-lg bg-slate-800`}>
                  <Icon className={`w-5 h-5 ${metric.color}`} />
                </div>
                <div className={`text-sm font-medium ${metric.trend === 'up' ? 'text-trading-success' : metric.trend === 'down' ? 'text-trading-danger' : 'text-trading-muted'}`}>
                  {metric.change}
                </div>
              </div>
              
              <div className="metric-value text-2xl font-bold mb-1">
                {metric.value}
              </div>
              
              <div className="metric-label">
                {metric.title}
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Price Prediction Chart */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="xl:col-span-2 trading-chart"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-trading-text">Price vs LSTM Prediction</h3>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-trading-accent rounded-full" />
                <span className="text-sm text-trading-muted">Actual Price</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-trading-success rounded-full" />
                <span className="text-sm text-trading-muted">LSTM Prediction</span>
              </div>
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={realTimeData}>
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

        {/* Model Performance */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="trading-chart"
        >
          <h3 className="text-lg font-semibold text-trading-text mb-4">Model Performance</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelStats}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#64748b" angle={-45} textAnchor="end" height={80} />
              <YAxis stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Recent Trades */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-trading-text mb-4">Recent Trades</h3>
          
          <div className="space-y-3">
            {recentTrades.map((trade, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${trade.action === 'BUY' ? 'bg-trading-success' : 'bg-trading-danger'}`} />
                  <div>
                    <div className="font-medium text-trading-text">{trade.symbol}</div>
                    <div className="text-sm text-trading-muted">{trade.action} {trade.quantity} @ ${trade.price}</div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm text-trading-success font-medium">{trade.pnl}</div>
                  <div className="text-xs text-trading-muted">{trade.time}</div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* System Alerts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-trading-text mb-4">System Alerts</h3>
          
          <div className="space-y-3">
            <div className="flex items-start space-x-3 p-3 bg-trading-success/10 border border-trading-success/20 rounded-lg">
              <CheckCircle className="w-5 h-5 text-trading-success mt-0.5" />
              <div>
                <div className="font-medium text-trading-text">Model Training Complete</div>
                <div className="text-sm text-trading-muted">LSTM model achieved 94.2% accuracy on validation set</div>
                <div className="text-xs text-trading-muted mt-1">2 minutes ago</div>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-trading-warning/10 border border-trading-warning/20 rounded-lg">
              <AlertCircle className="w-5 h-5 text-trading-warning mt-0.5" />
              <div>
                <div className="font-medium text-trading-text">High Volatility Detected</div>
                <div className="text-sm text-trading-muted">TSLA showing unusual price movements (+8.5%)</div>
                <div className="text-xs text-trading-muted mt-1">5 minutes ago</div>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-trading-accent/10 border border-trading-accent/20 rounded-lg">
              <Brain className="w-5 h-5 text-trading-accent mt-0.5" />
              <div>
                <div className="font-medium text-trading-text">RL Agent Update</div>
                <div className="text-sm text-trading-muted">DQN agent completed 1000 episodes, improving strategy</div>
                <div className="text-xs text-trading-muted mt-1">12 minutes ago</div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
} 