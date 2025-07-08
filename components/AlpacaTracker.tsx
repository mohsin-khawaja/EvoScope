'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Activity,
  Eye,
  BarChart3,
  PieChart,
  Zap,
  Shield,
  Clock,
  Database,
  Settings,
  Download,
  Bell
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts'

interface AlpacaTrackerData {
  account_info: any
  positions: any[]
  performance: any
  recent_orders: any[]
  timestamp?: string
}

interface Alert {
  id: string
  type: string
  message: string
  severity: 'INFO' | 'WARNING' | 'ERROR'
  timestamp: string
}

export default function AlpacaTracker() {
  const [trackerData, setTrackerData] = useState<AlpacaTrackerData | null>(null)
  const [performanceData, setPerformanceData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [autoRefresh, setAutoRefresh] = useState(false)

  // Fetch dashboard data
  const fetchDashboardData = async () => {
    try {
      setIsLoading(true)
      setError(null)
      
      const response = await fetch('/api/alpaca-tracker?action=dashboard')
      const result = await response.json()
      
      if (result.success) {
        setTrackerData(result.data)
        setIsConnected(true)
        setLastUpdate(new Date().toLocaleTimeString())
        
        // Check for alerts based on performance
        if (result.data.performance?.total_return < -10) {
          addAlert('RISK', 'Portfolio down more than 10%', 'ERROR')
        }
      } else {
        setError(result.error || 'Failed to fetch dashboard data')
        setIsConnected(false)
      }
    } catch (err) {
      setError('Network error: Could not connect to tracker')
      setIsConnected(false)
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch performance metrics
  const fetchPerformanceData = async () => {
    try {
      const response = await fetch('/api/alpaca-tracker?action=performance')
      const result = await response.json()
      
      if (result.success) {
        setPerformanceData(result.data)
      }
    } catch (err) {
      console.error('Error fetching performance data:', err)
    }
  }

  // Test connection
  const testConnection = async () => {
    try {
      setIsLoading(true)
      const response = await fetch('/api/alpaca-tracker?action=test-connection')
      const result = await response.json()
      
      if (result.success) {
        setIsConnected(true)
        addAlert('CONNECTION', 'Connection test successful', 'INFO')
      } else {
        setIsConnected(false)
        addAlert('CONNECTION', 'Connection test failed', 'ERROR')
      }
    } catch (err) {
      setIsConnected(false)
      addAlert('CONNECTION', 'Connection test failed', 'ERROR')
    } finally {
      setIsLoading(false)
    }
  }

  // Record portfolio snapshot
  const recordSnapshot = async () => {
    try {
      setIsLoading(true)
      const response = await fetch('/api/alpaca-tracker', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'record-snapshot' })
      })
      
      const result = await response.json()
      
      if (result.success) {
        addAlert('SNAPSHOT', 'Portfolio snapshot recorded', 'INFO')
        fetchDashboardData() // Refresh data
      } else {
        addAlert('SNAPSHOT', 'Failed to record snapshot', 'ERROR')
      }
    } catch (err) {
      addAlert('SNAPSHOT', 'Error recording snapshot', 'ERROR')
    } finally {
      setIsLoading(false)
    }
  }

  // Run daily update
  const runDailyUpdate = async () => {
    try {
      setIsLoading(true)
      const response = await fetch('/api/alpaca-tracker', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'run-daily-update' })
      })
      
      const result = await response.json()
      
      if (result.success) {
        addAlert('UPDATE', 'Daily update completed', 'INFO')
        fetchDashboardData()
        fetchPerformanceData()
      } else {
        addAlert('UPDATE', 'Daily update failed', 'ERROR')
      }
    } catch (err) {
      addAlert('UPDATE', 'Error running daily update', 'ERROR')
    } finally {
      setIsLoading(false)
    }
  }

  // Add alert
  const addAlert = (type: string, message: string, severity: 'INFO' | 'WARNING' | 'ERROR') => {
    const newAlert: Alert = {
      id: Date.now().toString(),
      type,
      message,
      severity,
      timestamp: new Date().toLocaleTimeString()
    }
    
    setAlerts(prev => [newAlert, ...prev.slice(0, 9)]) // Keep last 10 alerts
  }

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchDashboardData()
      }, 30000) // Refresh every 30 seconds
      
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  // Initial load
  useEffect(() => {
    fetchDashboardData()
    fetchPerformanceData()
  }, [])

  // Get alert icon
  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'ERROR': return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'WARNING': return <AlertCircle className="w-4 h-4 text-yellow-500" />
      default: return <CheckCircle className="w-4 h-4 text-green-500" />
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gradient">Alpaca Paper Trading Tracker</h1>
          <p className="text-trading-muted">Real-time portfolio monitoring and performance analytics</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-trading-muted">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-trading-muted" />
            <span className="text-sm text-trading-muted">
              Last updated: {lastUpdate}
            </span>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="trading-chart p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-trading-text">Control Panel</h3>
          <div className="flex items-center space-x-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm text-trading-muted">Auto-refresh</span>
            </label>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <button
            onClick={fetchDashboardData}
            disabled={isLoading}
            className="btn-primary flex items-center justify-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          
          <button
            onClick={testConnection}
            disabled={isLoading}
            className="btn-secondary flex items-center justify-center space-x-2"
          >
            <Activity className="w-4 h-4" />
            <span>Test Connection</span>
          </button>
          
          <button
            onClick={recordSnapshot}
            disabled={isLoading}
            className="btn-secondary flex items-center justify-center space-x-2"
          >
            <Database className="w-4 h-4" />
            <span>Record Snapshot</span>
          </button>
          
          <button
            onClick={runDailyUpdate}
            disabled={isLoading}
            className="btn-accent flex items-center justify-center space-x-2"
          >
            <Zap className="w-4 h-4" />
            <span>Daily Update</span>
          </button>
          
          <button
            onClick={fetchPerformanceData}
            disabled={isLoading}
            className="btn-secondary flex items-center justify-center space-x-2"
          >
            <BarChart3 className="w-4 h-4" />
            <span>Performance</span>
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-500/10 border border-red-500/20 rounded-lg p-4"
        >
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="text-red-400">{error}</span>
          </div>
        </motion.div>
      )}

      {/* Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* Account Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="trading-chart"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-trading-text">Account Overview</h3>
            <DollarSign className="w-5 h-5 text-trading-accent" />
          </div>
          
          {trackerData?.account_info ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-trading-muted">Portfolio Value</p>
                  <p className="text-xl font-bold text-trading-text">
                    ${parseFloat(trackerData.account_info.portfolio_value || 0).toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-trading-muted">Cash Balance</p>
                  <p className="text-xl font-bold text-trading-text">
                    ${parseFloat(trackerData.account_info.cash || 0).toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-trading-muted">Buying Power</p>
                  <p className="text-lg font-semibold text-trading-text">
                    ${parseFloat(trackerData.account_info.buying_power || 0).toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-trading-muted">Status</p>
                  <p className={`text-lg font-semibold ${
                    trackerData.account_info.status === 'ACTIVE' ? 'text-green-500' : 'text-yellow-500'
                  }`}>
                    {trackerData.account_info.status || 'Unknown'}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-trading-muted">No account data available</p>
              <button onClick={fetchDashboardData} className="btn-primary mt-2">
                Load Data
              </button>
            </div>
          )}
        </motion.div>

        {/* Performance Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="trading-chart"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-trading-text">Performance Metrics</h3>
            <TrendingUp className="w-5 h-5 text-trading-success" />
          </div>
          
          {trackerData?.performance && Object.keys(trackerData.performance).length > 0 ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 gap-3">
                <div className="flex justify-between items-center">
                  <span className="text-trading-muted">Total Return</span>
                  <span className={`font-semibold ${
                    trackerData.performance.total_return > 0 ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {trackerData.performance.total_return?.toFixed(2) || 0}%
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-trading-muted">Win Rate</span>
                  <span className="font-semibold text-trading-text">
                    {trackerData.performance.win_rate?.toFixed(1) || 0}%
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-trading-muted">Total Trades</span>
                  <span className="font-semibold text-trading-text">
                    {trackerData.performance.num_trades || 0}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-trading-muted">Sharpe Ratio</span>
                  <span className="font-semibold text-trading-text">
                    {trackerData.performance.sharpe_ratio?.toFixed(2) || 0}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-trading-muted">Max Drawdown</span>
                  <span className="font-semibold text-red-500">
                    {trackerData.performance.max_drawdown?.toFixed(2) || 0}%
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-trading-muted">No performance data available</p>
              <p className="text-sm text-trading-muted mt-2">
                Start trading to see metrics
              </p>
            </div>
          )}
        </motion.div>

        {/* Current Positions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="trading-chart"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-trading-text">Current Positions</h3>
            <PieChart className="w-5 h-5 text-trading-accent" />
          </div>
          
          {trackerData?.positions && trackerData.positions.length > 0 ? (
            <div className="space-y-3">
              {trackerData.positions.map((position, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <div>
                    <p className="font-medium text-trading-text">{position.symbol}</p>
                    <p className="text-sm text-trading-muted">{position.qty} shares</p>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-trading-text">
                      ${parseFloat(position.market_value || 0).toLocaleString()}
                    </p>
                    <p className={`text-sm ${
                      parseFloat(position.unrealized_pl || 0) > 0 ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {parseFloat(position.unrealized_pl || 0) > 0 ? '+' : ''}
                      ${parseFloat(position.unrealized_pl || 0).toFixed(2)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-trading-muted">No positions found</p>
              <p className="text-sm text-trading-muted mt-2">
                Start trading to see positions
              </p>
            </div>
          )}
        </motion.div>
      </div>

      {/* Recent Orders & Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Orders */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="trading-chart"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-trading-text">Recent Orders</h3>
            <Activity className="w-5 h-5 text-trading-accent" />
          </div>
          
          {trackerData?.recent_orders && trackerData.recent_orders.length > 0 ? (
            <div className="space-y-2">
              {trackerData.recent_orders.slice(0, 5).map((order, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <div>
                    <p className="font-medium text-trading-text">{order.symbol}</p>
                    <p className="text-sm text-trading-muted">
                      {order.side?.toUpperCase()} {order.qty} shares
                    </p>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-medium ${
                      order.status === 'filled' ? 'text-green-500' : 
                      order.status === 'pending' ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {order.status?.toUpperCase()}
                    </p>
                    <p className="text-xs text-trading-muted">
                      {order.created_at ? new Date(order.created_at).toLocaleString() : ''}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-trading-muted">No recent orders</p>
            </div>
          )}
        </motion.div>

        {/* Alerts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="trading-chart"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-trading-text">System Alerts</h3>
            <Bell className="w-5 h-5 text-trading-accent" />
          </div>
          
          {alerts.length > 0 ? (
            <div className="space-y-2">
              {alerts.map((alert) => (
                <div key={alert.id} className="flex items-center space-x-3 p-3 bg-slate-800/50 rounded-lg">
                  {getAlertIcon(alert.severity)}
                  <div className="flex-1">
                    <p className="text-sm font-medium text-trading-text">{alert.message}</p>
                    <p className="text-xs text-trading-muted">{alert.type} • {alert.timestamp}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-trading-muted">No alerts</p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
} 