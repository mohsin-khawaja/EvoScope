'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Brain,
  Zap,
  Target,
  TrendingUp,
  Activity,
  Clock,
  BarChart3,
  Cpu
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, RadialBarChart, RadialBar, PieChart, Pie, Cell } from 'recharts'

export default function ModelPerformance() {
  const [selectedModel, setSelectedModel] = useState('lstm')

  // Training history data
  const trainingData = {
    lstm: [
      { epoch: 1, trainLoss: 0.245, valLoss: 0.267, trainAcc: 0.623, valAcc: 0.598 },
      { epoch: 5, trainLoss: 0.156, valLoss: 0.178, trainAcc: 0.734, valAcc: 0.712 },
      { epoch: 10, trainLoss: 0.098, valLoss: 0.123, trainAcc: 0.856, valAcc: 0.834 },
      { epoch: 15, trainLoss: 0.078, valLoss: 0.089, trainAcc: 0.912, valAcc: 0.889 },
      { epoch: 20, trainLoss: 0.071, valLoss: 0.084, trainAcc: 0.932, valAcc: 0.918 },
      { epoch: 25, trainLoss: 0.068, valLoss: 0.081, trainAcc: 0.941, valAcc: 0.928 },
      { epoch: 30, trainLoss: 0.065, valLoss: 0.079, trainAcc: 0.947, valAcc: 0.942 }
    ],
    rl: [
      { episode: 100, reward: -234, avgReward: -567, epsilon: 0.9, qValue: 12.3 },
      { episode: 200, reward: -89, avgReward: -298, epsilon: 0.8, qValue: 23.7 },
      { episode: 500, reward: 145, avgReward: -45, epsilon: 0.6, qValue: 45.2 },
      { episode: 800, reward: 298, avgReward: 123, epsilon: 0.4, qValue: 67.8 },
      { episode: 1000, reward: 456, avgReward: 234, epsilon: 0.3, qValue: 89.4 },
      { episode: 1200, reward: 567, avgReward: 345, epsilon: 0.2, qValue: 112.6 },
      { episode: 1500, reward: 634, avgReward: 423, epsilon: 0.12, qValue: 134.2 }
    ]
  }

  // Performance metrics
  const lstmMetrics = {
    accuracy: 94.2,
    precision: 91.8,
    recall: 89.6,
    f1Score: 88.5,
    mae: 0.089,
    rmse: 0.134
  }

  const rlMetrics = {
    totalReturn: 25.6,
    sharpeRatio: 2.52,
    winRate: 79.4,
    maxDrawdown: 8.7,
    avgTrade: 2.3,
    volatility: 15.6
  }

  // Confusion matrix data
  const confusionData = [
    { name: 'True Positive', value: 234, color: '#10b981' },
    { name: 'True Negative', value: 189, color: '#3b82f6' },
    { name: 'False Positive', value: 23, color: '#f59e0b' },
    { name: 'False Negative', value: 18, color: '#ef4444' }
  ]

  // Feature importance
  const featureImportance = [
    { feature: 'Close Price', importance: 0.23, color: '#3b82f6' },
    { feature: 'Volume', importance: 0.18, color: '#10b981' },
    { feature: 'RSI', importance: 0.15, color: '#f59e0b' },
    { feature: 'MACD', importance: 0.12, color: '#8b5cf6' },
    { feature: 'Moving Avg', importance: 0.11, color: '#ef4444' },
    { feature: 'Bollinger Bands', importance: 0.09, color: '#06b6d4' },
    { feature: 'Price Change', importance: 0.08, color: '#84cc16' },
    { feature: 'Volume Ratio', importance: 0.04, color: '#f97316' }
  ]

  const models = [
    { id: 'lstm', name: 'LSTM Model', icon: Brain },
    { id: 'rl', name: 'RL Agent', icon: Zap }
  ]

  const renderLSTMPerformance = () => (
    <div className="space-y-6">
      {/* Training Progress */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="trading-chart">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Training Loss</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData.lstm}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Line type="monotone" dataKey="trainLoss" stroke="#3b82f6" strokeWidth={2} name="Training Loss" />
              <Line type="monotone" dataKey="valLoss" stroke="#10b981" strokeWidth={2} name="Validation Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="trading-chart">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Training Accuracy</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData.lstm}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#64748b" />
              <YAxis stroke="#64748b" domain={[0.5, 1]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Line type="monotone" dataKey="trainAcc" stroke="#3b82f6" strokeWidth={2} name="Training Accuracy" />
              <Line type="monotone" dataKey="valAcc" stroke="#10b981" strokeWidth={2} name="Validation Accuracy" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* LSTM Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <div className="metric-card">
          <div className="metric-value text-trading-success">{lstmMetrics.accuracy}%</div>
          <div className="metric-label">Accuracy</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-trading-accent">{lstmMetrics.precision}%</div>
          <div className="metric-label">Precision</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-trading-warning">{lstmMetrics.recall}%</div>
          <div className="metric-label">Recall</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-trading-success">{lstmMetrics.f1Score}%</div>
          <div className="metric-label">F1 Score</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{lstmMetrics.mae}</div>
          <div className="metric-label">MAE</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{lstmMetrics.rmse}</div>
          <div className="metric-label">RMSE</div>
        </div>
      </div>

      {/* Feature Importance & Confusion Matrix */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="trading-chart">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Feature Importance</h3>
          <div className="space-y-3">
            {featureImportance.map((feature, index) => (
              <div key={feature.feature} className="flex items-center space-x-3">
                <div className="w-20 text-sm text-trading-muted">{feature.feature}</div>
                <div className="flex-1 bg-slate-800 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${feature.importance * 100}%` }}
                    transition={{ delay: index * 0.1, duration: 0.8 }}
                    className="h-2 rounded-full"
                    style={{ backgroundColor: feature.color }}
                  />
                </div>
                <div className="w-12 text-sm text-trading-text text-right">
                  {(feature.importance * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="trading-chart">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Confusion Matrix</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={confusionData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {confusionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-4">
            {confusionData.map((item, index) => (
              <div key={index} className="flex items-center space-x-2 text-sm">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-trading-muted">{item.name}: {item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )

  const renderRLPerformance = () => (
    <div className="space-y-6">
      {/* RL Training Progress */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="trading-chart">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Reward Progress</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={trainingData.rl}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="episode" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Area type="monotone" dataKey="avgReward" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              <Line type="monotone" dataKey="reward" stroke="#10b981" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="trading-chart">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Exploration vs Exploitation</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData.rl}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="episode" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <YAxis yAxisId="right" orientation="right" stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Line type="monotone" dataKey="epsilon" stroke="#f59e0b" strokeWidth={2} name="Epsilon" />
              <Line type="monotone" dataKey="qValue" stroke="#8b5cf6" strokeWidth={2} name="Q-Value" yAxisId="right" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* RL Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <div className="metric-card">
          <div className="metric-value text-trading-success">+{rlMetrics.totalReturn}%</div>
          <div className="metric-label">Total Return</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-trading-warning">{rlMetrics.sharpeRatio}</div>
          <div className="metric-label">Sharpe Ratio</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-trading-accent">{rlMetrics.winRate}%</div>
          <div className="metric-label">Win Rate</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-trading-danger">{rlMetrics.maxDrawdown}%</div>
          <div className="metric-label">Max Drawdown</div>
        </div>
        <div className="metric-card">
          <div className="metric-value text-trading-success">+{rlMetrics.avgTrade}%</div>
          <div className="metric-label">Avg Trade</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{rlMetrics.volatility}%</div>
          <div className="metric-label">Volatility</div>
        </div>
      </div>

      {/* RL Analysis */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Action Distribution</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">BUY Actions</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-slate-800 rounded-full h-2">
                  <div className="w-16 h-2 bg-trading-success rounded-full" />
                </div>
                <span className="text-trading-text w-12">34%</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">HOLD Actions</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-slate-800 rounded-full h-2">
                  <div className="w-20 h-2 bg-trading-warning rounded-full" />
                </div>
                <span className="text-trading-text w-12">42%</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">SELL Actions</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-slate-800 rounded-full h-2">
                  <div className="w-12 h-2 bg-trading-danger rounded-full" />
                </div>
                <span className="text-trading-text w-12">24%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-trading-text mb-4">Training Statistics</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
              <div className="flex items-center space-x-3">
                <Clock className="w-5 h-5 text-trading-accent" />
                <span className="text-trading-muted">Training Episodes</span>
              </div>
              <span className="text-trading-text font-medium">1,500</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
              <div className="flex items-center space-x-3">
                <Activity className="w-5 h-5 text-trading-success" />
                <span className="text-trading-muted">Final Epsilon</span>
              </div>
              <span className="text-trading-text font-medium">0.12</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
              <div className="flex items-center space-x-3">
                <Target className="w-5 h-5 text-trading-warning" />
                <span className="text-trading-muted">Learning Rate</span>
              </div>
              <span className="text-trading-text font-medium">0.0008</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
              <div className="flex items-center space-x-3">
                <BarChart3 className="w-5 h-5 text-trading-accent" />
                <span className="text-trading-muted">Batch Size</span>
              </div>
              <span className="text-trading-text font-medium">128</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-trading-text">Model Performance</h1>
          <p className="text-trading-muted mt-1">
            Detailed analysis of LSTM and RL model training and performance metrics
          </p>
        </div>
      </div>

      {/* Model Selector */}
      <div className="flex space-x-3">
        {models.map((model) => {
          const Icon = model.icon
          return (
            <button
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all ${
                selectedModel === model.id
                  ? 'bg-trading-accent text-white'
                  : 'bg-trading-card text-trading-muted hover:text-trading-text border border-slate-700 hover:border-trading-accent'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{model.name}</span>
            </button>
          )
        })}
      </div>

      {/* Content */}
      <motion.div
        key={selectedModel}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {selectedModel === 'lstm' ? renderLSTMPerformance() : renderRLPerformance()}
      </motion.div>

      {/* Model Comparison */}
      <div className="card">
        <h3 className="text-lg font-semibold text-trading-text mb-4">Model Comparison Summary</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="font-medium text-trading-text flex items-center space-x-2">
              <Brain className="w-4 h-4 text-trading-accent" />
              <span>LSTM Model</span>
            </h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-trading-muted">Architecture</div>
                <div className="text-trading-text">128 units, 2 layers</div>
              </div>
              <div>
                <div className="text-trading-muted">Parameters</div>
                <div className="text-trading-text">98,432</div>
              </div>
              <div>
                <div className="text-trading-muted">Training Time</div>
                <div className="text-trading-text">2.3 hours</div>
              </div>
              <div>
                <div className="text-trading-muted">Best Accuracy</div>
                <div className="text-trading-success">94.2%</div>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="font-medium text-trading-text flex items-center space-x-2">
              <Zap className="w-4 h-4 text-trading-warning" />
              <span>RL Agent (DQN)</span>
            </h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-trading-muted">Episodes</div>
                <div className="text-trading-text">1,500</div>
              </div>
              <div>
                <div className="text-trading-muted">Best Sharpe</div>
                <div className="text-trading-warning">2.52</div>
              </div>
              <div>
                <div className="text-trading-muted">Training Time</div>
                <div className="text-trading-text">4.7 hours</div>
              </div>
              <div>
                <div className="text-trading-muted">Win Rate</div>
                <div className="text-trading-success">79.4%</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 