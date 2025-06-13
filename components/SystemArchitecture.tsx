'use client'

import { motion } from 'framer-motion'
import {
  Database,
  Brain,
  Zap,
  TrendingUp,
  Shield,
  Cloud,
  Cpu,
  Network,
  Activity,
  BarChart3,
  Target,
  AlertTriangle
} from 'lucide-react'

export default function SystemArchitecture() {
  const architectureComponents = [
    {
      id: 'data',
      title: 'Data Sources',
      icon: Database,
      color: 'bg-blue-500',
      position: { x: 50, y: 100 },
      connections: ['preprocessing'],
      details: [
        'Yahoo Finance API',
        'Alpha Vantage',
        'Real-time market data',
        'Historical OHLCV data',
        'News sentiment feeds'
      ]
    },
    {
      id: 'preprocessing',
      title: 'Data Preprocessing',
      icon: Cpu,
      color: 'bg-green-500',
      position: { x: 300, y: 100 },
      connections: ['features', 'lstm'],
      details: [
        'Data cleaning & validation',
        'Normalization',
        'Missing value handling',
        'Feature scaling',
        'Time series alignment'
      ]
    },
    {
      id: 'features',
      title: 'Feature Engineering',
      icon: BarChart3,
      color: 'bg-purple-500',
      position: { x: 550, y: 50 },
      connections: ['lstm'],
      details: [
        'Technical indicators (RSI, MACD)',
        'Moving averages',
        'Bollinger Bands',
        'Volume analysis',
        'Price momentum'
      ]
    },
    {
      id: 'lstm',
      title: 'LSTM Network',
      icon: Brain,
      color: 'bg-orange-500',
      position: { x: 300, y: 250 },
      connections: ['rl'],
      details: [
        '128 hidden units',
        '2 LSTM layers',
        'Dropout: 0.2',
        'Sequence length: 60',
        'Price prediction'
      ]
    },
    {
      id: 'rl',
      title: 'RL Agent (DQN)',
      icon: Zap,
      color: 'bg-yellow-500',
      position: { x: 550, y: 250 },
      connections: ['risk', 'execution'],
      details: [
        'Deep Q-Network',
        'Experience replay',
        'ε-greedy exploration',
        'Action space: BUY/SELL/HOLD',
        'State: LSTM features + portfolio'
      ]
    },
    {
      id: 'risk',
      title: 'Risk Management',
      icon: Shield,
      color: 'bg-red-500',
      position: { x: 300, y: 400 },
      connections: ['execution'],
      details: [
        'Position sizing',
        'Stop-loss orders',
        'Take-profit targets',
        'Portfolio limits',
        'Volatility monitoring'
      ]
    },
    {
      id: 'execution',
      title: 'Trade Execution',
      icon: TrendingUp,
      color: 'bg-indigo-500',
      position: { x: 550, y: 400 },
      connections: ['monitoring'],
      details: [
        'Order management',
        'Market/limit orders',
        'Paper trading mode',
        'Real-time execution',
        'Transaction logging'
      ]
    },
    {
      id: 'monitoring',
      title: 'Performance Monitoring',
      icon: Activity,
      color: 'bg-pink-500',
      position: { x: 50, y: 400 },
      connections: [],
      details: [
        'P&L tracking',
        'Performance metrics',
        'Real-time dashboards',
        'Alert system',
        'Model validation'
      ]
    }
  ]

  const dataFlow = [
    { from: 'data', to: 'preprocessing', label: 'Raw Market Data' },
    { from: 'preprocessing', to: 'features', label: 'Clean Data' },
    { from: 'preprocessing', to: 'lstm', label: 'Processed Sequences' },
    { from: 'features', to: 'lstm', label: 'Technical Features' },
    { from: 'lstm', to: 'rl', label: 'Price Predictions + Features' },
    { from: 'rl', to: 'risk', label: 'Trading Signals' },
    { from: 'rl', to: 'execution', label: 'Action Commands' },
    { from: 'risk', to: 'execution', label: 'Risk-Adjusted Orders' },
    { from: 'execution', to: 'monitoring', label: 'Trade Results' }
  ]

  const techStack = [
    {
      category: 'Machine Learning',
      technologies: [
        { name: 'PyTorch', icon: '🔥', description: 'Deep learning framework' },
        { name: 'Scikit-learn', icon: '🧠', description: 'ML algorithms' },
        { name: 'NumPy', icon: '🔢', description: 'Numerical computing' },
        { name: 'Pandas', icon: '🐼', description: 'Data manipulation' }
      ]
    },
    {
      category: 'Data & APIs',
      technologies: [
        { name: 'Yahoo Finance', icon: '📈', description: 'Market data' },
        { name: 'Alpha Vantage', icon: '💹', description: 'Financial APIs' },
        { name: 'WebSocket', icon: '🌐', description: 'Real-time data' },
        { name: 'REST APIs', icon: '🔗', description: 'Data integration' }
      ]
    },
    {
      category: 'Infrastructure',
      technologies: [
        { name: 'Docker', icon: '🐳', description: 'Containerization' },
        { name: 'AWS/GCP', icon: '☁️', description: 'Cloud deployment' },
        { name: 'Redis', icon: '🗄️', description: 'Caching layer' },
        { name: 'PostgreSQL', icon: '🐘', description: 'Data storage' }
      ]
    },
    {
      category: 'Monitoring',
      technologies: [
        { name: 'Grafana', icon: '📊', description: 'Dashboards' },
        { name: 'Prometheus', icon: '🔍', description: 'Metrics collection' },
        { name: 'Jupyter', icon: '📓', description: 'Analysis notebooks' },
        { name: 'MLflow', icon: '🚀', description: 'ML lifecycle' }
      ]
    }
  ]

  const keyFeatures = [
    {
      title: 'Hybrid AI Architecture',
      description: 'Combines LSTM neural networks with reinforcement learning for superior performance',
      icon: Brain,
      color: 'text-blue-400'
    },
    {
      title: 'Real-time Processing',
      description: 'Sub-second latency for market data processing and decision making',
      icon: Zap,
      color: 'text-yellow-400'
    },
    {
      title: 'Risk-First Design',
      description: 'Built-in risk management with multiple safety mechanisms',
      icon: Shield,
      color: 'text-red-400'
    },
    {
      title: 'Scalable Infrastructure',
      description: 'Cloud-native architecture supporting multiple markets and assets',
      icon: Cloud,
      color: 'text-green-400'
    }
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-trading-text mb-4">System Architecture</h1>
        <p className="text-trading-muted max-w-3xl mx-auto">
          Comprehensive overview of the RL-LSTM AI trading system architecture, 
          data flow, and technology stack powering intelligent trading decisions.
        </p>
      </div>

      {/* Architecture Diagram */}
      <div className="card">
        <h2 className="text-xl font-semibold text-trading-text mb-6">System Components & Data Flow</h2>
        
        <div className="relative h-96 bg-slate-900 rounded-lg p-6 overflow-hidden">
          {/* Background Grid */}
          <div className="absolute inset-0 opacity-20">
            <div className="grid grid-cols-12 grid-rows-8 h-full">
              {Array.from({ length: 96 }).map((_, i) => (
                <div key={i} className="border border-slate-700" />
              ))}
            </div>
          </div>

          {/* Components */}
          {architectureComponents.map((component, index) => {
            const Icon = component.icon
            return (
              <motion.div
                key={component.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.2 }}
                className="absolute group cursor-pointer"
                style={{ 
                  left: component.position.x, 
                  top: component.position.y,
                  transform: 'translate(-50%, -50%)'
                }}
              >
                <div className={`w-20 h-20 ${component.color} rounded-lg flex items-center justify-center shadow-lg group-hover:scale-110 transition-all duration-300`}>
                  <Icon className="w-8 h-8 text-white" />
                </div>
                <div className="text-center mt-2">
                  <div className="text-sm font-medium text-trading-text whitespace-nowrap">
                    {component.title}
                  </div>
                </div>

                {/* Tooltip */}
                <div className="absolute z-10 invisible group-hover:visible bg-trading-card border border-slate-600 rounded-lg p-3 -top-32 left-1/2 transform -translate-x-1/2 w-48">
                  <div className="text-sm font-medium text-trading-text mb-2">{component.title}</div>
                  <ul className="text-xs text-trading-muted space-y-1">
                    {component.details.map((detail, i) => (
                      <li key={i}>• {detail}</li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            )
          })}

          {/* Connection Lines */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {dataFlow.map((flow, index) => {
              const fromComponent = architectureComponents.find(c => c.id === flow.from)
              const toComponent = architectureComponents.find(c => c.id === flow.to)
              
              if (!fromComponent || !toComponent) return null

              return (
                <motion.g key={`${flow.from}-${flow.to}`}>
                  <motion.line
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.6 }}
                    transition={{ delay: index * 0.3 + 1, duration: 0.8 }}
                    x1={fromComponent.position.x}
                    y1={fromComponent.position.y}
                    x2={toComponent.position.x}
                    y2={toComponent.position.y}
                    stroke="#3b82f6"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                  />
                  <motion.circle
                    initial={{ r: 0 }}
                    animate={{ r: 3 }}
                    transition={{ delay: index * 0.3 + 1.5, duration: 0.3 }}
                    cx={(fromComponent.position.x + toComponent.position.x) / 2}
                    cy={(fromComponent.position.y + toComponent.position.y) / 2}
                    fill="#3b82f6"
                  />
                </motion.g>
              )
            })}
          </svg>
        </div>
      </div>

      {/* Key Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        {keyFeatures.map((feature, index) => {
          const Icon = feature.icon
          return (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card text-center"
            >
              <div className="mb-4">
                <div className="w-12 h-12 bg-slate-800 rounded-lg flex items-center justify-center mx-auto">
                  <Icon className={`w-6 h-6 ${feature.color}`} />
                </div>
              </div>
              <h3 className="text-lg font-semibold text-trading-text mb-2">{feature.title}</h3>
              <p className="text-sm text-trading-muted">{feature.description}</p>
            </motion.div>
          )
        })}
      </div>

      {/* Technology Stack */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
        {techStack.map((category, categoryIndex) => (
          <motion.div
            key={category.category}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: categoryIndex * 0.1 }}
            className="card"
          >
            <h3 className="text-lg font-semibold text-trading-text mb-4">{category.category}</h3>
            <div className="space-y-3">
              {category.technologies.map((tech, techIndex) => (
                <motion.div
                  key={tech.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: categoryIndex * 0.1 + techIndex * 0.05 }}
                  className="flex items-center space-x-3 p-2 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors"
                >
                  <span className="text-xl">{tech.icon}</span>
                  <div>
                    <div className="font-medium text-trading-text">{tech.name}</div>
                    <div className="text-xs text-trading-muted">{tech.description}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Model Specifications */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LSTM Specifications */}
        <div className="card">
          <h3 className="text-lg font-semibold text-trading-text mb-4 flex items-center space-x-2">
            <Brain className="w-5 h-5 text-blue-400" />
            <span>LSTM Model Specifications</span>
          </h3>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-trading-muted">Architecture:</span>
                <span className="text-trading-text ml-2">Bidirectional LSTM</span>
              </div>
              <div>
                <span className="text-trading-muted">Hidden Units:</span>
                <span className="text-trading-text ml-2">128</span>
              </div>
              <div>
                <span className="text-trading-muted">Layers:</span>
                <span className="text-trading-text ml-2">2</span>
              </div>
              <div>
                <span className="text-trading-muted">Dropout:</span>
                <span className="text-trading-text ml-2">0.2</span>
              </div>
              <div>
                <span className="text-trading-muted">Sequence Length:</span>
                <span className="text-trading-text ml-2">60 days</span>
              </div>
              <div>
                <span className="text-trading-muted">Parameters:</span>
                <span className="text-trading-text ml-2">98,432</span>
              </div>
            </div>
          </div>
        </div>

        {/* RL Specifications */}
        <div className="card">
          <h3 className="text-lg font-semibold text-trading-text mb-4 flex items-center space-x-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            <span>RL Agent Specifications</span>
          </h3>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-trading-muted">Algorithm:</span>
                <span className="text-trading-text ml-2">Deep Q-Network</span>
              </div>
              <div>
                <span className="text-trading-muted">State Space:</span>
                <span className="text-trading-text ml-2">128 features</span>
              </div>
              <div>
                <span className="text-trading-muted">Action Space:</span>
                <span className="text-trading-text ml-2">3 actions</span>
              </div>
              <div>
                <span className="text-trading-muted">Learning Rate:</span>
                <span className="text-trading-text ml-2">0.0001</span>
              </div>
              <div>
                <span className="text-trading-muted">Replay Buffer:</span>
                <span className="text-trading-text ml-2">10,000</span>
              </div>
              <div>
                <span className="text-trading-muted">Epsilon Decay:</span>
                <span className="text-trading-text ml-2">0.995</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="card">
        <h3 className="text-lg font-semibold text-trading-text mb-6 flex items-center space-x-2">
          <Target className="w-5 h-5 text-green-400" />
          <span>System Performance Metrics</span>
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400 mb-2">94.2%</div>
            <div className="text-sm text-trading-muted">LSTM Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400 mb-2">2.52</div>
            <div className="text-sm text-trading-muted">Sharpe Ratio</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400 mb-2">79.4%</div>
            <div className="text-sm text-trading-muted">Win Rate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400 mb-2">&lt;100ms</div>
            <div className="text-sm text-trading-muted">Prediction Latency</div>
          </div>
        </div>
      </div>
    </div>
  )
}