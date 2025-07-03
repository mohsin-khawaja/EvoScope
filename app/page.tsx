'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Brain, 
  TrendingUp, 
  BarChart3, 
  Zap, 
  Activity,
  DollarSign,
  Target,
  Shield,
  ArrowUpRight,
  ArrowDownRight,
  Eye,
  Play,
  Settings,
  FileText
} from 'lucide-react'
import Navigation from '@/components/Navigation'
import DashboardOverview from '@/components/DashboardOverview'
import LiveTradingDemo from '@/components/LiveTradingDemo'
import CryptoTradingDemo from '@/components/CryptoTradingDemo'
import ExperimentResults from '@/components/ExperimentResults'
import ModelPerformance from '@/components/ModelPerformance'
import SystemArchitecture from '@/components/SystemArchitecture'

export default function Home() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => setIsLoading(false), 1500)
    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-trading-bg flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center"
        >
          <motion.div
            animate={{ 
              rotate: 360,
              scale: [1, 1.1, 1]
            }}
            transition={{ 
              rotate: { duration: 2, repeat: Infinity, ease: "linear" },
              scale: { duration: 1, repeat: Infinity }
            }}
            className="w-16 h-16 border-4 border-trading-accent border-t-transparent rounded-full mx-auto mb-4"
          />
          <h2 className="text-xl font-semibold text-trading-text mb-2">
            Initializing AI Trading System
          </h2>
          <p className="text-trading-muted">
            Loading LSTM models and RL agents...
          </p>
        </motion.div>
      </div>
    )
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardOverview />
      case 'demo':
        return <LiveTradingDemo />
      case 'crypto':
        return <CryptoTradingDemo />
      case 'experiments':
        return <ExperimentResults />
      case 'performance':
        return <ModelPerformance />
      case 'architecture':
        return <SystemArchitecture />
      default:
        return <DashboardOverview />
    }
  }

  return (
    <div className="min-h-screen bg-trading-bg">
      {/* Header */}
      <header className="bg-trading-card border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-3"
            >
              <div className="bg-gradient-to-r from-trading-accent to-trading-success p-2 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gradient">
                  RL-LSTM AI Trading Agent
                </h1>
                <p className="text-sm text-trading-muted">
                  Advanced AI-Powered Trading System
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-4"
            >
              <div className="flex items-center space-x-2">
                <div className="status-online" />
                <span className="text-sm text-trading-muted">System Online</span>
              </div>
              
              <div className="flex items-center space-x-1 text-sm text-trading-muted">
                <Activity className="w-4 h-4" />
                <span>v1.0.0</span>
              </div>
            </motion.div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar Navigation */}
        <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Main Content */}
        <main className="flex-1 p-6">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderContent()}
          </motion.div>
        </main>
      </div>
    </div>
  )
} 