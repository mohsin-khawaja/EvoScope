'use client'

import { motion } from 'framer-motion'
import { 
  LayoutDashboard,
  Play,
  FlaskConical,
  Target,
  Network,
  Settings,
  Bitcoin
} from 'lucide-react'

interface NavigationProps {
  activeTab: string
  setActiveTab: (tab: string) => void
}

export default function Navigation({ activeTab, setActiveTab }: NavigationProps) {
  const navItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: LayoutDashboard,
      description: 'System Overview'
    },
    {
      id: 'demo',
      label: 'Live Demo',
      icon: Play,
      description: 'Trading Simulation'
    },
    {
      id: 'crypto',
      label: 'Crypto Trading',
      icon: Bitcoin,
      description: 'Real Crypto Data'
    },
    {
      id: 'experiments',
      label: 'Experiments',
      icon: FlaskConical,
      description: '26+ Test Results'
    },
    {
      id: 'performance',
      label: 'Performance',
      icon: Target,
      description: 'Model Metrics'
    },
    {
      id: 'architecture',
      label: 'Architecture',
      icon: Network,
      description: 'System Design'
    }
  ]

  return (
    <aside className="w-64 bg-trading-card border-r border-slate-700 min-h-screen">
      <div className="p-6">
        <h2 className="text-lg font-semibold text-trading-text mb-6">
          System Navigation
        </h2>
        
        <nav className="space-y-2">
          {navItems.map((item, index) => {
            const Icon = item.icon
            const isActive = activeTab === item.id
            
            return (
              <motion.button
                key={item.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setActiveTab(item.id)}
                className={`nav-link w-full text-left ${isActive ? 'active' : ''}`}
              >
                <Icon className="w-5 h-5 mr-3" />
                <div className="flex-1">
                  <div className="font-medium">{item.label}</div>
                  <div className="text-xs text-trading-muted">{item.description}</div>
                </div>
                
                {isActive && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="w-1 h-8 bg-trading-accent rounded-full ml-2"
                  />
                )}
              </motion.button>
            )
          })}
        </nav>

        {/* System Status */}
        <div className="mt-8 p-4 bg-slate-800 rounded-lg">
          <h3 className="text-sm font-medium text-trading-text mb-3">System Status</h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">LSTM Model</span>
              <span className="text-trading-success">Active</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">RL Agent</span>
              <span className="text-trading-success">Training</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">Data Feed</span>
              <span className="text-trading-success">Live</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">API Status</span>
              <span className="text-trading-warning">Demo Mode</span>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-6 p-4 bg-gradient-to-r from-trading-accent/10 to-trading-success/10 rounded-lg border border-trading-accent/20">
          <h3 className="text-sm font-medium text-trading-text mb-3">Quick Stats</h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">Total Experiments</span>
              <span className="text-trading-text font-medium">26</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">Best Accuracy</span>
              <span className="text-trading-success font-medium">94.2%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-trading-muted">Sharpe Ratio</span>
              <span className="text-trading-success font-medium">2.47</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
} 