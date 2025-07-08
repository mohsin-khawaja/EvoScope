#!/usr/bin/env python3
"""
🚀 Alpaca Paper Trading Tracker

This module provides comprehensive tracking for your Alpaca paper trading:
- Real-time portfolio monitoring
- Trade history and performance analysis
- Risk management alerts
- Daily/weekly/monthly reports
- Integration with your AI trading system
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing config
from src.utils.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

class AlpacaTracker:
    """Comprehensive tracking system for Alpaca paper trading"""
    
    def __init__(self, db_path: str = "data/alpaca_tracking.db"):
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.base_url = ALPACA_BASE_URL
        self.db_path = db_path
        
        # API headers
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        # Initialize database
        self._init_database()
        
        # Performance tracking
        self.performance_metrics = {}
        self.alerts = []
        
    def _init_database(self):
        """Initialize SQLite database for tracking"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                timestamp DATETIME,
                status TEXT,
                profit_loss REAL,
                commission REAL,
                ai_confidence REAL,
                ai_reasoning TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                portfolio_value REAL,
                cash REAL,
                equity REAL,
                buying_power REAL,
                day_trade_buying_power REAL,
                daily_pnl REAL,
                total_pnl REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_return REAL,
                daily_return REAL,
                volatility REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                num_trades INTEGER,
                avg_trade_return REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            response = requests.get(f"{self.base_url}/account", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting account info: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            response = requests.get(f"{self.base_url}/positions", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def get_orders(self, status='all', limit=100) -> List[Dict]:
        """Get order history"""
        try:
            params = {'status': status, 'limit': limit}
            response = requests.get(f"{self.base_url}/orders", headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []
    
    def get_portfolio_history(self, period='1M', timeframe='1D') -> Dict:
        """Get portfolio performance history"""
        try:
            params = {'period': period, 'timeframe': timeframe}
            response = requests.get(f"{self.base_url}/account/portfolio/history", 
                                  headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            print(f"Error getting portfolio history: {e}")
            return {}
    
    def record_trade(self, order_data: Dict, ai_confidence: float = 0, ai_reasoning: str = ""):
        """Record a trade in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO trades 
                (order_id, symbol, side, quantity, price, timestamp, status, ai_confidence, ai_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order_data['id'],
                order_data['symbol'],
                order_data['side'],
                float(order_data['qty']),
                float(order_data.get('filled_avg_price', 0)),
                order_data['created_at'],
                order_data['status'],
                ai_confidence,
                ai_reasoning
            ))
            conn.commit()
            print(f"✅ Trade recorded: {order_data['side']} {order_data['qty']} {order_data['symbol']}")
        except Exception as e:
            print(f"Error recording trade: {e}")
        finally:
            conn.close()
    
    def record_portfolio_snapshot(self):
        """Record current portfolio state"""
        account_info = self.get_account_info()
        if not account_info:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO portfolio_snapshots 
                (timestamp, portfolio_value, cash, equity, buying_power, day_trade_buying_power, daily_pnl, total_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                float(account_info.get('portfolio_value', 0)),
                float(account_info.get('cash', 0)),
                float(account_info.get('equity', 0)),
                float(account_info.get('buying_power', 0)),
                float(account_info.get('day_trade_buying_power', 0)),
                float(account_info.get('unrealized_pl', 0)),
                float(account_info.get('unrealized_pl', 0))
            ))
            conn.commit()
        except Exception as e:
            print(f"Error recording portfolio snapshot: {e}")
        finally:
            conn.close()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get portfolio history
        portfolio_df = pd.read_sql_query('''
            SELECT * FROM portfolio_snapshots 
            ORDER BY timestamp DESC LIMIT 30
        ''', conn)
        
        # Get trade history
        trades_df = pd.read_sql_query('''
            SELECT * FROM trades 
            WHERE status = 'filled'
            ORDER BY timestamp DESC
        ''', conn)
        
        conn.close()
        
        if portfolio_df.empty:
            return {}
        
        # Calculate metrics
        portfolio_values = portfolio_df['portfolio_value'].values
        if len(portfolio_values) < 2:
            return {}
        
        # Returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[0] - portfolio_values[-1]) / portfolio_values[-1] * 100
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown
        running_max = np.maximum.accumulate(portfolio_values[::-1])[::-1]
        drawdown = (portfolio_values - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # Trade metrics
        win_rate = 0
        avg_trade_return = 0
        if not trades_df.empty:
            profitable_trades = trades_df[trades_df['profit_loss'] > 0]
            win_rate = len(profitable_trades) / len(trades_df) * 100
            avg_trade_return = trades_df['profit_loss'].mean()
        
        metrics = {
            'total_return': total_return,
            'daily_return': np.mean(returns) * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades_df),
            'avg_trade_return': avg_trade_return,
            'current_portfolio_value': portfolio_values[0],
            'current_cash': portfolio_df['cash'].iloc[0]
        }
        
        return metrics
    
    def add_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """Add an alert to the system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, message, severity)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now(), alert_type, message, severity))
            conn.commit()
            
            # Print alert
            severity_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🚨"}
            print(f"{severity_icon.get(severity, 'ℹ️')} {alert_type}: {message}")
            
        except Exception as e:
            print(f"Error adding alert: {e}")
        finally:
            conn.close()
    
    def check_risk_alerts(self):
        """Check for risk management alerts"""
        metrics = self.calculate_performance_metrics()
        if not metrics:
            return
        
        # Loss alerts
        if metrics['total_return'] < -10:
            self.add_alert("RISK_ALERT", 
                         f"Portfolio down {metrics['total_return']:.2f}% - Consider stopping trading", 
                         "ERROR")
        
        # Volatility alerts
        if metrics['volatility'] > 30:
            self.add_alert("VOLATILITY_ALERT", 
                         f"High volatility detected: {metrics['volatility']:.2f}%", 
                         "WARNING")
        
        # Drawdown alerts
        if metrics['max_drawdown'] < -15:
            self.add_alert("DRAWDOWN_ALERT", 
                         f"Maximum drawdown: {metrics['max_drawdown']:.2f}%", 
                         "WARNING")
        
        # Win rate alerts
        if metrics['win_rate'] < 40 and metrics['num_trades'] > 10:
            self.add_alert("WIN_RATE_ALERT", 
                         f"Low win rate: {metrics['win_rate']:.1f}%", 
                         "WARNING")
    
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        metrics = self.calculate_performance_metrics()
        account_info = self.get_account_info()
        positions = self.get_positions()
        
        if not metrics:
            return "No data available for report"
        
        report = f"""
📊 DAILY ALPACA TRADING REPORT - {datetime.now().strftime('%Y-%m-%d')}
{'='*60}

💰 PORTFOLIO SUMMARY:
   Portfolio Value: ${metrics['current_portfolio_value']:,.2f}
   Cash Balance: ${metrics['current_cash']:,.2f}
   Total Return: {metrics['total_return']:+.2f}%
   Daily Return: {metrics['daily_return']:+.2f}%

📈 PERFORMANCE METRICS:
   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
   Volatility: {metrics['volatility']:.2f}%
   Max Drawdown: {metrics['max_drawdown']:.2f}%
   Win Rate: {metrics['win_rate']:.1f}%

🔢 TRADING ACTIVITY:
   Total Trades: {metrics['num_trades']}
   Average Trade Return: {metrics['avg_trade_return']:+.2f}%
   Current Positions: {len(positions)}

📋 CURRENT POSITIONS:
"""
        
        for position in positions:
            unrealized_pl = float(position.get('unrealized_pl', 0))
            unrealized_plpc = float(position.get('unrealized_plpc', 0)) * 100
            report += f"   {position['symbol']}: {position['qty']} shares | "
            report += f"P&L: ${unrealized_pl:+,.2f} ({unrealized_plpc:+.2f}%)\n"
        
        if not positions:
            report += "   No open positions\n"
        
        # Add recent alerts
        conn = sqlite3.connect(self.db_path)
        recent_alerts = pd.read_sql_query('''
            SELECT * FROM alerts 
            WHERE timestamp >= datetime('now', '-1 day')
            ORDER BY timestamp DESC LIMIT 5
        ''', conn)
        conn.close()
        
        if not recent_alerts.empty:
            report += "\n🚨 RECENT ALERTS:\n"
            for _, alert in recent_alerts.iterrows():
                report += f"   {alert['alert_type']}: {alert['message']}\n"
        
        return report
    
    def create_performance_chart(self, save_path: str = "data/performance_chart.png"):
        """Create performance visualization"""
        conn = sqlite3.connect(self.db_path)
        portfolio_df = pd.read_sql_query('''
            SELECT timestamp, portfolio_value, daily_pnl 
            FROM portfolio_snapshots 
            ORDER BY timestamp DESC LIMIT 30
        ''', conn)
        conn.close()
        
        if portfolio_df.empty:
            print("No data available for chart")
            return
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
                color='blue', linewidth=2)
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Daily P&L
        ax2.bar(portfolio_df['timestamp'], portfolio_df['daily_pnl'], 
                color=['green' if x > 0 else 'red' for x in portfolio_df['daily_pnl']])
        ax2.set_title('Daily P&L', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Daily P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance chart saved to {save_path}")
    
    def sync_trades(self):
        """Sync recent trades with database"""
        orders = self.get_orders(status='filled', limit=50)
        
        for order in orders:
            try:
                # Check if trade already exists
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM trades WHERE order_id = ?', (order['id'],))
                
                if not cursor.fetchone():
                    self.record_trade(order)
                
                conn.close()
            except Exception as e:
                print(f"Error syncing trade {order['id']}: {e}")
    
    def run_daily_update(self):
        """Run daily tracking update"""
        print("🔄 Running daily Alpaca tracking update...")
        
        # Record portfolio snapshot
        self.record_portfolio_snapshot()
        
        # Sync trades
        self.sync_trades()
        
        # Check for alerts
        self.check_risk_alerts()
        
        # Generate report
        report = self.generate_daily_report()
        
        # Save report
        report_path = f"data/daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Create chart
        self.create_performance_chart()
        
        print("✅ Daily update completed!")
        print(report)
        
        return report

# Convenience functions
def track_alpaca_performance():
    """Quick function to track performance"""
    tracker = AlpacaTracker()
    return tracker.run_daily_update()

def get_alpaca_dashboard():
    """Get dashboard summary"""
    tracker = AlpacaTracker()
    return {
        'account_info': tracker.get_account_info(),
        'positions': tracker.get_positions(),
        'performance': tracker.calculate_performance_metrics(),
        'recent_orders': tracker.get_orders(limit=10)
    }

if __name__ == "__main__":
    # Run daily tracking
    tracker = AlpacaTracker()
    tracker.run_daily_update() 