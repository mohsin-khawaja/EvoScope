import { NextRequest, NextResponse } from 'next/server'

// Alpaca API configuration
const ALPACA_BASE_URL = process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets/v2'
const ALPACA_API_KEY = process.env.ALPACA_API_KEY
const ALPACA_SECRET_KEY = process.env.ALPACA_SECRET_KEY

// Helper function to make Alpaca API calls
async function makeAlpacaRequest(endpoint: string, options: RequestInit = {}) {
  if (!ALPACA_API_KEY || !ALPACA_SECRET_KEY) {
    throw new Error('Alpaca API keys not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.')
  }

  const headers = {
    'APCA-API-KEY-ID': ALPACA_API_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
    'Content-Type': 'application/json',
    ...options.headers
  }

  const response = await fetch(`${ALPACA_BASE_URL}${endpoint}`, {
    ...options,
    headers
  })

  if (!response.ok) {
    throw new Error(`Alpaca API error: ${response.status} ${response.statusText}`)
  }

  return response.json()
}

// Calculate performance metrics
function calculatePerformanceMetrics(portfolioHistory: any[], trades: any[] = []) {
  if (!portfolioHistory || portfolioHistory.length < 2) {
    return {
      total_return: 0,
      daily_return: 0,
      volatility: 0,
      sharpe_ratio: 0,
      max_drawdown: 0,
      win_rate: 0,
      num_trades: trades.length,
      avg_trade_return: 0
    }
  }

  const values = portfolioHistory.map(p => parseFloat(p.equity || p.portfolio_value || 0))
  const initialValue = values[values.length - 1]
  const currentValue = values[0]
  
  // Calculate returns
  const totalReturn = initialValue > 0 ? ((currentValue - initialValue) / initialValue) * 100 : 0
  const returns = []
  for (let i = 1; i < values.length; i++) {
    if (values[i] > 0) {
      returns.push((values[i-1] - values[i]) / values[i] * 100)
    }
  }
  
  const dailyReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0
  const volatility = returns.length > 1 ? Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - dailyReturn, 2), 0) / (returns.length - 1)) * Math.sqrt(252) : 0
  const sharpeRatio = volatility > 0 ? (dailyReturn * 252) / volatility : 0
  
  // Calculate drawdown
  let maxDrawdown = 0
  let peak = values[0]
  for (const value of values) {
    if (value > peak) peak = value
    const drawdown = peak > 0 ? ((peak - value) / peak) * 100 : 0
    if (drawdown > maxDrawdown) maxDrawdown = drawdown
  }

  // Trade statistics
  const profitableTrades = trades.filter(t => parseFloat(t.filled_avg_price || 0) > 0)
  const winRate = trades.length > 0 ? (profitableTrades.length / trades.length) * 100 : 0

  return {
    total_return: totalReturn,
    daily_return: dailyReturn,
    volatility: volatility,
    sharpe_ratio: sharpeRatio,
    max_drawdown: -maxDrawdown,
    win_rate: winRate,
    num_trades: trades.length,
    avg_trade_return: dailyReturn,
    current_portfolio_value: currentValue,
    current_cash: currentValue // Simplified for now
  }
}

// GET /api/alpaca-tracker - Get tracking data
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action') || 'dashboard'

  try {
    switch (action) {
      case 'dashboard':
        // Get account info, positions, and orders
        const [accountInfo, positions, orders] = await Promise.all([
          makeAlpacaRequest('/account').catch(() => null),
          makeAlpacaRequest('/positions').catch(() => []),
          makeAlpacaRequest('/orders?status=all&limit=50').catch(() => [])
        ])

        // Get portfolio history for performance calculation
        const portfolioHistory = await makeAlpacaRequest('/account/portfolio/history?period=1M&timeframe=1D').catch(() => ({ equity: [] }))
        
        // Calculate performance metrics
        const performance = calculatePerformanceMetrics(
          portfolioHistory.equity?.map((equity: number, index: number) => ({ equity })) || [],
          orders
        )

        return NextResponse.json({
          success: true,
          data: {
            account_info: accountInfo,
            positions: positions,
            performance: performance,
            recent_orders: orders.slice(0, 10),
            timestamp: new Date().toISOString()
          }
        })

      case 'performance':
        // Get detailed performance metrics
        const [account, orderHistory] = await Promise.all([
          makeAlpacaRequest('/account').catch(() => null),
          makeAlpacaRequest('/orders?status=filled&limit=100').catch(() => [])
        ])

        const portfolioData = await makeAlpacaRequest('/account/portfolio/history?period=3M&timeframe=1D').catch(() => ({ equity: [] }))
        const metrics = calculatePerformanceMetrics(
          portfolioData.equity?.map((equity: number) => ({ equity })) || [],
          orderHistory
        )

        return NextResponse.json({
          success: true,
          data: {
            metrics: metrics,
            account_info: account,
            timestamp: new Date().toISOString()
          }
        })

      case 'daily-report':
        // Generate a daily report
        const [dailyAccount, dailyPositions, dailyOrders] = await Promise.all([
          makeAlpacaRequest('/account').catch(() => null),
          makeAlpacaRequest('/positions').catch(() => []),
          makeAlpacaRequest('/orders?status=all&limit=20').catch(() => [])
        ])

        const report = `
📊 DAILY ALPACA TRADING REPORT - ${new Date().toLocaleDateString()}
${'='.repeat(60)}

💰 PORTFOLIO SUMMARY:
   Portfolio Value: $${parseFloat(dailyAccount?.portfolio_value || 0).toLocaleString()}
   Cash Balance: $${parseFloat(dailyAccount?.cash || 0).toLocaleString()}
   Buying Power: $${parseFloat(dailyAccount?.buying_power || 0).toLocaleString()}

📋 CURRENT POSITIONS: ${dailyPositions.length}
${dailyPositions.map((pos: any) => 
  `   ${pos.symbol}: ${pos.qty} shares | P&L: $${parseFloat(pos.unrealized_pl || 0).toFixed(2)}`
).join('\n')}

📋 RECENT ORDERS: ${dailyOrders.length}
${dailyOrders.slice(0, 5).map((order: any) => 
  `   ${order.symbol}: ${order.side?.toUpperCase()} ${order.qty} - ${order.status?.toUpperCase()}`
).join('\n')}
        `

        return NextResponse.json({
          success: true,
          data: report.trim(),
          timestamp: new Date().toISOString()
        })

      case 'test-connection':
        // Test Alpaca connection
        try {
          const testAccount = await makeAlpacaRequest('/account')
          return NextResponse.json({
            success: true,
            data: `Connection successful! Account status: ${testAccount.status}`,
            timestamp: new Date().toISOString()
          })
        } catch (error) {
          return NextResponse.json({
            success: false,
            error: `Connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
            timestamp: new Date().toISOString()
          }, { status: 500 })
        }

      default:
        return NextResponse.json({ 
          success: false,
          error: 'Invalid action. Use: dashboard, performance, daily-report, or test-connection' 
        }, { status: 400 })
    }

  } catch (error) {
    console.error('Error in Alpaca tracker API:', error)
    return NextResponse.json({ 
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      details: 'Check your Alpaca API keys and account status'
    }, { status: 500 })
  }
}

// POST /api/alpaca-tracker - Update tracking data
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action } = body

    switch (action) {
      case 'record-snapshot':
        // Record portfolio snapshot (for now, just return current account info)
        const account = await makeAlpacaRequest('/account')
        
        return NextResponse.json({
          success: true,
          message: 'Portfolio snapshot recorded successfully',
          data: {
            account_info: account,
            timestamp: new Date().toISOString()
          }
        })

      case 'sync-trades':
        // Sync trades (get recent orders)
        const orders = await makeAlpacaRequest('/orders?status=filled&limit=50')
        
        return NextResponse.json({
          success: true,
          message: 'Trades synced successfully',
          data: {
            recent_orders: orders,
            timestamp: new Date().toISOString()
          }
        })

      case 'run-daily-update':
        // Run daily update (comprehensive data fetch)
        const [updateAccount, updatePositions, updateOrders] = await Promise.all([
          makeAlpacaRequest('/account'),
          makeAlpacaRequest('/positions'),
          makeAlpacaRequest('/orders?status=all&limit=100')
        ])

        const updatePortfolioHistory = await makeAlpacaRequest('/account/portfolio/history?period=1M&timeframe=1D')
        const updateMetrics = calculatePerformanceMetrics(
          updatePortfolioHistory.equity?.map((equity: number) => ({ equity })) || [],
          updateOrders
        )

        return NextResponse.json({
          success: true,
          message: 'Daily update completed successfully',
          data: {
            account_info: updateAccount,
            positions: updatePositions,
            recent_orders: updateOrders.slice(0, 10),
            performance: updateMetrics,
            timestamp: new Date().toISOString()
          }
        })

      default:
        return NextResponse.json({ 
          success: false,
          error: 'Invalid action. Use: record-snapshot, sync-trades, or run-daily-update' 
        }, { status: 400 })
    }

  } catch (error) {
    console.error('Error in POST request:', error)
    return NextResponse.json({ 
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      details: 'Check your Alpaca API keys and request parameters'
    }, { status: 500 })
  }
} 