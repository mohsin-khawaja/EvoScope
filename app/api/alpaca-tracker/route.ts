import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

// Helper function to run Python scripts
function runPythonScript(scriptPath: string, args: string[] = []): Promise<string> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath, ...args], {
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let stdout = ''
    let stderr = ''

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString()
    })

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(stdout)
      } else {
        reject(new Error(stderr || `Python script exited with code ${code}`))
      }
    })
  })
}

// GET /api/alpaca-tracker - Get tracking data
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action') || 'dashboard'

  try {
    let result

    switch (action) {
      case 'dashboard':
        // Get dashboard data
        result = await runPythonScript(
          path.join(process.cwd(), 'get_alpaca_dashboard.py')
        )
        break
        
      case 'performance':
        // Get performance metrics
        result = await runPythonScript(
          path.join(process.cwd(), 'get_alpaca_performance.py')
        )
        break
        
      case 'daily-report':
        // Generate daily report
        result = await runPythonScript(
          path.join(process.cwd(), 'daily_monitor.py')
        )
        break
        
      case 'test-connection':
        // Test Alpaca connection
        result = await runPythonScript(
          path.join(process.cwd(), 'test_alpaca_tracker.py')
        )
        break
        
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
    }

    // Try to parse as JSON, if it fails return as text
    try {
      const jsonResult = JSON.parse(result)
      return NextResponse.json(jsonResult)
    } catch {
      return NextResponse.json({ 
        success: true, 
        data: result.trim(),
        timestamp: new Date().toISOString() 
      })
    }

  } catch (error) {
    console.error('Error running Python script:', error)
    return NextResponse.json({ 
      error: 'Failed to execute tracker',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

// POST /api/alpaca-tracker - Update tracking data
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, data } = body

    let result

    switch (action) {
      case 'record-snapshot':
        // Record portfolio snapshot
        result = await runPythonScript(
          path.join(process.cwd(), 'record_snapshot.py')
        )
        break
        
      case 'sync-trades':
        // Sync trades
        result = await runPythonScript(
          path.join(process.cwd(), 'sync_trades.py')
        )
        break
        
      case 'run-daily-update':
        // Run daily update
        result = await runPythonScript(
          path.join(process.cwd(), 'daily_monitor.py')
        )
        break
        
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
    }

    return NextResponse.json({ 
      success: true, 
      data: result.trim(),
      timestamp: new Date().toISOString() 
    })

  } catch (error) {
    console.error('Error in POST request:', error)
    return NextResponse.json({ 
      error: 'Failed to update tracking data',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
} 