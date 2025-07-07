import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'EvoScope RL-LSTM AI Trading Agent | Interactive Showcase',
  description: 'Advanced AI-powered trading system combining LSTM neural networks with Reinforcement Learning for intelligent market analysis and automated trading decisions.',
  keywords: ['AI Trading', 'LSTM', 'Reinforcement Learning', 'Machine Learning', 'Quantitative Trading', 'Financial AI'],
  authors: [{ name: 'AI Trading Team' }],
  openGraph: {
    title: 'EvoScope RL-LSTM AI Trading Agent',
    description: 'Advanced AI-powered trading system showcase',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-trading-bg text-trading-text min-h-screen`}>
        {children}
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1e293b',
              color: '#f8fafc',
              border: '1px solid #334155'
            }
          }}
        />
      </body>
    </html>
  )
} 