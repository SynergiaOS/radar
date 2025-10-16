import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Providers } from './providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'GPW Smart Analyzer - Trading Chart Application',
  description: 'Advanced trading analysis for GPW stocks with ML/RL signals and professional charts',
  keywords: ['GPW', 'trading', 'charts', 'technical analysis', 'stocks', 'WIG30', 'WIG20'],
  authors: [{ name: 'GPW Smart Analyzer Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#131722',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="pl" className="dark">
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}