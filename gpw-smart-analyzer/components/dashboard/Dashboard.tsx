'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { useAnalysis, useRunAnalysis, useSystemStatus } from '@/lib/hooks/useAnalysis'
import { useStockData } from '@/lib/hooks/useStockData'
import { TradingChart } from '@/components/charts/TradingChart'
import { WatchlistManager } from '@/components/watchlist/WatchlistManager'
import { RefreshCw, TrendingUp, TrendingDown, Minus, Play, Pause, Settings, Search, Star } from 'lucide-react'
import Link from 'next/link'

export function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState('XTB.WA')
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState<'WIG30' | 'WIG20'>('WIG30')

  const { data: analysis, isLoading: analysisLoading } = useAnalysis(selectedIndex)
  const { data: status, isLoading: statusLoading } = useSystemStatus()
  const { mutate: runAnalysis, isPending: analysisRunning } = useRunAnalysis()
  const { data: selectedStock } = useStockData(selectedTicker, '1mo')

  // Use filtered stocks from backend
  const filteredStocks = analysis?.filtered_stocks || analysis?.all_stocks || []

  // Count signals from filtered stocks (current index)
  const buyCount = filteredStocks?.filter(s => s.decision === 'KUP').length || 0
  const sellCount = filteredStocks?.filter(s => s.decision === 'SPRZEDAJ').length || 0
  const holdCount = filteredStocks?.filter(s => s.decision === 'TRZYMAJ').length || 0

  // Get current index for display
  const currentIndex = selectedIndex

  const handleRunAnalysis = () => {
    runAnalysis()
  }

  const getRecommendationIcon = (decision: string) => {
    switch (decision) {
      case 'KUP':
        return <TrendingUp className="h-4 w-4" />
      case 'SPRZEDAJ':
        return <TrendingDown className="h-4 w-4" />
      default:
        return <Minus className="h-4 w-4" />
    }
  }

  const getRecommendationColor = (decision: string) => {
    switch (decision) {
      case 'KUP':
        return 'badge-buy'
      case 'SPRZEDAJ':
        return 'badge-sell'
      default:
        return 'badge-hold'
    }
  }

  return (
    <div className="flex h-screen bg-gray-900">
      {/* TradingView-style Sidebar */}
      <div className="tv-sidebar">
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-white font-bold text-lg">GPW Smart Analyzer</h1>
            <div className="text-xs text-gray-400">TradingView Style</div>
          </div>

          {/* Index Selector */}
          <div className="flex items-center space-x-2 mb-4">
            <Badge variant="outline" className="text-xs border-gray-600 text-gray-300">
              {currentIndex}
            </Badge>
            <Select value={currentIndex} onValueChange={(value: 'WIG30' | 'WIG20') => {
              setSelectedIndex(value)
            }}>
              <SelectTrigger className="w-24 tv-select">
                <SelectValue placeholder="Index" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="WIG30">WIG30</SelectItem>
                <SelectItem value="WIG20">WIG20</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search symbols..."
              className="w-full pl-10 pr-4 py-2 tv-input"
            />
          </div>
        </div>

        {/* Enhanced Watchlist */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Watchlist Manager */}
          <div className="flex-1 overflow-hidden">
            <div className="p-4 h-full">
              <WatchlistManager
                onStockSelect={setSelectedTicker}
                className="h-full"
              />
            </div>
          </div>

          {/* Market Overview */}
          <div className="p-4 border-t border-gray-700">
            <h3 className="text-white font-medium text-sm mb-3">Market Overview</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Buy Signals</span>
                <span className="tv-up font-medium">{buyCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Sell Signals</span>
                <span className="tv-down font-medium">{sellCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Hold Signals</span>
                <span className="tv-neutral font-medium">{holdCount}</span>
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="p-4 border-t border-gray-700 space-y-2">
            <Button
              onClick={handleRunAnalysis}
              disabled={analysisRunning}
              className="w-full tv-button-primary"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${analysisRunning ? 'animate-spin' : ''}`} />
              {analysisRunning ? 'Running...' : 'Run Analysis'}
            </Button>

            <Button
              variant={isMonitoring ? 'secondary' : 'outline'}
              onClick={() => setIsMonitoring(!isMonitoring)}
              className="w-full tv-button-secondary"
            >
              {isMonitoring ? (
                <Pause className="h-4 w-4 mr-2" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              {isMonitoring ? 'Stop' : 'Start'} Monitoring
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* TradingChart */}
        <div className="flex-1">
          <TradingChart
            ticker={selectedTicker}
            period="1y"
            className="h-full"
          />
        </div>

        {/* Bottom Info Bar */}
        <div className="bg-gray-800 border-t border-gray-700 px-4 py-2">
          <div className="flex items-center justify-between text-xs text-gray-400">
            <div className="flex items-center space-x-4">
              <span>System: {status?.last_update ? 'Online' : 'Offline'}</span>
              <span>Last Update: {status?.last_update ?
                new Date(status.last_update).toLocaleTimeString() :
                'No data'
              }</span>
              <span>Analysis: {analysis?.count || 0} stocks</span>
            </div>
            <div className="flex items-center space-x-4">
              <span>ROE Threshold: {analysis?.thresholds.roe}%</span>
              <span>P/E Threshold: {analysis?.thresholds.pe}</span>
              <span>Index: {currentIndex}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}