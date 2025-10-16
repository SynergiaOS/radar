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
import { RefreshCw, TrendingUp, TrendingDown, Minus, Play, Pause, Settings } from 'lucide-react'
import Link from 'next/link'

export function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState('XTB.WA')
  const [isMonitoring, setIsMonitoring] = useState(false)

  const { data: analysis, isLoading: analysisLoading } = useAnalysis()
  const { data: status, isLoading: statusLoading } = useSystemStatus()
  const { mutate: runAnalysis, isPending: analysisRunning } = useRunAnalysis()
  const { data: selectedStock } = useStockData(selectedTicker, '1mo')

  const buyCount = analysis?.all_stocks?.filter(s => s.decision === 'KUP').length || 0
  const sellCount = analysis?.all_stocks?.filter(s => s.decision === 'SPRZEDAJ').length || 0
  const holdCount = analysis?.all_stocks?.filter(s => s.decision === 'TRZYMAJ').length || 0

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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-textPrimary mb-2">
            Investment Dashboard
          </h1>
          <p className="text-textSecondary">
            Real-time analysis and recommendations for GPW stocks
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <Select value={selectedTicker} onValueChange={setSelectedTicker}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {analysis?.all_stocks?.map((stock) => (
                <SelectItem key={stock.ticker} value={stock.ticker}>
                  {stock.ticker} - {stock.name} ({stock.decision})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            onClick={handleRunAnalysis}
            disabled={analysisRunning}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${analysisRunning ? 'animate-spin' : ''}`} />
            <span>{analysisRunning ? 'Running...' : 'Run Analysis'}</span>
          </Button>

          <Button
            variant={isMonitoring ? 'secondary' : 'outline'}
            onClick={() => setIsMonitoring(!isMonitoring)}
            className="flex items-center space-x-2"
          >
            {isMonitoring ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            <span>{isMonitoring ? 'Stop' : 'Start'} Monitoring</span>
          </Button>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-textSecondary mb-1">Buy Signals</p>
              <p className="text-3xl font-bold text-actions-buy">{buyCount}</p>
            </div>
            <div className="h-12 w-12 rounded-full bg-actions-buy/20 flex items-center justify-center">
              <TrendingUp className="h-6 w-6 text-actions-buy" />
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-textSecondary mb-1">Sell Signals</p>
              <p className="text-3xl font-bold text-actions-sell">{sellCount}</p>
            </div>
            <div className="h-12 w-12 rounded-full bg-actions-sell/20 flex items-center justify-center">
              <TrendingDown className="h-6 w-6 text-actions-sell" />
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-textSecondary mb-1">Hold Signals</p>
              <p className="text-3xl font-bold text-actions-hold">{holdCount}</p>
            </div>
            <div className="h-12 w-12 rounded-full bg-actions-hold/20 flex items-center justify-center">
              <Minus className="h-6 w-6 text-actions-hold" />
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-textSecondary mb-1">System Status</p>
              <p className="text-lg font-semibold">
                {statusLoading ? 'Loading...' : status?.last_update ? 'Online' : 'Offline'}
              </p>
              <p className="text-xs text-textSecondary">
                {status?.last_update ?
                  `Updated: ${new Date(status.last_update).toLocaleTimeString()}` :
                  'No recent updates'
                }
              </p>
            </div>
            <div className={`h-12 w-12 rounded-full flex items-center justify-center ${
              status?.last_update ? 'bg-green-500/20' : 'bg-red-500/20'
            }`}>
              <div className={`h-4 w-4 rounded-full ${
                status?.last_update ? 'bg-green-500' : 'bg-red-500'
              }`} />
            </div>
          </div>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Featured Chart */}
        <div className="lg:col-span-2">
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-textPrimary">
                {selectedStock?.info?.name || selectedTicker}
              </h2>
              <Badge variant="outline">{selectedTicker}</Badge>
            </div>

            <div className="space-y-4">
              {/* Chart placeholder - would integrate TradingChart component here */}
              <div className="h-96 bg-surfaceLight rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <TrendingUp className="h-12 w-12 text-textSecondary mx-auto mb-4" />
                  <p className="text-textSecondary">Interactive Chart</p>
                  <p className="text-sm text-textSecondary mt-2">
                    Click to view detailed analysis
                  </p>
                </div>
              </div>

              {/* Quick Stats */}
              {selectedStock?.info && (
                <div className="grid grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-sm text-textSecondary">Price</p>
                    <p className="font-semibold">
                      {selectedStock.info.current_price.toFixed(2)} PLN
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-textSecondary">Change</p>
                    <p className={`font-semibold ${
                      selectedStock.info.change >= 0 ? 'text-up' : 'text-down'
                    }`}>
                      {selectedStock.info.change >= 0 ? '+' : ''}
                      {selectedStock.info.change_percent.toFixed(2)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-textSecondary">Volume</p>
                    <p className="font-semibold">
                      {selectedStock?.data?.[0]?.volume ?
                        (selectedStock.data[0].volume / 1000).toFixed(0) + 'K' :
                        'N/A'
                      }
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-textSecondary">Status</p>
                    <Badge
                      variant="outline"
                      className={
                        selectedStock.info.change >= 0 ? 'border-up text-up' : 'border-down text-down'
                      }
                    >
                      {selectedStock.info.change >= 0 ? 'Bullish' : 'Bearish'}
                    </Badge>
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* Top Signals List */}
        <div className="space-y-4">
          <Card className="p-6">
            <h2 className="text-xl font-semibold text-textPrimary mb-4">
              Top Recommendations
            </h2>

            <div className="space-y-3">
              {analysis?.all_stocks
                ?.filter(stock => stock.decision === 'KUP')
                .slice(0, 5)
                .map((stock) => (
                  <Link
                    key={stock.ticker}
                    href={`/chart/${stock.ticker}`}
                    className="block p-3 rounded-lg bg-surfaceLight hover:bg-surface transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{stock.ticker}</div>
                        <div className="text-sm text-textSecondary">
                          {stock.name}
                        </div>
                        <div className="text-xs text-textSecondary mt-1">
                          ROE: {stock.roe?.toFixed(1)}% | P/E: {stock.pe_ratio?.toFixed(1)}
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge className={getRecommendationColor(stock.decision)}>
                          {getRecommendationIcon(stock.decision)}
                          <span className="ml-1">{stock.decision}</span>
                        </Badge>
                        <div className="text-sm font-medium mt-1">
                          {stock.current_price?.toFixed(2)} PLN
                        </div>
                      </div>
                    </div>
                  </Link>
                )) || (
                <p className="text-textSecondary text-center py-4">
                  No buy recommendations available
                </p>
              )}
            </div>

            {analysis?.all_stocks && (
              <div className="mt-4 pt-4 border-t border-border">
                <Link
                  href="/signals"
                  className="block w-full text-center text-sm text-blue-600 hover:text-blue-500"
                >
                  View all signals →
                </Link>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Recent Activity */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-textPrimary mb-4">
          Recent Analysis Activity
        </h2>

        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-lg bg-surfaceLight">
            <div className="flex items-center space-x-3">
              <RefreshCw className="h-4 w-4 text-blue-500" />
              <div>
                <p className="font-medium">Analysis completed</p>
                <p className="text-sm text-textSecondary">
                  Processed {analysis?.count || 0} stocks
                </p>
              </div>
            </div>
            <span className="text-sm text-textSecondary">
              {analysis?.timestamp ?
                new Date(analysis.timestamp).toLocaleString() :
                'No recent activity'
              }
            </span>
          </div>

          {isMonitoring && (
            <div className="flex items-center justify-between p-3 rounded-lg bg-surfaceLight">
              <div className="flex items-center space-x-3">
                <Play className="h-4 w-4 text-green-500" />
                <div>
                  <p className="font-medium">Real-time monitoring active</p>
                  <p className="text-sm text-textSecondary">
                    Tracking price movements
                  </p>
                </div>
              </div>
              <Badge variant="outline" className="text-green-500 border-green-500">
                Live
              </Badge>
            </div>
          )}

          <div className="flex items-center justify-between p-3 rounded-lg bg-surfaceLight">
            <div className="flex items-center space-x-3">
              <Settings className="h-4 w-4 text-gray-500" />
              <div>
                <p className="font-medium">System configuration</p>
                <p className="text-sm text-textSecondary">
                  ROE: {analysis?.thresholds.roe}% | P/E: {analysis?.thresholds.pe}
                </p>
              </div>
            </div>
            <span className="text-sm text-textSecondary">
              Active
            </span>
          </div>
        </div>
      </Card>
    </div>
  )
}