'use client'

import { useEffect, useRef, useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs'
import {
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Activity,
  Zap
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface ProfessionalChartProps {
  ticker: string
  companyName?: string
  className?: string
}

interface ChartConfig {
  type: string
  data: any
  options: any
}

interface SystemConfig {
  site_name: string
  domain: string
  version: string
  features: {
    professional_charts: boolean
    watchlist: boolean
    realtime_data: boolean
    technical_indicators: string[]
    supported_indices: string[]
  }
  theme: {
    mode: string
    primary_color: string
    style: string
  }
}

const CHART_TYPES = [
  { id: 'candlestick', name: 'Świeczki', icon: BarChart3, description: 'OHLC + MA + Bollinger' },
  { id: 'rsi', name: 'RSI', icon: Activity, description: 'Relative Strength Index' },
  { id: 'macd', name: 'MACD', icon: Zap, description: 'MACD + Histogram' },
  { id: 'volume', name: 'Wolumen', icon: TrendingUp, description: 'Volume Analysis' }
]

export function ProfessionalChart({ ticker, companyName, className }: ProfessionalChartProps) {
  const [activeChart, setActiveChart] = useState('candlestick')
  const [chartData, setChartData] = useState<ChartConfig | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<any>(null)

  // Load system configuration
  useEffect(() => {
    fetch('http://localhost:5000/api/system/config')
      .then(res => res.json())
      .then(setSystemConfig)
      .catch(console.error)
  }, [])

  // Load chart data
  useEffect(() => {
    if (!ticker) return

    setLoading(true)
    setError(null)

    fetch(`http://localhost:5000/api/charts/${ticker}/${activeChart}`)
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`)
        }
        return res.json()
      })
      .then(data => {
        setChartData(data)
        renderChart(data)
      })
      .catch(err => {
        console.error('Error fetching chart data:', err)
        setError(err.message || 'Failed to load chart data')
      })
      .finally(() => {
        setLoading(false)
      })
  }, [ticker, activeChart])

  const renderChart = (config: ChartConfig) => {
    if (!chartRef.current) return

    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    // Import and use Chart.js dynamically
    import('chart.js').then(({ default: Chart }) => {
      const ctx = chartRef.current.getContext('2d')
      if (!ctx) return

      // Register additional plugins if needed
      if (config.type === 'candlestick' && !Chart.registry.getPlugin('candlestick')) {
        // You would need to register the candlestick plugin here
        // For now, we'll fallback to line chart
        config.type = 'line'
      }

      chartInstance.current = new Chart(ctx, config)
    })
  }

  const handleChartTypeChange = (chartType: string) => {
    setActiveChart(chartType)
  }

  const getChangeColor = (change: number) => {
    if (change > 0) return 'tv-up'
    if (change < 0) return 'tv-down'
    return 'tv-neutral'
  }

  const getChartIcon = (type: string) => {
    const chartType = CHART_TYPES.find(ct => ct.id === type)
    return chartType ? chartType.icon : BarChart3
  }

  if (error) {
    return (
      <Card className={cn("tv-panel", className)}>
        <div className="p-6">
          <div className="text-red-400 text-center">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="mb-4">Error loading chart: {error}</p>
            <Button onClick={() => window.location.reload()} variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </div>
      </Card>
    )
  }

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="tv-panel-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div>
              <h3 className="text-white font-bold text-lg">{ticker}</h3>
              {companyName && (
                <p className="text-gray-400 text-sm">{companyName}</p>
              )}
            </div>
            {systemConfig && (
              <Badge variant="secondary" className="text-xs">
                {systemConfig.version}
              </Badge>
            )}
          </div>

          <div className="flex items-center space-x-2">
            <Button
              onClick={() => window.location.reload()}
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-gray-400 hover:text-white"
            >
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
            </Button>

            {systemConfig?.features.professional_charts && (
              <Badge variant="outline" className="text-xs border-green-500 text-green-400">
                Professional
              </Badge>
            )}
          </div>
        </div>
      </div>

      {/* Chart Type Selector */}
      <div className="p-4 border-b border-gray-700">
        <Tabs value={activeChart} onValueChange={handleChartTypeChange} className="w-full">
          <TabsList className="grid grid-cols-4 w-full bg-gray-800 border border-gray-700">
            {CHART_TYPES.map((chartType) => {
              const Icon = chartType.icon
              return (
                <TabsTrigger
                  key={chartType.id}
                  value={chartType.id}
                  className="flex items-center space-x-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-gray-400 hover:text-white"
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{chartType.name}</span>
                </TabsTrigger>
              )
            })}
          </TabsList>

          {CHART_TYPES.map((chartType) => (
            <TabsContent key={chartType.id} value={chartType.id} className="mt-2">
              <div className="text-xs text-gray-400 text-center">
                {chartType.description}
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </div>

      {/* Chart Container */}
      <div className="flex-1 relative overflow-hidden">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10">
            <div className="text-center">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-2" />
              <p className="text-gray-400 text-sm">Loading {activeChart} chart...</p>
            </div>
          </div>
        )}

        <div className="relative h-full">
          <canvas
            ref={chartRef}
            className="w-full h-full"
            style={{ backgroundColor: '#1e1e1e' }}
          />
        </div>
      </div>

      {/* Footer */}
      <div className="tv-panel-footer">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-4 text-gray-400">
            <span>Chart: {activeChart}</span>
            <span>Domain: {systemConfig?.domain || 'radar-wig.pl'}</span>
          </div>

          {systemConfig?.features.realtime_data && (
            <div className="flex items-center space-x-1 text-green-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span>Live</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}