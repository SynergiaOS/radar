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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        try {
          chartInstance.current.destroy()
          chartInstance.current = null
        } catch (error) {
          console.warn('Error destroying chart on unmount:', error)
        }
      }
    }
  }, [])

  const renderFallbackChart = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    try {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Set canvas background
      ctx.fillStyle = '#1e1e1e'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw simple text fallback
      ctx.fillStyle = '#fff'
      ctx.font = '16px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('Chart data unavailable', canvas.width / 2, canvas.height / 2)

      // Draw ticker info if available
      if (ticker) {
        ctx.font = '20px Arial'
        ctx.fillText(ticker, canvas.width / 2, canvas.height / 2 - 30)
      }

      // Draw loading state
      ctx.font = '14px Arial'
      ctx.fillStyle = '#9ca3af'
      ctx.fillText('Professional charts require Chart.js', canvas.width / 2, canvas.height / 2 + 30)
    } catch (error) {
      console.error('Error rendering fallback chart:', error)
    }
  }

  const renderChart = (config: ChartConfig) => {
    if (!chartRef.current) return

    // Clear existing chart instance properly
    if (chartInstance.current) {
      try {
        chartInstance.current.destroy()
        chartInstance.current = null
      } catch (error) {
        console.warn('Error destroying chart instance:', error)
      }
    }

    // Clear canvas context
    const canvas = chartRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear the canvas completely
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Reset canvas dimensions to prevent "already in use" errors
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Import and use Chart.js dynamically with better error handling
    import('chart.js').then((ChartModule) => {
      const Chart = ChartModule.default || ChartModule.Chart || ChartModule
      if (!Chart) {
        console.error('Chart.js not available - falling back to simple chart')
        renderFallbackChart(ctx, canvas)
        return
      }

      // Register required scales
      try {
        if (Chart.register && Chart.CategoryScale) {
          Chart.register(Chart.CategoryScale)
        }
        if (Chart.register && Chart.LinearScale) {
          Chart.register(Chart.LinearScale)
        }
        if (Chart.register && Chart.TimeScale) {
          Chart.register(Chart.TimeScale)
        }
      } catch (scaleError) {
        console.warn('Error registering Chart.js scales:', scaleError)
      }

      try {
        // Helper: safely handle predefined callbacks without dynamic code execution
        const deserializeCallback = (cb: any) => {
          if (!cb) return cb
          if (typeof cb === 'object' && cb.type === 'function') {
            // Only allow predefined, safe callback functions
            const functionName = cb.name || 'unknown'
            switch (functionName) {
              case 'formatPrice':
                return (value: any) => {
                  if (typeof value === 'number') {
                    return value.toFixed(2)
                  }
                  return value
                }
              case 'formatPercentage':
                return (value: any) => {
                  if (typeof value === 'number') {
                    return (value * 100).toFixed(2) + '%'
                  }
                  return value
                }
              case 'formatDate':
                return (value: any) => {
                  if (value && typeof value === 'object' && value.toLocaleDateString) {
                    return value.toLocaleDateString()
                  }
                  return value
                }
              case 'formatVolume':
                return (value: any) => {
                  if (typeof value === 'number') {
                    if (value >= 1000000) {
                      return (value / 1000000).toFixed(1) + 'M'
                    } else if (value >= 1000) {
                      return (value / 1000).toFixed(1) + 'K'
                    }
                    return value.toString()
                  }
                  return value
                }
              default:
                console.warn('Unknown callback function:', functionName)
                // Return a safe default function instead of undefined
                return (value: any) => {
                  if (typeof value === 'number') {
                    return value.toString()
                  }
                  return value
                }
            }
          }
          return cb
        }

        // Handle different chart types
        let chartConfig = { ...config }

        if (config.type === 'candlestick') {
          // Convert candlestick to line chart for now (since we don't have the candlestick plugin)
          chartConfig = {
            ...config,
            type: 'line',
            data: {
              ...config.data,
              datasets: config.data.datasets.map((dataset: any) => ({
                ...dataset,
                // preserve styling props but ensure data points are numeric or {x,y}
                tension: dataset.tension ?? 0.1,
                fill: dataset.fill ?? false,
                pointRadius: dataset.pointRadius ?? 1,
                pointHoverRadius: dataset.pointHoverRadius ?? 4,
                data: (dataset.data || []).map((dp: any) => {
                  if (dp == null) return dp
                  if (typeof dp === 'object') {
                    const y = dp.c ?? dp.close ?? dp.y ?? dp.value
                    if (dp.x !== undefined) return { x: dp.x, y }
                    return y
                  }
                  return dp
                })
              }))
            }
          }
        }

        // Ensure all chart options have proper dark theme while preserving and deserializing
        // any existing nested options and callbacks
        const existingOptions = chartConfig.options || {}

        // Merge plugins preserving existing nested objects and callbacks
        const mergedPlugins = {
          ...(existingOptions.plugins || {}),
          legend: {
            ...(existingOptions.plugins?.legend || {}),
            display: existingOptions.plugins?.legend?.display ?? true,
            labels: {
              color: existingOptions.plugins?.legend?.labels?.color ?? '#fff',
              font: existingOptions.plugins?.legend?.labels?.font ?? { size: 11 },
              ...(existingOptions.plugins?.legend?.labels || {})
            }
          },
          tooltip: {
            mode: existingOptions.plugins?.tooltip?.mode ?? 'index',
            intersect: existingOptions.plugins?.tooltip?.intersect ?? false,
            backgroundColor: existingOptions.plugins?.tooltip?.backgroundColor ?? 'rgba(0, 0, 0, 0.8)',
            titleColor: existingOptions.plugins?.tooltip?.titleColor ?? '#fff',
            bodyColor: existingOptions.plugins?.tooltip?.bodyColor ?? '#fff',
            borderColor: existingOptions.plugins?.tooltip?.borderColor ?? '#374151',
            borderWidth: existingOptions.plugins?.tooltip?.borderWidth ?? 1,
            ...(existingOptions.plugins?.tooltip || {})
          }
        }

        // If there are callback objects, deserialize and preserve them
        if (existingOptions.plugins?.tooltip?.callbacks) {
          mergedPlugins.tooltip.callbacks = {
            ...(existingOptions.plugins?.tooltip?.callbacks || {}),
            ...Object.fromEntries(Object.entries(existingOptions.plugins.tooltip.callbacks).map(([k, v]) => [k, deserializeCallback(v)]))
          }
        }

        // Also handle legend labels callback if provided
        if (existingOptions.plugins?.legend?.labels?.generateLabels) {
          mergedPlugins.legend.labels.generateLabels = deserializeCallback(existingOptions.plugins.legend.labels.generateLabels)
        }

        chartConfig.options = {
          ...existingOptions,
          responsive: true,
          maintainAspectRatio: false,
          plugins: mergedPlugins,
          scales: {
            x: {
              ticks: {
                color: existingOptions.scales?.x?.ticks?.color ?? '#9ca3af',
                font: existingOptions.scales?.x?.ticks?.font ?? { size: 10 },
                ...(existingOptions.scales?.x?.ticks || {})
              },
              grid: {
                color: existingOptions.scales?.x?.grid?.color ?? '#374151',
                drawBorder: existingOptions.scales?.x?.grid?.drawBorder ?? false,
                ...(existingOptions.scales?.x?.grid || {})
              },
              ...(existingOptions.scales?.x || {})
            },
            y: {
              ticks: {
                color: existingOptions.scales?.y?.ticks?.color ?? '#9ca3af',
                font: existingOptions.scales?.y?.ticks?.font ?? { size: 10 },
                ...(existingOptions.scales?.y?.ticks || {})
              },
              grid: {
                color: existingOptions.scales?.y?.grid?.color ?? '#374151',
                drawBorder: existingOptions.scales?.y?.grid?.drawBorder ?? false,
                ...(existingOptions.scales?.y?.grid || {})
              },
              ...(existingOptions.scales?.y || {})
            }
          }
        }

        // Deserialize any other callbacks in options (e.g., scale callbacks)
        const walkAndDeserialize = (obj: any) => {
          if (!obj || typeof obj !== 'object') return
          Object.keys(obj).forEach(key => {
            const val = obj[key]
            if (val && typeof val === 'object' && val.type === 'function') {
              obj[key] = deserializeCallback(val)
            } else if (typeof val === 'object') {
              walkAndDeserialize(val)
            }
          })
        }

        walkAndDeserialize(chartConfig.options)

        // Create chart with better constructor detection
        if (Chart && typeof Chart === 'function') {
          try {
            // Test if Chart is constructible
            const testChart = new Chart(ctx, {
              type: 'line',
              data: { labels: [], datasets: [] },
              options: { responsive: false }
            })
            testChart.destroy() // Clean up test chart

            // Create actual chart
            chartInstance.current = new Chart(ctx, chartConfig)
          } catch (constructorError) {
            console.warn('Chart constructor failed, trying alternative Chart object:', constructorError)

            // Try alternative Chart object (for some Chart.js versions)
            if (Chart.Chart && typeof Chart.Chart === 'function') {
              chartInstance.current = new Chart.Chart(ctx, chartConfig)
            } else {
              throw new Error('Chart constructor not available')
            }
          }
        } else {
          console.error('Chart constructor not available')
          throw new Error('Chart object not a function')
        }
      } catch (error) {
        console.error('Error creating chart:', error)
        // Use the improved fallback chart
        renderFallbackChart(ctx, canvas)
      }
    }).catch(error => {
      console.error('Error loading Chart.js:', error)
      // Fallback when Chart.js fails to load
      if (chartRef.current) {
        const canvas = chartRef.current
        const ctx = canvas.getContext('2d')
        if (ctx) {
          renderFallbackChart(ctx, canvas)
        }
      }
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