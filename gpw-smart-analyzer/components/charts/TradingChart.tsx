'use client'

import { useEffect, useRef, useState } from 'react'
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TrendingUp, TrendingDown, BarChart3, Activity, Settings } from 'lucide-react'
import { useStockData } from '@/lib/hooks/useStockData'
import { ProfessionalChart } from './ProfessionalChart'

interface TradingChartProps {
  ticker: string
  period?: string
  className?: string
  onReady?: ({ chart, candlestickSeries }: { chart: IChartApi; candlestickSeries: ISeriesApi<'Candlestick'> }) => void
}

export function TradingChart({ ticker, period = '1y', className, onReady }: TradingChartProps) {
  const [chartMode, setChartMode] = useState<'lightweight' | 'professional'>('professional')
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  const { data: stockData, isLoading, error } = useStockData(ticker, period)

  useEffect(() => {
    if (!chartContainerRef.current || !stockData) return

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1e1e1e' },
        textColor: '#d1d4dc',
        fontSize: 12,
        fontFamily: 'Inter',
      },
      grid: {
        vertLines: { color: '#2b2b43' },
        horzLines: { color: '#2b2b43' },
      },
      crosshair: {
        mode: 0,
        vertLine: {
          width: 1,
          color: '#758696',
          style: 3,
        },
        horzLine: {
          width: 1,
          color: '#758696',
          style: 3,
        },
      },
      rightPriceScale: {
        borderColor: '#2b2b43',
        textColor: '#d1d4dc',
      },
      timeScale: {
        borderColor: '#2b2b43',
        textColor: '#d1d4dc',
        timeVisible: true,
        secondsVisible: false,
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    })

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a80',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // Overlay on main pane
    })

    // Apply scale margins after creation
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.7,
        bottom: 0,
      },
    })

    // Add SMA indicators if available
    if (stockData.indicators?.SMA_20) {
      const sma20Series = chart.addLineSeries({
        color: '#2962FF',
        lineWidth: 2,
        title: 'SMA 20',
      })

      const smaData = stockData.indicators.SMA_20
        .filter(item => item.y !== null)
        .map(item => ({
          time: Math.floor(item.x / 1000), // Convert milliseconds to seconds
          value: item.y,
        }))

      sma20Series.setData(smaData)
    }

    if (stockData.indicators?.SMA_50) {
      const sma50Series = chart.addLineSeries({
        color: '#FF6D00',
        lineWidth: 2,
        title: 'SMA 50',
      })

      const smaData = stockData.indicators.SMA_50
        .filter(item => item.y !== null)
        .map(item => ({
          time: Math.floor(item.x / 1000), // Convert milliseconds to seconds
          value: item.y,
        }))

      sma50Series.setData(smaData)
    }

    // Format and set candlestick data
    const candlestickData = stockData.candlestick.map(item => ({
      time: Math.floor(item.x / 1000), // Convert milliseconds to seconds
      open: item.o,
      high: item.h,
      low: item.l,
      close: item.c,
    }))

    const volumeData = stockData.volume.map((item, index) => {
      const candleData = stockData.candlestick[index]
      return {
        time: Math.floor(item.x / 1000), // Convert milliseconds to seconds
        value: item.y,
        color: candleData.c >= candleData.o ? '#26a69a80' : '#ef535080',
      }
    })

    candlestickSeries.setData(candlestickData)
    volumeSeries.setData(volumeData)

    // Fit content to show all data
    chart.timeScale().fitContent()

    // Store refs
    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries

    // Notify parent component that chart is ready
    if (onReady) {
      onReady({ chart, candlestickSeries })
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [stockData, ticker, period])

  if (isLoading) {
    return (
      <Card className={className}>
        <div className="p-6">
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4" />
              <p className="text-textSecondary">Loading chart data...</p>
            </div>
          </div>
        </div>
      </Card>
    )
  }

  if (error || !stockData) {
    return (
      <Card className={className}>
        <div className="p-6">
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <TrendingDown className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <p className="text-red-500 font-medium">Failed to load chart data</p>
              <p className="text-textSecondary text-sm mt-1">
                Please try again later
              </p>
            </div>
          </div>
        </div>
      </Card>
    )
  }

  if (chartMode === 'professional') {
    return (
      <div className={`tv-chart-container ${className}`}>
        {/* TradingView-style toolbar with mode switcher */}
        <div className="tv-toolbar">
          <div className="flex items-center space-x-2">
            <span className="text-white font-medium">{ticker}</span>
            <span className="text-gray-400 text-sm">{stockData.info?.name}</span>
            <Badge variant="outline" className="text-xs border-green-500 text-green-400">
              Professional Charts
            </Badge>
          </div>

          <div className="flex items-center space-x-2 ml-auto">
            <Tabs value={chartMode} onValueChange={(value: any) => setChartMode(value)} className="w-auto">
              <TabsList className="bg-gray-800 border border-gray-700 h-8">
                <TabsTrigger
                  value="professional"
                  className="flex items-center space-x-1 h-6 px-3 text-xs data-[state=active]:bg-blue-600 data-[state=active]:text-white"
                >
                  <BarChart3 className="h-3 w-3" />
                  <span className="hidden sm:inline">Pro</span>
                </TabsTrigger>
                <TabsTrigger
                  value="lightweight"
                  className="flex items-center space-x-1 h-6 px-3 text-xs data-[state=active]:bg-blue-600 data-[state=active]:text-white"
                >
                  <Activity className="h-3 w-3" />
                  <span className="hidden sm:inline">Basic</span>
                </TabsTrigger>
              </TabsList>
            </Tabs>

            <div className="border-l border-gray-600 h-4 mx-2"></div>

            <button className="tv-toolbar-button">1D</button>
            <button className="tv-toolbar-button">1W</button>
            <button className="tv-toolbar-button">1M</button>
            <button className="tv-toolbar-button bg-blue-600 text-white">1Y</button>
            <button className="tv-toolbar-button">ALL</button>
          </div>
        </div>

        {/* Professional Chart Component */}
        <ProfessionalChart
          ticker={ticker}
          companyName={stockData.info?.name}
          className="flex-1"
        />
      </div>
    )
  }

  return (
    <div className={`tv-chart-container ${className}`}>
      {/* TradingView-style toolbar */}
      <div className="tv-toolbar">
        <div className="flex items-center space-x-2">
          <span className="text-white font-medium">{ticker}</span>
          <span className="text-gray-400 text-sm">{stockData.info?.name}</span>
          <Badge variant="outline" className="text-xs border-gray-500 text-gray-400">
            Basic Charts
          </Badge>
        </div>

        <div className="flex items-center space-x-2 ml-auto">
          <Tabs value={chartMode} onValueChange={(value: any) => setChartMode(value)} className="w-auto">
            <TabsList className="bg-gray-800 border border-gray-700 h-8">
              <TabsTrigger
                value="professional"
                className="flex items-center space-x-1 h-6 px-3 text-xs data-[state=active]:bg-blue-600 data-[state=active]:text-white"
              >
                <BarChart3 className="h-3 w-3" />
                <span className="hidden sm:inline">Pro</span>
              </TabsTrigger>
              <TabsTrigger
                value="lightweight"
                className="flex items-center space-x-1 h-6 px-3 text-xs data-[state=active]:bg-blue-600 data-[state=active]:text-white"
              >
                <Activity className="h-3 w-3" />
                <span className="hidden sm:inline">Basic</span>
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="border-l border-gray-600 h-4 mx-2"></div>

          <button className="tv-toolbar-button">1D</button>
          <button className="tv-toolbar-button">1W</button>
          <button className="tv-toolbar-button">1M</button>
          <button className="tv-toolbar-button bg-blue-600 text-white">1Y</button>
          <button className="tv-toolbar-button">ALL</button>

          <div className="border-l border-gray-600 h-4 mx-2"></div>

          <button className="tv-toolbar-button">Indicators</button>
          <button className="tv-toolbar-button">Settings</button>
        </div>
      </div>

      {/* Price header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <div className="flex items-center space-x-4">
          <div>
            <div className={`text-2xl font-bold ${
              stockData.info?.change >= 0 ? 'tv-up' : 'tv-down'
            }`}>
              {stockData.info?.current_price?.toFixed(2)} PLN
            </div>
            <div className={`text-sm font-medium flex items-center ${
              stockData.info?.change >= 0 ? 'tv-up' : 'tv-down'
            }`}>
              {stockData.info?.change >= 0 ? (
                <TrendingUp className="h-3 w-3 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1" />
              )}
              {stockData.info?.change >= 0 ? '+' : ''}
              {stockData.info?.change?.toFixed(2)} ({stockData.info?.change_percent?.toFixed(2)}%)
            </div>
          </div>

          <div className="flex items-center space-x-4 text-sm">
            <div className="text-gray-400">
              <div>O: {stockData.candlestick[stockData.candlestick.length - 1]?.o?.toFixed(2)}</div>
              <div>H: {stockData.candlestick[stockData.candlestick.length - 1]?.h?.toFixed(2)}</div>
            </div>
            <div className="text-gray-400">
              <div>L: {stockData.candlestick[stockData.candlestick.length - 1]?.l?.toFixed(2)}</div>
              <div>C: {stockData.candlestick[stockData.candlestick.length - 1]?.c?.toFixed(2)}</div>
            </div>
            <div className="text-gray-400">
              <div>Vol: {(stockData.volume[stockData.volume.length - 1]?.y / 1000000)?.toFixed(1)}M</div>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-xs text-gray-400">SMA20</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-orange-500 rounded"></div>
            <span className="text-xs text-gray-400">SMA50</span>
          </div>
          {stockData.indicators?.RSI_14 && (
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-purple-500 rounded"></div>
              <span className="text-xs text-gray-400">RSI14</span>
            </div>
          )}
        </div>
      </div>

      {/* Chart container */}
      <div ref={chartContainerRef} className="w-full" style={{ height: '500px' }} />

      {/* Bottom status bar */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-gray-700 text-xs text-gray-400">
        <div>
          Period: {period.toUpperCase()} | Data points: {stockData.candlestick.length}
        </div>
        <div>
          Last update: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}