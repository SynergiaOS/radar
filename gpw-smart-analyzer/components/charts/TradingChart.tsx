'use client'

import { useEffect, useRef } from 'react'
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { useStockData } from '@/lib/hooks/useStockData'

interface TradingChartProps {
  ticker: string
  period?: string
  className?: string
}

export function TradingChart({ ticker, period = '1y', className }: TradingChartProps) {
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
      priceScaleId: 'volume',
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
        .filter(item => item.value !== null)
        .map(item => ({
          time: item.x,
          value: item.value,
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
        .filter(item => item.value !== null)
        .map(item => ({
          time: item.x,
          value: item.value,
        }))

      sma50Series.setData(smaData)
    }

    // Format and set candlestick data
    const candlestickData = stockData.candlestick.map(item => ({
      time: item.x,
      open: item.o,
      high: item.h,
      low: item.l,
      close: item.c,
    }))

    const volumeData = stockData.volume.map((item, index) => {
      const candleData = stockData.candlestick[index]
      return {
        time: item.x,
        value: item.y,
        color: candleData.c >= candleData.o ? '#26a69a80' : '#ef535080',
      }
    })

    candlestickSeries.setData(candlestickData)
    volumeSeries.setData(volumeData)

    // Store refs
    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries

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

  return (
    <Card className={className}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <h3 className="text-lg font-semibold text-textPrimary">
              {stockData.info?.name || ticker}
            </h3>
            <Badge variant="outline">{ticker}</Badge>
          </div>

          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-xl font-bold text-textPrimary">
                {stockData.info?.current_price?.toFixed(2)} PLN
              </div>
              <div className={`text-sm font-medium flex items-center ${
                stockData.info?.change >= 0 ? 'text-up' : 'text-down'
              }`}>
                {stockData.info?.change >= 0 ? (
                  <TrendingUp className="h-3 w-3 mr-1" />
                ) : (
                  <TrendingDown className="h-3 w-3 mr-1" />
                )}
                {stockData.info?.change >= 0 ? '+' : ''}
                {stockData.info?.change_percent?.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {/* Chart container */}
          <div ref={chartContainerRef} className="w-full" />

          {/* Chart info */}
          <div className="flex items-center justify-between text-sm text-textSecondary">
            <div>
              Period: {period} | Data points: {stockData.candlestick.length}
            </div>
            <div>
              Last update: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}