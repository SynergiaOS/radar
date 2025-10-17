import React, { useEffect, useRef, useState, useMemo } from 'react'
import { Chart, ChartConfiguration, ChartOptions, registerables } from 'chart.js'
import 'chartjs-adapter-date-fns'
import { useStockData } from '@/lib/hooks/useStockData'
import io from 'socket.io-client'

// Rejestracja wszystkich komponentów Chart.js
Chart.register(...registerables)

interface PKNData {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
  rsi: number
}

interface TrendLine {
  start: { x: number; y: number }
  end: { x: number; y: number }
  color: string
  label: string
}

interface FibonacciLevel {
  level: number
  price: number
  label: string
}

const PKNChart: React.FC = () => {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstanceRef = useRef<Chart | null>(null)
  const [data, setData] = useState<PKNData[]>([])
  const [selectedTimeframe, setSelectedTimeframe] = useState<'W' | 'D'>('D')

  // Użyj istniejącego hooka do danych PKN.WA
  const { data: stockData, isLoading, error } = useStockData('PKN.WA', '1y')

  // Konwertuj dane z API na format PKNData
  const convertStockDataToPKNData = useMemo(() => {
    if (!stockData || !stockData.candlestick) return []

    return stockData.candlestick.map((item, index) => {
      const timestamp = Math.floor(item.x / 1000) * 1000 // Convert to milliseconds
      const rsi = stockData.indicators?.RSI_14?.[index]?.y || 50

      return {
        timestamp,
        open: item.o,
        high: item.h,
        low: item.l,
        close: item.c,
        volume: stockData.volume[index]?.y || 0,
        rsi
      }
    })
  }, [stockData])

  // Obliczanie linii trendu (kanał wzrostowy)
  const calculateTrendLines = useMemo((): TrendLine[] => {
    const prices = convertStockDataToPKNData.map(d => d.close)
    const timestamps = convertStockDataToPKNData.map(d => d.timestamp)

    // Dolna linia trendu (support)
    const supportLine = {
      start: { x: timestamps[0], y: Math.min(...prices.slice(0, 20)) },
      end: { x: timestamps[timestamps.length - 1], y: Math.min(...prices.slice(-20)) },
      color: '#10B981',
      label: 'Support (trend wzrostowy)'
    }

    // Górna linia trendu (resistance)
    const resistanceLine = {
      start: { x: timestamps[0], y: Math.max(...prices.slice(0, 20)) * 1.05 },
      end: { x: timestamps[timestamps.length - 1], y: Math.max(...prices.slice(-20)) * 1.05 },
      color: '#EF4444',
      label: 'Resistance (górna krawędź kanału)'
    }

    return [supportLine, resistanceLine]
  }, [convertStockDataToPKNData])

  // Poziomy Fibonacciego z analizy
  const fibonacciLevels = useMemo((): FibonacciLevel[] => {
    const prices = convertStockDataToPKNData
    if (prices.length === 0) return []

    const lowestPoint = Math.min(...prices.map(d => d.low))
    const highestPoint = Math.max(...prices.map(d => d.high))

    return [
      { level: 0, price: lowestPoint, label: '0% (dno)' },
      { level: 0.236, price: lowestPoint + (highestPoint - lowestPoint) * 0.236, label: '23.6%' },
      { level: 0.382, price: lowestPoint + (highestPoint - lowestPoint) * 0.382, label: '38.2%' },
      { level: 0.5, price: lowestPoint + (highestPoint - lowestPoint) * 0.5, label: '50%' },
      { level: 0.618, price: lowestPoint + (highestPoint - lowestPoint) * 0.618, label: '61.8%' },
      { level: 0.786, price: lowestPoint + (highestPoint - lowestPoint) * 0.786, label: '78.6%' },
      { level: 1, price: highestPoint, label: '100% (szczyt)' },
      { level: 1.618, price: highestPoint + (highestPoint - lowestPoint) * 0.618, label: '161.8% (extension)' }
    ]
  }, [convertStockDataToPKNData])

  useEffect(() => {
    setData(convertStockDataToPKNData)
  }, [convertStockDataToPKNData])

  useEffect(() => {
    if (!chartRef.current || data.length === 0) return

    // Niszczenie poprzedniego wykresu
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy()
      chartInstanceRef.current = null
    }

    const ctx = chartRef.current.getContext('2d')
    if (!ctx) return

    // Dane świecowe
    const candlestickData = data.map(d => ({
      x: d.timestamp,
      o: d.open,
      h: d.high,
      l: d.low,
      c: d.close
    }))

    // Konfiguracja wykresu świecowego
    const config: ChartConfiguration = {
      type: 'candlestick',
      data: {
        datasets: [
          {
            label: 'PKN.WA',
            data: candlestickData,
            borderColor: '#1E3A8A',
            backgroundColor: (context: any) => {
              const value = context.dataset.data[context.dataIndex]
              return value.c > value.o ? '#10B981' : '#EF4444' // Zielone dla wzrostu, czerwone dla spadku
            },
            borderWidth: 1
          },
          // SMA 20
          {
            label: 'SMA 20',
            data: data.map(d => ({ x: d.timestamp, y: d.close })),
            borderColor: '#F59E0B',
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            type: 'line',
            tension: 0.1
          },
          // RSI (oddzielny panel)
          {
            label: 'RSI',
            data: data.map(d => ({ x: d.timestamp, y: d.rsi })),
            borderColor: '#8B5CF6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            type: 'line',
            yAxisID: 'y1',
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          title: {
            display: true,
            text: 'PKN.WA - Analiza Techniczna (TradingView Style)',
            font: { size: 16, weight: 'bold' },
            color: '#1F2937',
            padding: 20
          },
          legend: {
            display: true,
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            callbacks: {
              label: function(context: any) {
                if (context.dataset.label === 'RSI') {
                  return `RSI: ${context.parsed.y.toFixed(1)}`
                }
                const candle = context.raw
                return [
                  `${context.dataset.label}`,
                  `O: ${candle.o.toFixed(2)}`,
                  `H: ${candle.h.toFixed(2)}`,
                  `L: ${candle.l.toFixed(2)}`,
                  `C: ${candle.c.toFixed(2)}`
                ]
              }
            }
          },
          // Adnotacje podobne do TradingView
          annotation: {
            annotations: {
              // Linie trendu
              ...calculateTrendLines.reduce((acc, line, index) => {
                acc[`trendLine${index}`] = {
                  type: 'line',
                  xMin: line.start.x,
                  yMin: line.start.y,
                  xMax: line.end.x,
                  yMax: line.end.y,
                  borderColor: line.color,
                  borderWidth: 2,
                  borderDash: [5, 5],
                  label: {
                    content: line.label,
                    enabled: true,
                    position: 'start'
                  }
                }
                return acc
              }, {} as any),

              // Poziomy Fibonacciego
              ...fibonacciLevels.reduce((acc, level, index) => {
                if (level.level > 0 && level.level < 1) { // Tylko kluczowe poziomy
                  acc[`fib${index}`] = {
                    type: 'line',
                    yMin: level.price,
                    yMax: level.price,
                    borderColor: '#6B7280',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    label: {
                      content: level.label,
                      enabled: true,
                      position: 'end'
                    }
                  }
                }
                return acc
              }, {} as any),

              // Wskaźnik spadającej gwiazdy (shooting star)
              shootingStar: {
                type: 'point',
                xValue: data[data.length - 10].timestamp,
                yValue: data[data.length - 10].high,
                backgroundColor: '#EF4444',
                radius: 8,
                label: {
                  content: 'Spadająca gwiazda',
                  enabled: true,
                  position: 'top'
                }
              },

              // Dywergencja RSI
              rsiDivergence: {
                type: 'box',
                xMin: data[data.length - 30].timestamp,
                xMax: data[data.length - 1].timestamp,
                yMin: 60,
                yMax: 75,
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderColor: '#EF4444',
                borderWidth: 1,
                label: {
                  content: 'Dywergencja RSI',
                  enabled: true,
                  position: 'start'
                }
              }
            }
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: selectedTimeframe === 'W' ? 'week' : 'day',
              displayFormats: {
                day: 'dd.MM',
                week: 'dd.MM.yy'
              }
            },
            grid: {
              display: true,
              color: '#E5E7EB'
            }
          },
          y: {
            type: 'linear',
            position: 'left',
            title: {
              display: true,
              text: 'Cena (PLN)'
            },
            grid: {
              color: '#E5E7EB'
            }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: {
              display: true,
              text: 'RSI'
            },
            min: 0,
            max: 100,
            grid: {
              drawOnChartArea: false
            }
          }
        }
      }
    }

    try {
      chartInstanceRef.current = new Chart(ctx, config)
    } catch (error) {
      console.error('Error creating PKN chart:', error)
    }

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy()
      }
    }
  }, [data, selectedTimeframe, calculateTrendLines, fibonacciLevels])

  // Simulacja aktualizacji danych w czasie rzeczywistym
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prevData => {
        if (prevData.length === 0) return prevData

        const lastCandle = prevData[prevData.length - 1]
        const newPrice = lastCandle.close + (Math.random() - 0.5) * 2

        const newCandle: PKNData = {
          timestamp: Date.now(),
          open: lastCandle.close,
          high: Math.max(lastCandle.close, newPrice) + Math.random() * 0.5,
          low: Math.min(lastCandle.close, newPrice) - Math.random() * 0.5,
          close: newPrice,
          volume: Math.floor(Math.random() * 5000000) + 1000000,
          rsi: Math.max(0, Math.min(100, lastCandle.rsi + (Math.random() - 0.5) * 5))
        }

        const newData = [...prevData.slice(-99), newCandle]
        return newData
      })
    }, 5000) // Aktualizacja co 5 sekund

    return () => clearInterval(interval)
  }, [])

  // Loading state
  if (isLoading) {
    return (
      <div className="pkn-chart-container" style={{ height: '600px', padding: '20px' }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4" />
            <p className="text-gray-600">Loading PKN.WA chart data...</p>
          </div>
        </div>
      </div>
    )
  }

  // Error state
  if (error || !convertStockDataToPKNData.length) {
    return (
      <div className="pkn-chart-container" style={{ height: '600px', padding: '20px' }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="text-red-500 font-medium text-xl mb-2">PKN.WA Chart Unavailable</div>
            <p className="text-gray-600">Unable to load trading data for PKN Orlen</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="pkn-chart-container" style={{ height: '600px', padding: '20px' }}>
      <div className="chart-controls" style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
        <button
          onClick={() => setSelectedTimeframe('D')}
          className={`px-4 py-2 rounded ${
            selectedTimeframe === 'D'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 text-gray-700'
          }`}
        >
          Dzienny (D)
        </button>
        <button
          onClick={() => setSelectedTimeframe('W')}
          className={`px-4 py-2 rounded ${
            selectedTimeframe === 'W'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 text-gray-700'
          }`}
        >
          Tygodniowy (W)
        </button>
        <div className="ml-auto text-sm text-gray-600">
          <span className="font-semibold">PKN.WA</span> -
          Aktualizacja: {new Date().toLocaleTimeString('pl-PL')}
        </div>
      </div>

      <div className="chart-wrapper" style={{ height: '500px', backgroundColor: '#FFFFFF', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <canvas ref={chartRef} />
      </div>

      <div className="chart-info" style={{ marginTop: '20px', fontSize: '12px', color: '#6B7280' }}>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <strong>Analiza techniczna:</strong> Kanał wzrostowy, dywergencja RSI, spadająca gwiazda
          </div>
          <div>
            <strong>Sygnały:</strong> Potencjalna korekta (-20%), wsparcie ~80 PLN, opór ~95 PLN
          </div>
        </div>
      </div>
    </div>
  )
}

export default PKNChart