'use client'

import React, { useState, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import {
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Activity,
  Wifi,
  WifiOff
} from 'lucide-react'
import { useRealtimeTable } from '@/lib/hooks/useRealtimeTable'
import { formatVolume, formatMarketCap, formatPrice, formatPercent } from '@/lib/utils/formatters'
import { formatTimestamp } from '@/lib/utils'
import { cn } from '@/lib/utils'
import { RealtimeTableRow } from '@/types/stock'

interface RealtimeTableProps {
  className?: string
  onRowClick?: (ticker: string) => void
}

export function RealtimeTable({ className, onRowClick }: RealtimeTableProps) {
  const [sortBy, setSortBy] = useState<keyof RealtimeTableRow>('ticker')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')

  const {
    data,
    isLoading,
    error,
    isConnected,
    lastUpdate,
    refetch
  } = useRealtimeTable()

  // Sort stocks
  const sortedStocks = useMemo(() => {
    if (!data?.stocks) return []

    return [...data.stocks].sort((a, b) => {
      const aValue = a[sortBy]
      const bValue = b[sortBy]

      if (aValue === null && bValue === null) return 0
      if (aValue === null) return 1
      if (bValue === null) return -1

      let comparison = 0
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        comparison = aValue.localeCompare(bValue)
      } else if (typeof aValue === 'number' && typeof bValue === 'number') {
        comparison = aValue - bValue
      }

      return sortOrder === 'asc' ? comparison : -comparison
    })
  }, [data?.stocks, sortBy, sortOrder])

  const handleSort = (column: keyof RealtimeTableRow) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(column)
      setSortOrder('asc')
    }
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'upward':
        return <TrendingUp className="h-4 w-4 text-green-500" />
      case 'downward':
        return <TrendingDown className="h-4 w-4 text-red-500" />
      case 'sideways':
        return <Minus className="h-4 w-4 text-gray-500" />
      default:
        return <Minus className="h-4 w-4 text-gray-400" />
    }
  }

  const getSectorColor = (sector: string | null) => {
    if (!sector) return 'bg-gray-100 text-gray-800'

    const sectorColors: Record<string, string> = {
      'Banking': 'bg-blue-100 text-blue-800',
      'Energy': 'bg-green-100 text-green-800',
      'Telecom': 'bg-purple-100 text-purple-800',
      'Insurance': 'bg-orange-100 text-orange-800',
      'Retail': 'bg-yellow-100 text-yellow-800',
      'Technology': 'bg-cyan-100 text-cyan-800',
      'Finance': 'bg-pink-100 text-pink-800',
      'Industrial': 'bg-gray-100 text-gray-800',
      'Other': 'bg-gray-100 text-gray-800'
    }

    return sectorColors[sector] || sectorColors['Other']
  }

  const getRSIColor = (rsi: number | null) => {
    if (rsi === null) return 'text-gray-500'
    if (rsi > 70) return 'text-red-500'
    if (rsi < 30) return 'text-green-500'
    return 'text-gray-500'
  }

  const getRSITooltip = (rsi: number | null) => {
    if (rsi === null) return 'RSI: N/A'
    if (rsi > 70) return `RSI: ${rsi.toFixed(1)} (Overbought)`
    if (rsi < 30) return `RSI: ${rsi.toFixed(1)} (Oversold)`
    return `RSI: ${rsi.toFixed(1)} (Neutral)`
  }

  if (isLoading) {
    return (
      <div className={cn("w-full p-4", className)}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Market Watch</h2>
          <Skeleton className="h-8 w-24" />
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Last</TableHead>
              <TableHead>Change</TableHead>
              <TableHead>Change %</TableHead>
              <TableHead>Volume</TableHead>
              <TableHead>Market Cap</TableHead>
              <TableHead>Sector</TableHead>
              <TableHead>RSI</TableHead>
              <TableHead>Trend</TableHead>
              <TableHead>Analysis</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[1, 2, 3].map((i) => (
              <TableRow key={i}>
                <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                <TableCell><Skeleton className="h-4 w-12" /></TableCell>
                <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                <TableCell><Skeleton className="h-4 w-32" /></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    )
  }

  if (error) {
    return (
      <div className={cn("w-full p-8 text-center", className)}>
        <div className="text-red-500 text-xl mb-4">Failed to load market data</div>
        <p className="text-gray-600 mb-4">Please try again later</p>
        <Button onClick={() => refetch()} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    )
  }

  if (!data || data.stocks.length === 0) {
    return (
      <div className={cn("w-full p-8 text-center", className)}>
        <div className="text-gray-500 text-xl mb-4">No stocks in watchlist</div>
        <p className="text-gray-600">Add stocks to your watchlist to see real-time data</p>
      </div>
    )
  }

  return (
    <div className={cn("w-full p-4", className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <h2 className="text-xl font-semibold">Market Watch</h2>
          <Badge variant="outline" className="text-sm">
            {data.count} stocks
          </Badge>
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <>
                <Wifi className="h-4 w-4 text-green-500" />
                <span className="text-xs text-green-500">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4 text-red-500" />
                <span className="text-xs text-red-500">Disconnected</span>
              </>
            )}
          </div>
          {lastUpdate && (
            <span className="text-xs text-gray-500">
              Last update: {formatTimestamp(lastUpdate)}
            </span>
          )}
        </div>
        <Button
          onClick={() => refetch()}
          variant="outline"
          size="sm"
          disabled={isLoading}
        >
          <RefreshCw className={cn("h-4 w-4 mr-2", isLoading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Table */}
      <ScrollArea className="h-[600px] rounded-md border">
        <Table>
          <TableHeader>
            <TableRow className="bg-gray-50 dark:bg-gray-800">
              <TableHead
                className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={() => handleSort('ticker')}
              >
                Symbol {sortBy === 'ticker' && (sortOrder === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={() => handleSort('price')}
              >
                Last {sortBy === 'price' && (sortOrder === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={() => handleSort('change')}
              >
                Change {sortBy === 'change' && (sortOrder === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={() => handleSort('change_percent')}
              >
                Change % {sortBy === 'change_percent' && (sortOrder === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead>Volume</TableHead>
              <TableHead>Market Cap</TableHead>
              <TableHead>Sector</TableHead>
              <TableHead>RSI</TableHead>
              <TableHead>Trend</TableHead>
              <TableHead>Analysis</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedStocks.map((stock) => (
              <TableRow
                key={stock.ticker}
                className={cn(
                  "cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors",
                  stock.priceFlash === 'up' && "bg-green-500/10",
                  stock.priceFlash === 'down' && "bg-red-500/10"
                )}
                onClick={() => onRowClick?.(stock.ticker)}
              >
                <TableCell>
                  <div>
                    <div className="font-medium">{stock.ticker}</div>
                    <div className="text-sm text-gray-500">{stock.name}</div>
                  </div>
                </TableCell>
                <TableCell className="font-mono">
                  {formatPrice(stock.price)}
                </TableCell>
                <TableCell>
                  <div className="flex items-center space-x-1">
                    {stock.change > 0 ? (
                      <TrendingUp className="h-3 w-3 text-green-500" />
                    ) : stock.change < 0 ? (
                      <TrendingDown className="h-3 w-3 text-red-500" />
                    ) : (
                      <Minus className="h-3 w-3 text-gray-500" />
                    )}
                    <span className={cn(
                      "font-mono text-sm",
                      stock.change > 0 ? "text-green-500" :
                      stock.change < 0 ? "text-red-500" : "text-gray-500"
                    )}>
                      {formatPrice(Math.abs(stock.change))}
                    </span>
                  </div>
                </TableCell>
                <TableCell>
                  <span className={cn(
                    "font-mono text-sm font-medium",
                    stock.change_percent > 0 ? "text-green-500" :
                    stock.change_percent < 0 ? "text-red-500" : "text-gray-500"
                  )}>
                    {formatPercent(stock.change_percent)}
                  </span>
                </TableCell>
                <TableCell className="font-mono text-sm">
                  {formatVolume(stock.volume)}
                </TableCell>
                <TableCell className="font-mono text-sm">
                  {formatMarketCap(stock.market_cap)}
                </TableCell>
                <TableCell>
                  <Badge className={getSectorColor(stock.sector)} variant="secondary">
                    {stock.sector || 'N/A'}
                  </Badge>
                </TableCell>
                <TableCell>
                  <div
                    className={cn("font-mono text-sm", getRSIColor(stock.rsi))}
                    title={getRSITooltip(stock.rsi)}
                  >
                    {stock.rsi?.toFixed(1) ?? 'N/A'}
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex items-center space-x-2">
                    {getTrendIcon(stock.trend)}
                    <span className="text-sm">{stock.trend_label}</span>
                  </div>
                </TableCell>
                <TableCell>
                  <div className="text-xs">
                    {stock.rsi && stock.macd !== null && stock.macd_signal !== null && (
                      <div className={cn(
                        "font-medium",
                        stock.trend === 'upward' ? "text-green-500" :
                        stock.trend === 'downward' ? "text-red-500" : "text-gray-500"
                      )}>
                        RSI: {stock.rsi.toFixed(1)}, MACD {stock.macd > stock.macd_signal ? '>' : '<'} Signal
                      </div>
                    )}
                    <div className="text-gray-500">
                      SMA: {stock.sma20 ? formatPrice(stock.sma20) : 'N/A'}
                    </div>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>

      {/* Footer */}
      <div className="mt-4 text-xs text-gray-500 text-center">
        Real-time data updates every 30 seconds • Click rows to view charts
      </div>
    </div>
  )
}