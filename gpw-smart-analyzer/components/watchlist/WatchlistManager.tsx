'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import {
  Plus,
  X,
  Search,
  TrendingUp,
  TrendingDown,
  Minus,
  Star,
  StarOff,
  RefreshCw
} from 'lucide-react'
import { useWatchlist, useStockSearch } from '@/lib/hooks/useWatchlist'
import { cn } from '@/lib/utils'

interface WatchlistManagerProps {
  onStockSelect?: (ticker: string) => void
  className?: string
}

export function WatchlistManager({ onStockSelect, className }: WatchlistManagerProps) {
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)

  const {
    watchlist,
    count,
    isLoading,
    error,
    refetch,
    addToWatchlist,
    removeFromWatchlist,
    isAdding,
    isRemoving,
  } = useWatchlist()

  const {
    searchQuery,
    searchResults,
    isSearching,
    handleSearch,
    clearSearch,
  } = useStockSearch()

  const handleAddStock = async (ticker: string) => {
    try {
      await addToWatchlist(ticker)
      clearSearch()
      setIsAddDialogOpen(false)
    } catch (error) {
      console.error('Failed to add stock:', error)
    }
  }

  const handleRemoveStock = async (ticker: string) => {
    try {
      await removeFromWatchlist(ticker)
    } catch (error) {
      console.error('Failed to remove stock:', error)
    }
  }

  const getChangeColor = (change: number) => {
    if (change > 0) return 'tv-up'
    if (change < 0) return 'tv-down'
    return 'tv-neutral'
  }

  const getChangeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-3 w-3" />
    if (change < 0) return <TrendingDown className="h-3 w-3" />
    return <Minus className="h-3 w-3" />
  }

  if (error) {
    return (
      <Card className={cn("tv-panel", className)}>
        <div className="p-4">
          <div className="text-red-400 text-sm">Error loading watchlist</div>
          <Button
            onClick={() => refetch()}
            variant="outline"
            size="sm"
            className="mt-2"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </Card>
    )
  }

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="tv-panel-header flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Star className="h-4 w-4 text-yellow-500" />
          <span className="font-medium">Watchlist</span>
          <Badge variant="secondary" className="text-xs">
            {count}
          </Badge>
        </div>

        <div className="flex items-center space-x-1">
          <Button
            onClick={() => refetch()}
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0 text-gray-400 hover:text-white"
          >
            <RefreshCw className={cn("h-3 w-3", isLoading && "animate-spin")} />
          </Button>

          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 text-gray-400 hover:text-white"
              >
                <Plus className="h-3 w-3" />
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-gray-800 border-gray-700 text-white max-w-md">
              <DialogHeader>
                <DialogTitle>Add to Watchlist</DialogTitle>
              </DialogHeader>

              <div className="space-y-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Search stocks (e.g., PKN, PEO, WIG)..."
                    value={searchQuery}
                    onChange={(e) => handleSearch(e.target.value)}
                    className="pl-10 bg-gray-700 border-gray-600 text-white placeholder-gray-400"
                  />
                </div>

                {(isSearching || searchResults.length > 0) && (
                  <ScrollArea className="h-60">
                    <div className="space-y-2">
                      {searchResults.map((stock) => (
                        <Card
                          key={stock.ticker}
                          className="bg-gray-700 border-gray-600 cursor-pointer hover:bg-gray-600 transition-colors"
                        >
                          <div className="p-3 flex items-center justify-between">
                            <div
                              className="flex-1 cursor-pointer"
                              onClick={() => handleAddStock(stock.ticker)}
                            >
                              <div className="flex items-center space-x-2">
                                <span className="font-medium text-white">{stock.ticker}</span>
                                <span className="text-sm text-gray-400">{stock.name}</span>
                              </div>
                              <div className="flex items-center space-x-2 mt-1">
                                <span className="text-sm text-white">
                                  {stock.price?.toFixed(2)} PLN
                                </span>
                                <div className={cn("flex items-center space-x-1", getChangeColor(stock.change))}>
                                  {getChangeIcon(stock.change)}
                                  <span className="text-xs">
                                    {stock.change >= 0 ? '+' : ''}
                                    {stock.change_percent?.toFixed(2)}%
                                  </span>
                                </div>
                              </div>
                            </div>

                            <Button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleAddStock(stock.ticker)
                              }}
                              disabled={isAdding}
                              size="sm"
                              className="h-6 w-6 p-0 bg-blue-600 hover:bg-blue-700"
                            >
                              <Plus className="h-3 w-3" />
                            </Button>
                          </div>
                        </Card>
                      ))}

                      {isSearching && (
                        <div className="text-center text-gray-400 py-4">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto mb-2" />
                          <p className="text-sm">Searching...</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                )}

                {!isSearching && searchQuery.length >= 2 && searchResults.length === 0 && (
                  <div className="text-center text-gray-400 py-4">
                    <p className="text-sm">No stocks found matching "{searchQuery}"</p>
                  </div>
                )}

                {searchQuery.length < 2 && (
                  <div className="text-center text-gray-400 py-4">
                    <p className="text-sm">Enter at least 2 characters to search</p>
                  </div>
                )}
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Watchlist Content */}
      <div className="flex-1 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />
          </div>
        ) : watchlist.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <StarOff className="h-8 w-8 mb-2" />
            <p className="text-sm text-center">No stocks in watchlist</p>
            <p className="text-xs text-center mt-1">Click + to add stocks</p>
          </div>
        ) : (
          <ScrollArea className="h-full">
            <div className="p-2 space-y-1">
              {watchlist.map((stock) => (
                <Card
                  key={stock.ticker}
                  className={cn(
                    "bg-gray-700 border-gray-600 cursor-pointer transition-colors group",
                    "hover:bg-gray-600"
                  )}
                  onClick={() => onStockSelect?.(stock.ticker)}
                >
                  <div className="p-2 flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-white text-sm truncate">
                          {stock.ticker}
                        </span>
                        {stock.error && (
                          <Badge variant="destructive" className="text-xs">
                            Error
                          </Badge>
                        )}
                      </div>

                      {!stock.error && (
                        <>
                          <div className="text-xs text-gray-400 truncate">
                            {stock.name}
                          </div>
                          <div className="flex items-center space-x-2 mt-1">
                            <span className="text-sm text-white">
                              {stock.price?.toFixed(2)} PLN
                            </span>
                            <div className={cn("flex items-center space-x-1", getChangeColor(stock.change))}>
                              {getChangeIcon(stock.change)}
                              <span className="text-xs">
                                {stock.change >= 0 ? '+' : ''}
                                {stock.change_percent?.toFixed(2)}%
                              </span>
                            </div>
                          </div>
                        </>
                      )}
                    </div>

                    <Button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleRemoveStock(stock.ticker)
                      }}
                      disabled={isRemoving}
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 text-gray-400 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          </ScrollArea>
        )}
      </div>
    </div>
  )
}