'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import {
  Home,
  TrendingUp,
  Activity,
  Bell,
  ChevronLeft,
  ChevronRight,
  Star,
  Clock,
  Settings,
  Plus,
  Minus,
} from 'lucide-react'

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)

  const navigationItems = [
    {
      href: '/',
      icon: Home,
      label: 'Dashboard',
      active: true,
    },
    {
      href: '/signals',
      icon: TrendingUp,
      label: 'Signals',
      badge: '12',
    },
    {
      href: '/monitor',
      icon: Activity,
      label: 'Monitor',
      badge: '3',
    },
    {
      href: '/alerts',
      icon: Bell,
      label: 'Alerts',
      badge: '7',
    },
  ]

  const watchlistStocks = [
    { ticker: 'XTB.WA', price: 67.84, change: 0.65 },
    { ticker: 'TXT.WA', price: 50.75, change: -1.2 },
    { ticker: 'PKN.WA', price: 88.13, change: 0.0 },
  ]

  return (
    <div
      className={cn(
        'relative border-r border-border bg-surface transition-all duration-300 ease-in-out',
        isCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Toggle Button */}
      <Button
        variant="ghost"
        size="sm"
        className="absolute -right-3 top-6 z-10 h-6 w-6 rounded-full border border-border bg-surface p-0"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        {isCollapsed ? (
          <ChevronRight className="h-3 w-3" />
        ) : (
          <ChevronLeft className="h-3 w-3" />
        )}
      </Button>

      <div className="flex h-full flex-col">
        {/* Navigation */}
        <nav className="p-4">
          <ul className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={cn(
                      'flex items-center justify-between rounded-lg p-2 transition-colors',
                      item.active
                        ? 'bg-blue-600/10 text-blue-600'
                        : 'text-textSecondary hover:text-textPrimary hover:bg-surfaceLight'
                    )}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon className="h-5 w-5" />
                      {!isCollapsed && <span>{item.label}</span>}
                    </div>
                    {!isCollapsed && item.badge && (
                      <Badge variant="secondary" className="text-xs">
                        {item.badge}
                      </Badge>
                    )}
                  </Link>
                </li>
              )
            })}
          </ul>
        </nav>

        {/* Watchlist */}
        {!isCollapsed && (
          <div className="flex-1 overflow-hidden">
            <div className="px-4 py-2">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-textSecondary">
                  Watchlist
                </h3>
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <Plus className="h-3 w-3" />
                </Button>
              </div>

              <div className="space-y-2">
                {watchlistStocks.map((stock) => (
                  <Link
                    key={stock.ticker}
                    href={`/chart/${stock.ticker}`}
                    className="block rounded-lg p-2 hover:bg-surfaceLight transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-sm">{stock.ticker}</div>
                        <div className="text-xs text-textSecondary">
                          {stock.price.toFixed(2)} PLN
                        </div>
                      </div>
                      <div
                        className={cn(
                          'text-xs font-medium',
                          stock.change >= 0 ? 'text-up' : 'text-down'
                        )}
                      >
                        {stock.change >= 0 ? '+' : ''}
                        {stock.change.toFixed(2)}%
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Quick Stats */}
        {!isCollapsed && (
          <div className="border-t border-border p-4">
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-textSecondary">Active Signals</span>
                <Badge variant="buy">12</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-textSecondary">Monitoring</span>
                <Badge variant="outline">Running</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-textSecondary">Last Update</span>
                <span className="text-textSecondary">2m ago</span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-border">
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start"
              >
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        )}

        {/* Collapsed Quick Stats */}
        {isCollapsed && (
          <div className="border-t border-border p-4">
            <div className="space-y-4">
              <div className="text-center">
                <Badge variant="buy" className="text-xs">
                  12
                </Badge>
                <div className="text-xs text-textSecondary mt-1">Signals</div>
              </div>
              <div className="text-center">
                <div className="h-2 w-2 bg-green-500 rounded-full mx-auto" />
                <div className="text-xs text-textSecondary mt-1">Live</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}