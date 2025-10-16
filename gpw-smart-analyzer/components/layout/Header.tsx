'use client'

import { useState } from 'react'
import Link from 'next/link'
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  RefreshCw,
  Play,
  Pause,
  Bell,
  Settings,
  Menu,
  TrendingUp,
  AlertTriangle,
  Activity,
} from 'lucide-react'

export function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [systemStatus, setSystemStatus] = useState<'online' | 'offline' | 'warning'>('online')

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-surface/95 backdrop-blur supports-[backdrop-filter]:bg-surface/60">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Logo and Navigation */}
        <div className="flex items-center space-x-6">
          <Link href="/" className="flex items-center space-x-2">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <TrendingUp className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-bold text-textPrimary">GPW Smart Analyzer</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            <Link
              href="/"
              className="text-textSecondary hover:text-textPrimary transition-colors"
            >
              Dashboard
            </Link>
            <Link
              href="/signals"
              className="text-textSecondary hover:text-textPrimary transition-colors"
            >
              Signals
            </Link>
            <Link
              href="/monitor"
              className="text-textSecondary hover:text-textPrimary transition-colors"
            >
              Monitor
            </Link>
            <Link
              href="/alerts"
              className="text-textSecondary hover:text-textPrimary transition-colors"
            >
              Alerts
            </Link>
          </nav>
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center space-x-4">
          {/* System Status */}
          <div className="hidden md:flex items-center space-x-2">
            <div
              className={`h-2 w-2 rounded-full ${
                systemStatus === 'online'
                  ? 'bg-green-500'
                  : systemStatus === 'warning'
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-textSecondary capitalize">
              {systemStatus}
            </span>
          </div>

          {/* Index Selector */}
          <Select defaultValue="WIG30">
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="WIG30">WIG30</SelectItem>
              <SelectItem value="WIG20">WIG20</SelectItem>
            </SelectContent>
          </Select>

          {/* Action Buttons */}
          <div className="hidden md:flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              className="flex items-center space-x-2"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Run Analysis</span>
            </Button>

            <Button
              variant={isMonitoring ? "secondary" : "outline"}
              size="sm"
              className="flex items-center space-x-2"
              onClick={() => setIsMonitoring(!isMonitoring)}
            >
              {isMonitoring ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span>{isMonitoring ? 'Stop' : 'Start'} Monitoring</span>
            </Button>

            <Button variant="outline" size="sm" className="relative">
              <Bell className="h-4 w-4" />
              <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full" />
            </Button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm">
                  <Settings className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>Settings</DropdownMenuItem>
                <DropdownMenuItem>Preferences</DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem>Export Data</DropdownMenuItem>
                <DropdownMenuItem>Import Configuration</DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem>About</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Mobile Menu Toggle */}
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden border-t border-border bg-surface p-4">
          <div className="flex flex-col space-y-4">
            <nav className="flex flex-col space-y-2">
              <Link
                href="/"
                className="text-textSecondary hover:text-textPrimary transition-colors px-2 py-1"
              >
                Dashboard
              </Link>
              <Link
                href="/signals"
                className="text-textSecondary hover:text-textPrimary transition-colors px-2 py-1"
              >
                Signals
              </Link>
              <Link
                href="/monitor"
                className="text-textSecondary hover:text-textPrimary transition-colors px-2 py-1"
              >
                Monitor
              </Link>
              <Link
                href="/alerts"
                className="text-textSecondary hover:text-textPrimary transition-colors px-2 py-1"
              >
                Alerts
              </Link>
            </nav>

            <div className="flex flex-col space-y-2">
              <Button size="sm" className="w-full">
                <RefreshCw className="h-4 w-4 mr-2" />
                Run Analysis
              </Button>
              <Button
                variant={isMonitoring ? "secondary" : "outline"}
                size="sm"
                className="w-full"
                onClick={() => setIsMonitoring(!isMonitoring)}
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
      )}
    </header>
  )
}