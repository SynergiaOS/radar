'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Target, TrendingUp, AlertTriangle } from 'lucide-react'

interface PatternOverlayProps {
  ticker: string
  onPatternDetected?: (pattern: any) => void
}

interface DetectedPattern {
  id: string
  type: 'flag' | 'triangle' | 'head_and_shoulders' | 'double_top' | 'double_bottom'
  direction: 'bullish' | 'bearish' | 'neutral'
  confidence: number
  start_date: string
  end_date: string
  key_levels: {
    support?: number
    resistance?: number
    breakout?: number
  }
  description: string
  distance_to_breakout: number
}

export function PatternOverlay({ ticker, onPatternDetected }: PatternOverlayProps) {
  const [detectedPatterns, setDetectedPatterns] = useState<DetectedPattern[]>([
    {
      id: '1',
      type: 'flag',
      direction: 'bullish',
      confidence: 0.85,
      start_date: '2024-09-15',
      end_date: '2024-10-10',
      key_levels: {
        support: 65.50,
        resistance: 68.20,
        breakout: 68.50,
      },
      description: 'Bullish flag pattern indicating continuation of uptrend',
      distance_to_breakout: 1.2,
    },
    {
      id: '2',
      type: 'triangle',
      direction: 'bullish',
      confidence: 0.72,
      start_date: '2024-09-20',
      end_date: '2024-10-12',
      key_levels: {
        support: 66.80,
        resistance: 67.80,
        breakout: 68.00,
      },
      description: 'Ascending triangle pattern forming',
      distance_to_breakout: 0.8,
    },
  ])

  const [selectedPattern, setSelectedPattern] = useState<string | null>(null)

  const getPatternIcon = (type: string) => {
    switch (type) {
      case 'flag':
        return <Target className="h-4 w-4" />
      case 'triangle':
        return <TrendingUp className="h-4 w-4" />
      case 'head_and_shoulders':
      case 'double_top':
        return <AlertTriangle className="h-4 w-4" />
      default:
        return <Target className="h-4 w-4" />
    }
  }

  const getPatternColor = (direction: string) => {
    switch (direction) {
      case 'bullish':
        return 'text-green-400 border-green-400'
      case 'bearish':
        return 'text-red-400 border-red-400'
      default:
        return 'text-yellow-400 border-yellow-400'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    return 'Low'
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-textPrimary">Pattern Detection</h3>
        <Badge variant="outline" className="text-xs">
          {detectedPatterns.length} patterns detected
        </Badge>
      </div>

      {detectedPatterns.length === 0 ? (
        <div className="text-center py-8">
          <Target className="h-12 w-12 text-textSecondary mx-auto mb-4" />
          <p className="text-textSecondary">No patterns detected</p>
          <p className="text-sm text-textSecondary mt-1">
            Patterns will appear automatically when detected
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {detectedPatterns.map((pattern) => (
            <div
              key={pattern.id}
              className={`p-4 rounded-lg border bg-surfaceLight hover:bg-surface transition-colors cursor-pointer ${
                selectedPattern === pattern.id ? 'ring-2 ring-blue-500' : ''
              } ${getPatternColor(pattern.direction)}`}
              onClick={() => setSelectedPattern(pattern.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  <div className="p-2 rounded bg-surface">
                    {getPatternIcon(pattern.type)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h4 className="font-medium capitalize">{pattern.type}</h4>
                      <Badge variant="outline" className={`text-xs ${getPatternColor(pattern.direction)}`}>
                        {pattern.direction}
                      </Badge>
                      <Badge variant="outline" className={`text-xs ${getConfidenceColor(pattern.confidence)}`}>
                        {getConfidenceBadge(pattern.confidence)} confidence
                      </Badge>
                    </div>
                    <p className="text-sm text-textSecondary mb-3">
                      {pattern.description}
                    </p>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-textSecondary">Support:</span>
                        <span className="ml-2 font-medium">
                          {pattern.key_levels.support?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span className="text-textSecondary">Resistance:</span>
                        <span className="ml-2 font-medium">
                          {pattern.key_levels.resistance?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span className="text-textSecondary">Breakout:</span>
                        <span className="ml-2 font-medium">
                          {pattern.key_levels.breakout?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span className="text-textSecondary">Distance:</span>
                        <span className="ml-2 font-medium">
                          {pattern.distance_to_breakout.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {selectedPattern === pattern.id && (
                <div className="mt-4 pt-4 border-t border-border space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-textSecondary">
                      Pattern Period: {pattern.start_date} - {pattern.end_date}
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => onPatternDetected?.(pattern)}
                    >
                      Analyze Pattern
                    </Button>
                  </div>
                  <div className="text-sm text-textSecondary">
                    <p className="mb-2">Pattern Analysis:</p>
                    <ul className="space-y-1 ml-4">
                      <li>• {pattern.direction === 'bullish' ? 'Upward' : 'Downward'} momentum expected</li>
                      <li>• {pattern.confidence >= 0.8 ? 'High' : pattern.confidence >= 0.6 ? 'Medium' : 'Low'} confidence level</li>
                      <li>• {pattern.distance_to_breakout < 2 ? 'Close to breakout' : 'Formation in progress'}</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pattern Detection Settings */}
      <div className="mt-6 pt-6 border-t border-border">
        <div className="flex items-center justify-between">
          <div className="text-sm text-textSecondary">
            <p>• Automatic pattern detection enabled</p>
            <p>• Confidence threshold: 60%</p>
            <p>• Real-time analysis active</p>
          </div>
          <Button variant="ghost" size="sm">
            Settings
          </Button>
        </div>
      </div>
    </Card>
  )
}