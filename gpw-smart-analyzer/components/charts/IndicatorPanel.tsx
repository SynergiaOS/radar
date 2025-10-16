'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Settings, Eye, EyeOff } from 'lucide-react'
import { useStockIndicators } from '@/lib/hooks/useStockData'

interface IndicatorPanelProps {
  ticker: string
  onIndicatorChange?: (indicator: string, enabled: boolean) => void
}

interface IndicatorConfig {
  id: string
  name: string
  period?: number
  color: string
  enabled: boolean
  value?: number
  interpretation?: string
}

export function IndicatorPanel({ ticker, onIndicatorChange }: IndicatorPanelProps) {
  const [indicators, setIndicators] = useState<IndicatorConfig[]>([
    { id: 'sma20', name: 'SMA 20', period: 20, color: '#2962FF', enabled: true },
    { id: 'sma50', name: 'SMA 50', period: 50, color: '#FF6D00', enabled: true },
    { id: 'sma200', name: 'SMA 200', period: 200, color: '#F50057', enabled: false },
    { id: 'ema12', name: 'EMA 12', period: 12, color: '#00C853', enabled: false },
    { id: 'ema26', name: 'EMA 26', period: 26, color: '#FFAB00', enabled: false },
    { id: 'rsi14', name: 'RSI 14', period: 14, color: '#FF4081', enabled: false },
    { id: 'macd', name: 'MACD', color: '#7C4DFF', enabled: false },
    { id: 'bb', name: 'Bollinger Bands', color: '#00B0FF', enabled: false },
    { id: 'atr', name: 'ATR 14', period: 14, color: '#FFD600', enabled: false },
    { id: 'vwap', name: 'VWAP', color: '#B388FF', enabled: false },
  ])

  const { data: indicatorData, isLoading } = useStockIndicators(ticker)

  const toggleIndicator = (indicatorId: string) => {
    setIndicators(prev => prev.map(ind =>
      ind.id === indicatorId ? { ...ind, enabled: !ind.enabled } : ind
    ))
    onIndicatorChange?.(indicatorId, !indicators.find(ind => ind.id === indicatorId)?.enabled)
  }

  const getRSIInterpretation = (rsi: number): string => {
    if (rsi >= 70) return 'Overbought - Consider selling'
    if (rsi <= 30) return 'Oversold - Consider buying'
    if (rsi >= 60) return 'Bullish momentum'
    if (rsi <= 40) return 'Bearish momentum'
    return 'Neutral'
  }

  const getRSIColor = (rsi: number): string => {
    if (rsi >= 70) return 'text-red-400'
    if (rsi <= 30) return 'text-green-400'
    return 'text-textSecondary'
  }

  const indicatorPresets = [
    { name: 'Trend Following', indicators: ['sma20', 'sma50', 'sma200'] },
    { name: 'Momentum', indicators: ['rsi14', 'macd', 'ema12'] },
    { name: 'Volatility', indicators: ['bb', 'atr', 'vwap'] },
    { name: 'All Indicators', indicators: indicators.map(ind => ind.id) },
  ]

  const applyPreset = (presetIndicators: string[]) => {
    setIndicators(prev => prev.map(ind => ({
      ...ind,
      enabled: presetIndicators.includes(ind.id)
    })))
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-textPrimary">Technical Indicators</h3>
        <Button variant="ghost" size="sm">
          <Settings className="h-4 w-4" />
        </Button>
      </div>

      {/* Preset Buttons */}
      <div className="mb-6">
        <p className="text-sm text-textSecondary mb-3">Quick Presets:</p>
        <div className="grid grid-cols-2 gap-2">
          {indicatorPresets.map((preset) => (
            <Button
              key={preset.name}
              variant="outline"
              size="sm"
              onClick={() => applyPreset(preset.indicators)}
              className="text-xs"
            >
              {preset.name}
            </Button>
          ))}
        </div>
      </div>

      {/* Moving Averages */}
      <div className="space-y-4">
        <div>
          <h4 className="text-sm font-medium text-textPrimary mb-3">Moving Averages</h4>
          <div className="space-y-3">
            {indicators
              .filter(ind => ind.id.includes('sma') || ind.id.includes('ema'))
              .map((indicator) => (
                <div
                  key={indicator.id}
                  className="flex items-center justify-between p-3 rounded-lg bg-surfaceLight hover:bg-surface transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleIndicator(indicator.id)}
                      className="h-6 w-6 p-0"
                    >
                      {indicator.enabled ? (
                        <Eye className="h-3 w-3" />
                      ) : (
                        <EyeOff className="h-3 w-3" />
                      )}
                    </Button>
                    <div className="flex items-center space-x-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: indicator.color }}
                      />
                      <span className="text-sm font-medium">{indicator.name}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">
                      {indicatorData && indicatorData[
                        indicator.id.replace('sma', 'sma_').replace('ema', 'ema_')
                      ] ?
                        indicatorData[indicator.id.replace('sma', 'sma_').replace('ema', 'ema_')]?.toFixed(2) :
                        'N/A'
                      }
                    </div>
                    {indicatorData && indicatorData.price && indicatorData[
                      indicator.id.replace('sma', 'sma_').replace('ema', 'ema_')
                    ] && (
                      <div className="text-xs text-textSecondary">
                        {indicatorData.price > indicatorData[
                          indicator.id.replace('sma', 'sma_').replace('ema', 'ema_')
                        ] ? 'Above' : 'Below'} MA
                      </div>
                    )}
                  </div>
                </div>
              ))}
          </div>
        </div>

        {/* Oscillators */}
        <div>
          <h4 className="text-sm font-medium text-textPrimary mb-3">Oscillators</h4>
          <div className="space-y-3">
            {indicators
              .filter(ind => ind.id.includes('rsi') || ind.id.includes('macd'))
              .map((indicator) => (
                <div
                  key={indicator.id}
                  className="flex items-center justify-between p-3 rounded-lg bg-surfaceLight hover:bg-surface transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleIndicator(indicator.id)}
                      className="h-6 w-6 p-0"
                    >
                      {indicator.enabled ? (
                        <Eye className="h-3 w-3" />
                      ) : (
                        <EyeOff className="h-3 w-3" />
                      )}
                    </Button>
                    <div className="flex items-center space-x-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: indicator.color }}
                      />
                      <span className="text-sm font-medium">{indicator.name}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    {indicator.id === 'rsi14' && indicatorData?.rsi && (
                      <>
                        <div className={`text-sm font-medium ${getRSIColor(indicatorData.rsi)}`}>
                          {indicatorData.rsi.toFixed(2)}
                        </div>
                        <div className="text-xs text-textSecondary">
                          {getRSIInterpretation(indicatorData.rsi)}
                        </div>
                      </>
                    )}
                    {indicator.id === 'macd' && indicatorData?.macd && (
                      <>
                        <div className="text-sm font-medium">
                          {indicatorData.macd.toFixed(4)}
                        </div>
                        <div className="text-xs text-textSecondary">
                          {indicatorData.macd > 0 ? 'Bullish' : 'Bearish'}
                        </div>
                      </>
                    )}
                    {((indicator.id === 'rsi14' && !indicatorData?.rsi) ||
                      (indicator.id === 'macd' && !indicatorData?.macd)) && (
                      <div className="text-sm font-medium">N/A</div>
                    )}
                  </div>
                </div>
              ))}
          </div>
        </div>

        {/* Volatility */}
        <div>
          <h4 className="text-sm font-medium text-textPrimary mb-3">Volatility</h4>
          <div className="space-y-3">
            {indicators
              .filter(ind => ind.id.includes('bb') || ind.id.includes('atr') || ind.id.includes('vwap'))
              .map((indicator) => (
                <div
                  key={indicator.id}
                  className="flex items-center justify-between p-3 rounded-lg bg-surfaceLight hover:bg-surface transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleIndicator(indicator.id)}
                      className="h-6 w-6 p-0"
                    >
                      {indicator.enabled ? (
                        <Eye className="h-3 w-3" />
                      ) : (
                        <EyeOff className="h-3 w-3" />
                      )}
                    </Button>
                    <div className="flex items-center space-x-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: indicator.color }}
                      />
                      <span className="text-sm font-medium">{indicator.name}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    {indicator.id === 'atr14' && indicatorData?.atr && (
                      <div className="text-sm font-medium">
                        {indicatorData.atr.toFixed(4)}
                      </div>
                    )}
                    {(indicator.id === 'atr14' && !indicatorData?.atr) ||
                      (indicator.id === 'bb' || indicator.id === 'vwap') && (
                      <div className="text-sm font-medium">N/A</div>
                    )}
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="mt-6 pt-6 border-t border-border">
        <div className="text-xs text-textSecondary">
          <p>• Click eye icon to toggle indicators on chart</p>
          <p>• Presets apply multiple indicators at once</p>
          <p>• Values update in real-time with price data</p>
        </div>
      </div>
    </Card>
  )
}