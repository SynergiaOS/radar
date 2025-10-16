export interface IndicatorConfig {
  name: string
  type: 'overlay' | 'oscillator'
  parameters: Record<string, number>
  color?: string
  enabled: boolean
}

export interface IndicatorData {
  name: string
  data: Array<{
    time: number
    value: number | null
  }>
  color: string
  type: 'line' | 'histogram' | 'area'
}

export interface TechnicalIndicator {
  sma: (data: number[], period: number) => number[]
  ema: (data: number[], period: number) => number[]
  rsi: (data: number[], period: number) => number[]
  macd: (data: number[], fast: number, slow: number, signal: number) => {
    macd: number[]
    signal: number[]
    histogram: number[]
  }
  bollinger: (data: number[], period: number, stdDev: number) => {
    upper: number[]
    middle: number[]
    lower: number[]
  }
  atr: (high: number[], low: number[], close: number[], period: number) => number[]
  adx: (high: number[], low: number[], close: number[], period: number) => {
    adx: number[]
    di_plus: number[]
    di_minus: number[]
  }
}

export interface IndicatorResult {
  name: string
  values: number[]
  signals?: Array<{
    time: number
    type: 'buy' | 'sell' | 'hold'
    strength: number
    reason: string
  }>
}