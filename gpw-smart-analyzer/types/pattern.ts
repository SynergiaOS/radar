export interface Pattern {
  id: string
  name: string
  type: 'reversal' | 'continuation' | 'consolidation'
  direction: 'bullish' | 'bearish' | 'neutral'
  confidence: number
  start_time: number
  end_time: number
  start_price: number
  end_price: number
  targets?: number[]
  stop_loss?: number
}

export interface CandlestickPattern {
  type: 'doji' | 'hammer' | 'engulfing' | 'morning_star' | 'evening_star' | 'harami'
  direction: 'bullish' | 'bearish'
  confidence: number
  time: number
  description: string
}

export interface TrendLine {
  id: string
  start_time: number
  end_time: number
  start_price: number
  end_price: number
  type: 'support' | 'resistance'
  strength: number
  touches: number
  is_active: boolean
}

export interface SupportResistanceLevel {
  price: number
  strength: number
  touches: number
  last_touch: number
  type: 'support' | 'resistance'
}

export interface PatternResult {
  pattern: Pattern
  success_rate: number
  average_profit: number
  average_holding_period: number
  samples: number
}