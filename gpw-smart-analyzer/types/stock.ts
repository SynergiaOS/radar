export interface OHLCVData {
  time: string | number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface StockInfo {
  ticker: string
  name: string
  currency: string
  current_price: number
  change: number
  change_percent: number
}

export interface CandlestickData {
  x: number  // milliseconds timestamp
  o: number  // open
  h: number  // high
  l: number  // low
  c: number  // close
  v: number  // volume
}

export interface IndicatorData {
  x: number  // milliseconds timestamp
  y: number | null
}

export interface VolumeData {
  x: number  // milliseconds timestamp
  y: number  // volume value
}

export interface ChartResponse {
  info: StockInfo
  candlestick: CandlestickData[]
  volume: VolumeData[]
  indicators: Record<string, IndicatorData[]>
}

export interface StockData {
  ticker: string
  data: OHLCVData[]
  info: StockInfo
}

// Keep the old interface for backward compatibility, but prefer ChartResponse for new code
export interface LegacyStockData {
  ticker: string
  data: OHLCVData[]
  info: StockInfo
}

export interface StockAnalysis {
  ticker: string
  name: string
  net_income: number
  equity: number | null
  roe: number | null
  current_price: number | null
  eps: number | null
  pe_ratio: number | null
  profitable: boolean
  quarter_end: string
  decision: 'KUP' | 'TRZYMAJ' | 'SPRZEDAJ'
  decision_color: string
}

export interface ComparisonData {
  [ticker: string]: {
    name: string
    prices: { x: number; y: number }[]
  }
}