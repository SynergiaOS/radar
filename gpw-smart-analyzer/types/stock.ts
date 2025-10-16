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

export interface StockData {
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