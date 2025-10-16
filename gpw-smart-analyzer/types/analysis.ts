export interface Analysis {
  all_stocks: StockAnalysis[]
  count: number
  recommendations: StockAnalysis[]
  index: 'WIG30' | 'WIG20'
  thresholds: {
    roe: number
    pe: number
    dual_filter: boolean
  }
  timestamp: string
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

export interface SystemStatus {
  last_update: string | null
  active_index: 'WIG30' | 'WIG20'
  recommendations_count: number
  thresholds: {
    roe_threshold: number
    pe_threshold: number
    dual_filter_enabled: boolean
  }
  monitoring?: boolean
  mlModels?: boolean
  rlAgent?: boolean
}

export interface Config {
  active_index: 'WIG30' | 'WIG20'
  roe_threshold: number
  pe_threshold: number
  dual_filter: boolean
}

export interface AnalysisResult {
  ticker: string
  score: number
  factors: {
    fundamental: number
    technical: number
    sentiment: number
  }
  recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL'
  confidence: number
  reasons: string[]
}