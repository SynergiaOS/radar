export interface ChartOptions {
  width?: number
  height?: number
  layout: {
    background: {
      type: 'solid' | 'gradient'
      color: string
    }
    textColor: string
  }
  grid: {
    vertLines: { color: string }
    horzLines: { color: string }
  }
  timeScale: {
    borderColor: string
    textColor: string
    timeVisible: boolean
    secondsVisible: boolean
  }
}

export interface SeriesOptions {
  color?: string
  lineWidth?: number
  lineStyle?: number
  title?: string
  priceScaleId?: string
}

export interface PriceLineOptions {
  price: number
  color: string
  lineWidth?: number
  lineStyle?: number
  title?: string
}

export interface ChartMarker {
  time: number
  position: 'aboveBar' | 'belowBar' | 'inBar'
  color: string
  shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown'
  text?: string
}

export interface PatternAnnotation {
  id: string
  type: 'support' | 'resistance' | 'trendline' | 'breakout'
  startPrice: number
  endPrice: number
  startTime: number
  endTime: number
  color: string
  label?: string
}