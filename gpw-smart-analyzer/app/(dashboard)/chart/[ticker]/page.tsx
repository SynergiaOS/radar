import { TradingChart } from '@/components/charts/TradingChart'
import { IndicatorPanel } from '@/components/charts/IndicatorPanel'
import { PatternOverlay } from '@/components/charts/PatternOverlay'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { notFound } from 'next/navigation'

interface ChartPageProps {
  params: {
    ticker: string
  }
}

export default function ChartPage({ params }: ChartPageProps) {
  const ticker = params.ticker

  // Validate ticker format
  if (!ticker || !ticker.match(/^[A-Z]{2,4}\.WA$/)) {
    notFound()
  }

  return (
    <div className="space-y-6">
      {/* Stock Info Header */}
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-textPrimary mb-2">
              {ticker}
            </h1>
            <div className="flex items-center space-x-4">
              <Badge variant="outline">GPW</Badge>
              <Badge variant="secondary">Real-time</Badge>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-textPrimary mb-1">
              Loading...
            </div>
            <div className="text-sm text-textSecondary">
              Last update: --
            </div>
          </div>
        </div>
      </Card>

      {/* Main Chart Area */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <div className="space-y-4">
            <TradingChart ticker={ticker} />
            <PatternOverlay ticker={ticker} />
          </div>
        </div>

        <div className="space-y-4">
          <IndicatorPanel ticker={ticker} />
        </div>
      </div>
    </div>
  )
}