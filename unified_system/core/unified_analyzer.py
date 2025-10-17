# -*- coding: utf-8 -*-
"""
Zunifikowany Analyzer WIG30 - Łączy funkcjonalności wig30_bot, simple_wig30_analyzer i dynamic_wig30_analyzer
Kompleksowy system analizy fundamentalnej, technicznej i future scoring
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from typing import Dict, List, Optional, Tuple

# Importy z zunifikowanego systemu
from ..config.settings import config

class UnifiedWIG30Analyzer:
    """Zunifikowany analyzer WIG30 z pełnym zakresem analizy"""

    def __init__(self):
        """Inicjalizacja analyzer"""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        print("🚀 Zunifikowany WIG30 Analyzer v2.0 initialized")
        print(f"📊 Aktywny indeks: {config.active_index}")
        print(f"🎯 Progi: ROE ≥ {config.fundamental['roe_min']}%, P/E ≤ {config.fundamental['pe_max']}, P/B ≤ {config.fundamental['pb_max']}")

        self.tickers = config.get_active_tickers()
        self.analysis_results = []
        self.future_stocks = []
        self.analysis_start_time = datetime.now()

        # Walidacja konfiguracji
        config.validate_config()

    def setup_logging(self):
        """Konfiguracja logowania"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, config.logging['level']),
            format=config.logging['format'],
            handlers=[
                logging.FileHandler(config.logging['file']),
                logging.StreamHandler()
            ]
        )

    def analyze_single_stock(self, ticker: str) -> Optional[Dict]:
        """
        Kompleksowa analiza pojedynczej spółki
        Łączy analizę fundamentalną, techniczną i future scoring
        """
        try:
            print(f"🔍 Analizuję {ticker}...")

            # Pobierz dane z Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='1mo')

            if hist.empty:
                print(f"❌ Brak danych historycznych dla {ticker}")
                return None

            # === ANALIZA FUNDAMENTALNA ===
            fundamental_data = self._analyze_fundamentals(ticker, info)

            # === ANALIZA TECHNICZNA ===
            technical_data = self._analyze_technicals(hist)

            # === FUTURE SCORE ANALYSIS ===
            future_data = self._calculate_future_score(ticker, fundamental_data, technical_data)

            # === OCENA WIARYGODNOŚCI ===
            reliability_score = self._calculate_reliability_score(fundamental_data, technical_data)

            # Połączenie wszystkich wyników
            result = {
                # Podstawowe dane
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'current_price': info.get('currentPrice', hist['Close'].iloc[-1]),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'PLN'),
                'analysis_date': datetime.now().isoformat(),

                # Dane fundamentalne
                'fundamental': fundamental_data,

                # Dane techniczne
                'technical': technical_data,

                # Future Score
                'future_score': future_data['score'],
                'future_factors': future_data['factors'],
                'sector': future_data['sector'],

                # Ocena wiarygodności
                'reliability_score': reliability_score,
                'recommendation': self._get_recommendation(fundamental_data, technical_data, future_data),

                # Dane historyczne dla wykresów
                'historical_prices': {str(k): v for k, v in hist['Close'].to_dict().items()} if not hist.empty else {}
            }

            print(f"✅ {ticker}: ROE {fundamental_data.get('roe', 0):.1f}% | "
                  f"Future Score {future_data['score']} | "
                  f"Reliability {reliability_score:.1f}")

            return result

        except Exception as e:
            self.logger.error(f"Błąd analizy {ticker}: {e}")
            print(f"❌ Błąd analizy {ticker}: {str(e)[:50]}...")
            return None

    def _analyze_fundamentals(self, ticker: str, info: Dict) -> Dict:
        """Analiza fundamentalna spółki"""

        # Kluczowe wskaźniki
        net_income = info.get('netIncomeToCommon', 0)
        total_revenue = info.get('totalRevenue', 0)
        roe = info.get('returnOnEquity', 0)
        pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
        pb_ratio = info.get('priceToBook', 0)
        debt_to_equity = info.get('debtToEquity', 0)
        current_ratio = info.get('currentRatio', 0)

        # Wskaźniki rentowności
        gross_margin = info.get('grossMargins', 0)
        operating_margin = info.get('operatingMargins', 0)
        net_margin = info.get('profitMargins', 0)

        # Wskaźniki efektywności
        roa = info.get('returnOnAssets', 0)
        roic = info.get('returnOnCapital', 0)

        # Ocena jakości danych
        data_quality = self._assess_data_quality(info)

        return {
            'net_income': net_income,
            'total_revenue': total_revenue,
            'roe': roe,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'roa': roa,
            'roic': roic,
            'data_quality': data_quality,
            'profitable': net_income > 0,
            'meets_roe_criteria': roe >= config.fundamental['roe_min'],
            'meets_pe_criteria': 0 < pe_ratio <= config.fundamental['pe_max'],
            'meets_pb_criteria': 0 < pb_ratio <= config.fundamental['pb_max'],
            'meets_debt_criteria': debt_to_equity <= config.fundamental['debt_to_equity_max'],
            'meets_liquidity_criteria': current_ratio >= config.fundamental['current_ratio_min']
        }

    def _analyze_technicals(self, hist: pd.DataFrame) -> Dict:
        """Analiza techniczna na podstawie danych historycznych"""

        if hist.empty:
            return {}

        close_prices = hist['Close']
        volumes = hist['Volume']

        # Moving averages
        ma_data = {}
        for period in config.technical['ma_periods']:
            if len(close_prices) >= period:
                ma_data[f'ma_{period}'] = close_prices.rolling(period).mean().iloc[-1]

        # RSI
        rsi = self._calculate_rsi(close_prices)

        # MACD
        macd_data = self._calculate_macd(close_prices)

        # Bollinger Bands
        bollinger_data = self._calculate_bollinger_bands(close_prices)

        # Trend analysis
        trend_data = self._analyze_trend(close_prices, ma_data)

        # Wolumen analysis
        volume_analysis = self._analyze_volume(volumes)

        return {
            'ma': ma_data,
            'rsi': rsi,
            'macd': macd_data,
            'bollinger_bands': bollinger_data,
            'trend': trend_data,
            'volume': volume_analysis,
            'volatility': self._calculate_volatility(close_prices),
            'price_momentum': self._calculate_momentum(close_prices)
        }

    def _calculate_future_score(self, ticker: str, fundamental: Dict, technical: Dict) -> Dict:
        """Oblicza Future Score (0-100)"""

        score = 0
        factors = []

        # Punkty za sektor
        sector_bonus = config.get_sector_bonus(ticker)
        sector_name = config.get_sector_name(ticker)
        if sector_bonus > 0:
            score += sector_bonus
            factors.append(sector_name)

        # Punkty za trend techniczny
        trend = technical.get('trend', {}).get('direction', 'unknown')
        if trend == 'uptrend':
            score += config.future_score['trend_bonus']['uptrend']
            factors.append('Trend wzrostowy')
        elif trend == 'sideways':
            rsi = technical.get('rsi', {})
            if rsi and config.technical['rsi_oversold'] <= rsi <= config.technical['rsi_overbought']:
                score += config.future_score['trend_bonus']['sideways']
                factors.append('Stabilizacja')

        # Punkty za rekomendacje DM
        if ticker in config.recommendations:
            score += config.future_score['recommendation_bonus']
            factors.append(f"Rekomendacje: {', '.join(config.recommendations[ticker])}")

        # Punkty za RSI w optymalnym zakresie
        rsi = technical.get('rsi')
        if rsi and config.technical['rsi_oversold'] <= rsi <= config.technical['rsi_overbought']:
            score += config.future_score['rsi_bonus']
            factors.append('RSI w zakresie')

        # Punkty za przychody
        if fundamental.get('total_revenue', 0) > 0:
            score += config.future_score['revenue_bonus']
            factors.append('Przychody wzrostowe')

        # Bonus za jakość fundamentalną
        if fundamental.get('meets_roe_criteria'):
            score += 10
            factors.append('Wysoka rentowność')

        if fundamental.get('meets_pe_criteria') and fundamental.get('meets_pb_criteria'):
            score += 8
            factors.append('Atrakcyjna wycena')

        return {
            'score': min(score, 100),
            'factors': factors,
            'sector': sector_name,
            'meets_threshold': score >= config.future_score['min_score_threshold']
        }

    def _calculate_reliability_score(self, fundamental: Dict, technical: Dict) -> float:
        """Oblicza ocenę wiarygodności spółki (0-100)"""

        score = 0

        # Jakość danych fundamentalnych (0-30)
        data_quality = fundamental.get('data_quality', {})
        if data_quality.get('completeness', 0) > 0.7:
            score += 15
        if data_quality.get('freshness', 0) > 0.8:
            score += 15

        # Stabilność finansowa (0-25)
        if fundamental.get('meets_debt_criteria'):
            score += 10
        if fundamental.get('meets_liquidity_criteria'):
            score += 15

        # Rentowność (0-25)
        if fundamental.get('meets_roe_criteria'):
            score += 15
        if fundamental.get('profitable'):
            score += 10

        # Wskaźniki techniczne (0-20)
        if technical.get('trend', {}).get('strength', 0) > 0.6:
            score += 10
        if technical.get('rsi') and 30 <= technical['rsi'] <= 70:
            score += 10

        return min(score, 100)

    def _get_recommendation(self, fundamental: Dict, technical: Dict, future: Dict) -> str:
        """Generuje rekomendację inwestycyjną"""

        future_score = future['score']
        reliability = self._calculate_reliability_score(fundamental, technical)

        if future_score >= 70 and reliability >= 70:
            return "STRONG_BUY"
        elif future_score >= 60 and reliability >= 60:
            return "BUY"
        elif future_score >= 50 and reliability >= 50:
            return "HOLD"
        elif future_score >= 40:
            return "WEAK_HOLD"
        else:
            return "SELL"

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Oblicza RSI"""
        if len(prices) < period + 1:
            return None

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Oblicza MACD"""
        if len(prices) < 26:
            return {}

        ema_fast = prices.ewm(span=config.technical['macd_fast']).mean()
        ema_slow = prices.ewm(span=config.technical['macd_slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=config.technical['macd_signal']).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }

    def _calculate_bollinger_bands(self, prices: pd.Series) -> Dict:
        """Oblicza Bollinger Bands"""
        if len(prices) < config.technical['bollinger_period']:
            return {}

        sma = prices.rolling(config.technical['bollinger_period']).mean()
        std = prices.rolling(config.technical['bollinger_period']).std()

        return {
            'upper': (sma + std * config.technical['bollinger_std']).iloc[-1],
            'middle': sma.iloc[-1],
            'lower': (sma - std * config.technical['bollinger_std']).iloc[-1]
        }

    def _analyze_trend(self, prices: pd.Series, ma_data: Dict) -> Dict:
        """Analiza trendu"""
        current_price = prices.iloc[-1]

        # Check if we have enough MAs for trend analysis
        if len(ma_data) < 3:
            return {'direction': 'unknown', 'strength': 0}

        ma5 = ma_data.get('ma_5')
        ma10 = ma_data.get('ma_10')
        ma20 = ma_data.get('ma_20')

        if not all([ma5, ma10, ma20]):
            return {'direction': 'unknown', 'strength': 0}

        # Trend determination
        if current_price > ma5 > ma10 > ma20:
            direction = 'uptrend'
            strength = 0.8
        elif current_price < ma5 < ma10 < ma20:
            direction = 'downtrend'
            strength = 0.8
        else:
            direction = 'sideways'
            strength = 0.5

        return {
            'direction': direction,
            'strength': strength,
            'current_vs_ma20': (current_price - ma20) / ma20 * 100
        }

    def _analyze_volume(self, volumes: pd.Series) -> Dict:
        """Analiza wolumenu"""
        if volumes.empty:
            return {}

        recent_volume = volumes.iloc[-5:].mean()
        avg_volume = volumes.mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        return {
            'recent_avg': recent_volume,
            'overall_avg': avg_volume,
            'ratio': volume_ratio,
            'trend': 'increasing' if volume_ratio > 1.2 else 'normal' if volume_ratio > 0.8 else 'decreasing'
        }

    def _calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Oblicza zmienność"""
        if len(prices) < period:
            return 0

        returns = prices.pct_change().dropna()
        volatility = returns.rolling(period).std().iloc[-1] * np.sqrt(252)  # Annualized

        return volatility if not pd.isna(volatility) else 0

    def _calculate_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """Oblicza momentum"""
        if len(prices) < period + 1:
            return 0

        momentum = (prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1] * 100
        return momentum if not pd.isna(momentum) else 0

    def _assess_data_quality(self, info: Dict) -> Dict:
        """Ocena jakości danych fundamentalnych"""

        required_fields = ['netIncomeToCommon', 'totalRevenue', 'returnOnEquity', 'forwardPE', 'priceToBook']
        available_fields = sum(1 for field in required_fields if info.get(field) is not None)

        completeness = available_fields / len(required_fields)

        # Check if data is recent (assuming recent if market cap is available)
        freshness = 1.0 if info.get('marketCap', 0) > 0 else 0.5

        return {
            'completeness': completeness,
            'freshness': freshness,
            'overall_score': (completeness + freshness) / 2
        }

    def run_full_analysis(self) -> None:
        """Uruchamia pełną analizę wszystkich spółek"""

        print(f"\n🔍 Rozpoczynam kompleksową analizę {len(self.tickers)} spółek {config.active_index}...")
        print(f"⏱️  Szacowany czas: {len(self.tickers) * config.data_sources['yahoo_finance']['rate_limit']:.1f} sekund")
        print("=" * 80)

        self.analysis_results = []

        for i, ticker in enumerate(self.tickers):
            result = self.analyze_single_stock(ticker)
            if result:
                self.analysis_results.append(result)

            # Rate limiting
            if i < len(self.tickers) - 1:
                time.sleep(config.data_sources['yahoo_finance']['rate_limit'])

        # Filtrowanie przyszłościowych spółek
        self.future_stocks = [
            stock for stock in self.analysis_results
            if stock['future_score'] >= config.future_score['min_score_threshold']
        ]

        # Sortowanie po Future Score
        self.future_stocks = sorted(self.future_stocks, key=lambda x: x['future_score'], reverse=True)

        analysis_time = datetime.now() - self.analysis_start_time
        print(f"\n🎯 Analiza zakończona w {analysis_time.total_seconds():.1f} sekund")
        print(f"✅ Przeanalizowano: {len(self.analysis_results)} spółek")
        print(f"🚀 Przyszłościowych (Future Score ≥ {config.future_score['min_score_threshold']}): {len(self.future_stocks)} spółek")

    def generate_comprehensive_report(self) -> None:
        """Generuje kompletny raport analizy"""

        if not self.analysis_results:
            print("❌ Brak wyników analizy")
            return

        print(f"\n🏆 KOMPLEKSOWY RAPORT ANALIZY {config.active_index} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)

        # Statystyki ogólne
        self._print_executive_summary()

        # Top przyszłościowe spółki
        if self.future_stocks:
            self._print_future_stocks()

        # Analiza sektorowa
        self._print_sector_analysis()

        # Rekomendacje
        self._print_recommendations()

        # Eksport wyników
        self._export_results()

    def _print_executive_summary(self) -> None:
        """Drukuje podsumowanie wykonawcze"""

        total_stocks = len(self.analysis_results)
        profitable_count = sum(1 for s in self.analysis_results if s['fundamental']['profitable'])
        reliable_count = sum(1 for s in self.analysis_results if s['reliability_score'] >= 60)
        future_count = len(self.future_stocks)

        print(f"\n📊 PODSUMOWANIE WYKONAWCZE:")
        print(f"   • Łącznie spółek: {total_stocks}")
        print(f"   • Rentowne: {profitable_count} ({profitable_count/total_stocks*100:.1f}%)")
        print(f"   • Wiarygodne (score ≥ 60): {reliable_count} ({reliable_count/total_stocks*100:.1f}%)")
        print(f"   • Przyszłościowe (Future Score ≥ {config.future_score['min_score_threshold']}): {future_count} ({future_count/total_stocks*100:.1f}%)")

        if self.analysis_results:
            avg_roe = np.mean([s['fundamental']['roe'] for s in self.analysis_results if s['fundamental']['roe'] > 0])
            avg_pe = np.mean([s['fundamental']['pe_ratio'] for s in self.analysis_results if s['fundamental']['pe_ratio'] > 0])
            avg_future_score = np.mean([s['future_score'] for s in self.analysis_results])

            print(f"\n📈 ŚREDNIE WSKAŹNIKI:")
            print(f"   • Średnie ROE: {avg_roe:.1f}%")
            print(f"   • Średnie P/E: {avg_pe:.1f}")
            print(f"   • Średni Future Score: {avg_future_score:.1f}")

    def _print_future_stocks(self) -> None:
        """Drukuje przyszłościowe spółki"""

        print(f"\n🚀 TOP PRZYSZŁOŚCIOWYCH SPÓŁEK (Future Score):")
        print("-" * 120)
        print(f"{'Lp.':>3} {'Ticker':<8} | {'Nazwa':<25} | {'ROE':>6} | {'P/E':>6} | {'P/B':>6} | {'Future':>7} | {'Reliab':>7} | {'Rekomendacja'}")
        print("-" * 120)

        for i, stock in enumerate(self.future_stocks[:12], 1):
            ticker = stock['ticker']
            name = stock['name'][:24]
            roe = stock['fundamental']['roe']
            pe = stock['fundamental']['pe_ratio']
            pb = stock['fundamental']['pb_ratio']
            future_score = stock['future_score']
            reliability = stock['reliability_score']
            recommendation = stock['recommendation']

            print(f"{i:>3}. {ticker:<8} | {name:<25} | {roe:>6.1f} | {pe:>6.1f} | {pb:>6.2f} | "
                  f"{future_score:>7.0f} | {reliability:>7.0f} | {recommendation}")

            if stock['future_factors']:
                print(f"     📊 {', '.join(stock['future_factors'])}")

    def _print_sector_analysis(self) -> None:
        """Drukuje analizę sektorową"""

        print(f"\n🏢 ANALIZA SEKTOROWA:")

        sector_stats = {}
        for stock in self.future_stocks:
            sector = stock.get('sector', 'Inne')
            if sector not in sector_stats:
                sector_stats[sector] = {'count': 0, 'avg_future_score': 0, 'avg_reliability': 0}

            sector_stats[sector]['count'] += 1
            sector_stats[sector]['avg_future_score'] += stock['future_score']
            sector_stats[sector]['avg_reliability'] += stock['reliability_score']

        for sector, stats in sector_stats.items():
            if stats['count'] > 0:
                stats['avg_future_score'] /= stats['count']
                stats['avg_reliability'] /= stats['count']
                print(f"   • {sector}: {stats['count']} spółek | "
                      f"Śr. Future Score: {stats['avg_future_score']:.1f} | "
                      f"Śr. Reliability: {stats['avg_reliability']:.1f}")

    def _print_recommendations(self) -> None:
        """Drukuje rekomendacje inwestycyjne"""

        print(f"\n🎯 REKOMENDACJE INWESTYCYJNE:")

        buy_signals = [s for s in self.analysis_results if s['recommendation'] in ['STRONG_BUY', 'BUY']]
        hold_signals = [s for s in self.analysis_results if s['recommendation'] == 'HOLD']
        sell_signals = [s for s in self.analysis_results if s['recommendation'] in ['WEAK_HOLD', 'SELL']]

        print(f"   • KUP ({len(buy_signals)}): {[s['ticker'] for s in buy_signals[:5]]}")
        print(f"   • TRZYMAJ ({len(hold_signals)}): {[s['ticker'] for s in hold_signals[:5]]}")
        print(f"   • SPRZEDAJ ({len(sell_signals)}): {[s['ticker'] for s in sell_signals[:5]]}")

    def _export_results(self) -> None:
        """Eksportuje wyniki do plików"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Eksport CSV z pełną analizą
        df = pd.DataFrame(self.analysis_results)
        csv_file = f"{config.export['output_dir']}/wig30_unified_analysis_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Pełna analiza wyeksportowana do: {csv_file}")

        # Eksport JSON z przyszłościowymi spółkami
        future_file = f"{config.export['output_dir']}/wig30_future_stocks_{timestamp}.json"
        with open(future_file, 'w', encoding='utf-8') as f:
            json.dump(self.future_stocks, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Przyszłościowe spółki zapisane w: {future_file}")

        # Eksport podsumowania
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'index': config.active_index,
            'total_analyzed': len(self.analysis_results),
            'future_stocks_count': len(self.future_stocks),
            'top_10': [s['ticker'] for s in self.future_stocks[:10]],
            'average_future_score': np.mean([s['future_score'] for s in self.analysis_results]),
            'analysis_duration_seconds': (datetime.now() - self.analysis_start_time).total_seconds(),
            'config': {
                'roe_threshold': config.fundamental['roe_min'],
                'pe_threshold': config.fundamental['pe_max'],
                'pb_threshold': config.fundamental['pb_max']
            }
        }

        summary_file = f"{config.export['output_dir']}/wig30_analysis_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 Podsumowanie zapisane w: {summary_file}")


def main():
    """Główna funkcja"""
    print("🎯 ZUNIFIKOWANY SYSTEM ANALIZY WIG30 v2.0")
    print("=" * 70)

    analyzer = UnifiedWIG30Analyzer()

    try:
        # Pełna analiza
        analyzer.run_full_analysis()

        # Generuj raport
        analyzer.generate_comprehensive_report()

        print(f"\n✅ Analiza zakończona sukcesem!")
        print(f"📁 Sprawdź pliki w folderze: {config.export['output_dir']}")

    except KeyboardInterrupt:
        print(f"\n🛑 Analiza przerwana przez użytkownika")
    except Exception as e:
        print(f"❌ Błąd systemu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()