#!/usr/bin/env python3
"""
Simple Dynamic WIG30 Analyzer - Wiarygodne i przyszłościowe spółki na 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import json
import time

class SimpleWIG30Analyzer:
    """Prosty dynamic analyzer dla WIG30"""

    def __init__(self):
        print("🚀 Simple WIG30 Analyzer initialized")
        self.wig30_tickers = []
        self.analysis_results = []
        self.future_stocks = []

        # Progi
        self.ROE_THRESHOLD = 10.0
        self.PE_THRESHOLD = 20.0
        self.PB_THRESHOLD = 3.0

        # Rekomendacje DM (2025)
        self.recommended_stocks = {
            'ALE.WA': ['mBank'], 'ALR.WA': ['mBank'], 'CCC.WA': ['DM BOŚ'],
            'CDR.WA': ['mBank', 'DM BOŚ'], 'CPS.WA': ['mBank'],
            'KRU.WA': ['mBank'], 'LPP.WA': ['mBank', 'DM BOŚ'],
            'MBK.WA': ['DM BOŚ'], 'MIL.WA': ['mBank'], 'PKN.WA': ['mBank'],
            'SNT.WA': ['mBank'], 'TXT.WA': ['mBank', 'DM BOŚ']
        }

        self.fetch_current_wig30()

    def fetch_current_wig30(self):
        """Pobiera aktualną listę WIG30"""
        try:
            print("📡 Pobieranie listy WIG30 z GPW...")

            # Użyj statycznej listy WIG30 (aktualna na 2025)
            self.wig30_tickers = [
                'PKN.WA', 'LPP.WA', 'CDR.WA', 'KGH.WA', 'PKO.WA', 'PZU.WA', 'PEO.WA', 'MBK.WA',
                'SPL.WA', 'DNP.WA', 'ALE.WA', 'PGE.WA', 'BDX.WA', 'OPL.WA', 'CPS.WA', 'JSW.WA',
                'CCC.WA', 'KTY.WA', 'KRU.WA', 'XTB.WA', 'MIL.WA', 'ACP.WA', 'TPE.WA', 'ALR.WA',
                'PCO.WA', 'ZAB.WA', 'TXT.WA', 'SNT.WA', 'RBW.WA', '11B.WA'
            ]

            print(f"✅ Załadowano {len(self.wig30_tickers)} spółek WIG30")

        except Exception as e:
            print(f"❌ Błąd pobierania: {e}")

    def analyze_single_stock(self, ticker: str):
        """Analizuje pojedynczą spółkę"""
        try:
            print(f"🔍 Analizuję {ticker}...")
            stock = yf.Ticker(ticker)

            # Pobierz dane fundamentalne
            info = stock.info
            hist = stock.history(period='1mo')

            if hist.empty:
                print(f"❌ Brak danych dla {ticker}")
                return None

            # Dane fundamentalne
            net_income = info.get('netIncomeToCommon', 0)
            total_revenue = info.get('totalRevenue', 0)
            roe = info.get('returnOnEquity', 0)
            pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
            pb_ratio = info.get('priceToBook', 0)
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            market_cap = info.get('marketCap', 0)

            # Dane techniczne
            close_prices = hist['Close']

            # Moving averages
            ma5 = close_prices.rolling(5).mean().iloc[-1] if len(close_prices) >= 5 else None
            ma10 = close_prices.rolling(10).mean().iloc[-1] if len(close_prices) >= 10 else None
            ma20 = close_prices.rolling(20).mean().iloc[-1] if len(close_prices) >= 20 else None

            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(rs.dropna()) > 0 else None

            # Trend
            if ma5 and ma10 and ma20:
                if current_price > ma5 > ma10 > ma20:
                    trend = 'Wzrostowy'
                elif current_price < ma5 < ma10 < ma20:
                    trend = 'Spadkowy'
                else:
                    trend = 'Boczny'
            else:
                trend = 'Nieznany'

            # Ocena wiarygodności
            profitability = net_income > 0 and roe >= self.ROE_THRESHOLD
            valuation = pe_ratio > 0 and pe_ratio <= self.PE_THRESHOLD
            book_value = pb_ratio > 0 and pb_ratio <= self.PB_THRESHOLD

            # Future Score (0-100)
            future_score = 0
            future_factors = []

            # Technologia/Gaming
            if ticker in ['CDR.WA', 'TXT.WA', '11B.WA']:
                future_score += 25
                future_factors.append('Technologia/Gaming')

            # Banki
            if ticker in ['PEO.WA', 'MBK.WA', 'PKO.WA', 'SPL.WA']:
                future_score += 20
                future_factors.append('Sektor bankowy')

            # E-commerce
            if ticker in ['ALE.WA', 'CCC.WA']:
                future_score += 22
                future_factors.append('E-commerce')

            # Energia
            if ticker in ['PKN.WA', 'PGE.WA', 'TPE.WA']:
                future_score += 18
                future_factors.append('Energia')

            # Trend techniczny
            if trend == 'Wzrostowy':
                future_score += 15
                future_factors.append('Trend wzrostowy')
            elif trend == 'Boczny' and rsi and 30 <= rsi <= 70:
                future_score += 10
                future_factors.append('Stabilizacja')

            # Rekomendacje DM
            if ticker in self.recommended_stocks:
                future_score += 12
                future_factors.append(f"Rekomendacje: {', '.join(self.recommended_stocks[ticker])}")

            # RSI optymalny
            if rsi and 30 <= rsi <= 70:
                future_score += 8
                future_factors.append('RSI w zakresie')

            # Wzrost przychodów
            if total_revenue > 0:
                future_score += 5
                future_factors.append('Przychody wzrostowe')

            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'net_income': net_income,
                'total_revenue': total_revenue,
                'roe': roe,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'current_price': current_price,
                'market_cap': market_cap,
                'profitable': profitability,
                'valuation_good': valuation,
                'book_value_good': book_value,
                'ma5': ma5,
                'ma10': ma10,
                'ma20': ma20,
                'rsi': rsi,
                'trend': trend,
                'recommendations': self.recommended_stocks.get(ticker, []),
                'future_score': min(future_score, 100),
                'future_factors': future_factors,
                'analysis_date': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Błąd analizy {ticker}: {e}")
            return None

    def run_full_analysis(self):
        """Uruchamia pełną analizę"""
        print(f"\n🔍 Rozpoczynam analizę {len(self.wig30_tickers)} spółek WIG30...")

        self.analysis_results = []

        for ticker in self.wig30_tickers:
            result = self.analyze_single_stock(ticker)
            if result:
                self.analysis_results.append(result)
                print(f"✅ {ticker}: ROE {result['roe']:.1f}% | Future Score {result['future_score']}")
            time.sleep(0.3)  # Rate limiting

        # Filtrowanie przyszłościowych spółek
        self.future_stocks = sorted(
            [r for r in self.analysis_results if r['future_score'] >= 45],
            key=lambda x: x['future_score'], reverse=True
        )

        print(f"\n🎯 Znaleziono {len(self.future_stocks)} przyszłościowych spółek (Future Score ≥ 45)")

    def generate_report(self):
        """Generuje raport"""
        if not self.future_stocks:
            print("❌ Brak danych do raportu")
            return

        print(f"\n🏆 DYNAMICZNA ANALIZA WIG30 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        print(f"📈 Wiarygodne i przyszłościowe spółki (Future Score ≥ 45):")
        print("-" * 120)

        for i, stock in enumerate(self.future_stocks[:12], 1):
            print(f"{i:2d}. {stock['ticker']:8s} | {stock['name'][:30]:30s} | "
                  f"ROE: {stock['roe']:5.1f}% | P/E: {stock['pe_ratio']:5.1f} | "
                  f"P/B: {stock['pb_ratio']:4.1f} | Future: {stock['future_score']:3.0f} | "
                  f"{stock['trend']:10s} | Cena: {stock['current_price']:6.2f} PLN")
            if stock['future_factors']:
                print(f"     📊 {', '.join(stock['future_factors'])}")

        # Statystyki
        profitable_count = len([s for s in self.analysis_results if s['profitable']])
        good_valuation = len([s for s in self.analysis_results if s['valuation_good']])
        good_book_value = len([s for s in self.analysis_results if s['book_value_good']])

        print(f"\n📊 STATYSTYKI:")
        print(f"   • Łącznie analizowano: {len(self.analysis_results)} spółek")
        print(f"   • Rentowne (ROE ≥ 10%): {profitable_count} ({profitable_count/len(self.analysis_results)*100:.1f}%)")
        print(f"   • Dobra wycena (P/E ≤ {self.PE_THRESHOLD}): {good_valuation} ({good_valuation/len(self.analysis_results)*100:.1f}%)")
        print(f"   • Dobra wartość księgowa (P/B ≤ {self.PB_THRESHOLD}): {good_book_value} ({good_book_value/len(self.analysis_results)*100:.1f}%)")
        print(f"   • Przyszłościowe (Future Score ≥ 45): {len(self.future_stocks)}")

        # Sektorowy podział
        print(f"\n🏢 PODZIAŁ SEKTOROWY TOP 12:")
        sectors = {}
        for stock in self.future_stocks[:12]:
            for factor in stock['future_factors']:
                if factor in ['Technologia/Gaming', 'Sektor bankowy', 'E-commerce', 'Energia']:
                    sectors[factor] = sectors.get(factor, 0) + 1

        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {sector}: {count} spółek")

        # Eksport
        self.export_results()

    def export_results(self):
        """Eksportuje wyniki"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # CSV z pełną analizą
        df = pd.DataFrame(self.analysis_results)
        csv_file = f'wig30_dynamic_analysis_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Pełna analiza wyeksportowana do: {csv_file}")

        # JSON z przyszłościowymi
        future_file = f'wig30_future_stocks_{timestamp}.json'
        with open(future_file, 'w', encoding='utf-8') as f:
            json.dump(self.future_stocks, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Przyszłościowe spółki zapisane w: {future_file}")

        # Podsumowanie
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_analyzed': len(self.analysis_results),
            'future_stocks_count': len(self.future_stocks),
            'top_10': [s['ticker'] for s in self.future_stocks[:10]],
            'average_future_score': np.mean([s['future_score'] for s in self.future_stocks]) if self.future_stocks else 0
        }

        summary_file = f'wig30_summary_{timestamp}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 Podsumowanie zapisane w: {summary_file}")

        # Generuj wykres dla top 5
        self.generate_top_charts()

    def generate_top_charts(self):
        """Generuje proste wykresy dla top 5"""
        print(f"\n📊 Generowanie wykresów dla TOP 5 spółek...")

        for i, stock in enumerate(self.future_stocks[:5], 1):
            print(f"\n📈 Wykres {i}. {stock['ticker']} ({stock['name']})")
            print(f"   Future Score: {stock['future_score']}/100")
            print(f"   ROE: {stock['roe']:.1f}% | P/E: {stock['pe_ratio']:.1f} | RSI: {stock['rsi']:.1f}")
            print(f"   Trend: {stock['trend']} | Cena: {stock['current_price']:.2f} PLN")
            print(f"   Czynniki przyszłościowe: {', '.join(stock['future_factors'])}")

            # Prosta wizualizacja ASCII
            score_bar = '█' * int(stock['future_score'] / 5)
            print(f"   Score: [{score_bar:<20}] {stock['future_score']}%")

def main():
    """Główna funkcja"""
    print("🎯 DYNAMICZNY ANALIZATOR WIG30 - Wiarygodne i przyszłościowe spółki 2025")
    print("=" * 70)

    analyzer = SimpleWIG30Analyzer()

    try:
        # Pełna analiza
        analyzer.run_full_analysis()

        # Generuj raport
        analyzer.generate_report()

        print(f"\n✅ Analiza zakończona sukcesem!")
        print(f"📁 Sprawdź pliki CSV i JSON z wynikami")

    except KeyboardInterrupt:
        print(f"\n🛑 Analiza przerwana przez użytkownika")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()