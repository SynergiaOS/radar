import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from analyzer import analyze_company_fundamentals
from config import (
    WIG30_TICKERS, WIG20_TICKERS, OUTPUT_CSV_FILE, WIG20_OUTPUT_CSV_FILE,
    LEGACY_CSV_FILE, DEPRECATED_PE_CSV_FILE, TOP_N_DISPLAY, ACTIVE_INDEX,
    ROE_THRESHOLD, PE_THRESHOLD, PB_THRESHOLD, ENABLE_DUAL_FILTER,
    ENABLE_TRIPLE_FILTER, ENABLE_PB_FILTER, ENABLE_CORRELATION_ANALYSIS,
    ENABLE_CHART_GENERATION, CHART_MAX_STOCKS, USE_CQG_API
)
import visualization
import advanced_charts
import logging

logger = logging.getLogger(__name__)


def meets_dual_filter_criteria(stock):
    """
    Check if a stock meets ROE and P/E filtering criteria.

    Args:
        stock: Dictionary with stock analysis data

    Returns:
        Boolean indicating if stock meets ROE and P/E criteria
    """
    # Check ROE criteria
    roe_ok = stock.get('roe') is not None and stock['roe'] >= ROE_THRESHOLD

    # Check P/E criteria
    pe_ok = stock.get('pe_ratio') is not None and stock['pe_ratio'] <= PE_THRESHOLD

    return roe_ok and pe_ok


def meets_triple_filter_criteria(stock):
    """
    Check if a stock meets ROE, P/E, and P/B filtering criteria.

    Args:
        stock: Dictionary with stock analysis data

    Returns:
        Boolean indicating if stock meets all enabled criteria
    """
    # Check ROE criteria
    roe_ok = stock.get('roe') is not None and stock['roe'] >= ROE_THRESHOLD

    # Check P/E criteria
    pe_ok = stock.get('pe_ratio') is not None and stock['pe_ratio'] <= PE_THRESHOLD

    # Check P/B criteria if enabled
    if ENABLE_PB_FILTER:
        pb_ok = stock.get('pb_ratio') is not None and stock['pb_ratio'] <= PB_THRESHOLD
    else:
        pb_ok = True  # Skip P/B filtering if disabled

    return roe_ok and pe_ok and pb_ok


def calculate_roe_pb_correlation(stocks):
    """
    Calculate Pearson correlation between ROE and P/B ratios.

    Args:
        stocks: List of stock dictionaries with ROE and P/B data

    Returns:
        Float correlation coefficient or None if insufficient data
    """
    if not ENABLE_CORRELATION_ANALYSIS or not stocks:
        return None

    # Extract aligned ROE and P/B pairs
    roe_values = []
    pb_values = []

    for stock in stocks:
        roe = stock.get('roe')
        pb = stock.get('pb_ratio')

        # Only include valid (non-None, non-NaN) pairs
        if roe is not None and pb is not None and not pd.isna(roe) and not pd.isna(pb):
            roe_values.append(roe)
            pb_values.append(pb)

    # Need at least 3 pairs for meaningful correlation
    if len(roe_values) < 3:
        logger.warning(f"Insufficient data for correlation analysis: {len(roe_values)} pairs (minimum 3 required)")
        return None

    try:
        correlation_matrix = np.corrcoef(roe_values, pb_values)
        correlation = correlation_matrix[0, 1]

        # Handle NaN correlation
        if pd.isna(correlation):
            logger.warning("Correlation calculation resulted in NaN")
            return None

        return float(correlation)

    except Exception as e:
        logger.error(f"Error calculating ROE-P/B correlation: {e}")
        return None


def main():
    """Main function to analyze WIG30/WIG20 companies for profitability, valuation, and technical analysis."""
    # Select tickers and output file based on active index
    tickers = WIG20_TICKERS if ACTIVE_INDEX == 'WIG20' else WIG30_TICKERS
    output_file = WIG20_OUTPUT_CSV_FILE if ACTIVE_INDEX == 'WIG20' else OUTPUT_CSV_FILE

    # Determine strategy description based on enabled filters
    strategy_parts = [f"ROE ≥ {ROE_THRESHOLD}%"]
    strategy_parts.append(f"P/E ≤ {PE_THRESHOLD}")
    if ENABLE_PB_FILTER:
        strategy_parts.append(f"P/B ≤ {PB_THRESHOLD}")

    strategy_description = " + ".join(strategy_parts)

    # Header with CQG API indication
    api_indicator = " z CQG API" if USE_CQG_API else ""

    print(f"🏆 Analiza {ACTIVE_INDEX}{api_indicator} na {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"🎯 Strategia: Value Investing ({strategy_description})")
    print("💡 Cel: Identyfikacja fundamentalnie mocnych i niedowartościowanych spółek")
    if ENABLE_TRIPLE_FILTER:
        print("🔍 Filtracja: Potrójna (ROE + P/E + P/B)")
    else:
        print("🔍 Filtracja: Podwójna (ROE + P/E)")
    print()

    # Analyze all companies
    all_results = []

    print("🔍 PRZEBIEG ANALIZY:")
    for i, ticker in enumerate(tickers):
        print(f"   Analizowanie {ticker}...", end=' ')
        result = analyze_company_fundamentals(ticker)
        if result:
            all_results.append(result)
            print("✅")
        else:
            print("❌")

        # Rate limiting: add delay between tickers (except last one)
        if i < len(tickers) - 1:
            time.sleep(np.random.uniform(0.2, 0.5))  # Random delay between 0.2-0.5 seconds

    print()

    # Filter profitable companies
    profitable_stocks = [stock for stock in all_results if stock['profitable']]

    # Apply filtering based on configuration
    if ENABLE_TRIPLE_FILTER:
        filtered_stocks = [stock for stock in profitable_stocks if meets_triple_filter_criteria(stock)]
        filter_type = "potrójnym filtrującym"
    elif ENABLE_DUAL_FILTER:
        filtered_stocks = [stock for stock in profitable_stocks if meets_dual_filter_criteria(stock)]
        filter_type = "podwójnym filtrującym"
    else:
        filtered_stocks = profitable_stocks
        filter_type = "rentowności"

    # Sort filtered companies by ROE (descending), None values last
    filtered_stocks = sorted(
        filtered_stocks,
        key=lambda x: x.get('roe') or float('-inf'),
        reverse=True
    )

    # Calculate and display ROE-P/B correlation
    correlation = calculate_roe_pb_correlation(filtered_stocks)
    if correlation is not None:
        print(f"📊 Korelacja ROE vs P/B: {correlation:.2f}")
    else:
        print("📊 Korelacja ROE vs P/B: N/A")

    # Print executive summary
    print("📈 PODSUMOWANIE DLA ZARZĄDU:")
    print(f"   • Łącznie przeanalizowano spółek: {len(all_results)}")
    print(f"   • Spółki rentowne (zysk netto > 0): {len(profitable_stocks)} ({len(profitable_stocks)/len(all_results)*100:.1f}%)" if len(all_results) > 0 else "   • Spółki rentowne (zysk netto > 0): 0 (0.0%)")
    print(f"   • Spółki spełniające {filter_type}: {len(filtered_stocks)} ({len(filtered_stocks)/len(all_results)*100:.1f}%)" if len(all_results) > 0 else f"   • Spółki spełniające {filter_type}: 0 (0.0%)")
    print()

    # Print investment recommendations
    if filtered_stocks:
        print("🎯 REKOMENDACJE INWESTYCYJNE:")
        print("📊 Spółki spełniające kryteria Value Investing:")
        print("=" * 130)
        print(f"{'Lp.':>3} {'Ticker':<8} | {'ROE':>6} | {'P/E':>6} | {'P/B':>6} | {'MA5':>7} | {'MA10':>7} | {'MA20':>7} | {'Trend':>8} | {'Zysk Netto':>15} | {'Nazwa Spółki'}")
        print("=" * 130)

        for i, stock in enumerate(filtered_stocks[:TOP_N_DISPLAY], 1):
            ticker = stock['ticker']
            name = stock['name']
            roe = stock.get('roe')
            pe_ratio = stock.get('pe_ratio')
            pb_ratio = stock.get('pb_ratio')
            ma5 = stock.get('ma5')
            ma10 = stock.get('ma10')
            ma20 = stock.get('ma20')
            trend_label = stock.get('trend_label', 'Nieznany')

            # Format values
            roe_str = f"{roe:.1f}%" if roe is not None else "N/A"
            pe_str = f"{pe_ratio:.1f}" if pe_ratio is not None else "N/A"
            pb_str = f"{pb_ratio:.2f}" if pb_ratio is not None else "N/A"
            ma5_str = f"{ma5:.2f}" if ma5 is not None else "N/A"
            ma10_str = f"{ma10:.2f}" if ma10 is not None else "N/A"
            ma20_str = f"{ma20:.2f}" if ma20 is not None else "N/A"
            net_income_str = f"{stock['net_income']:,.0f} PLN"

            # Truncate name if too long
            display_name = name[:25] + '..' if len(name) > 25 else name

            print(f"{i:>3}. {ticker:<8} | {roe_str:>6} | {pe_str:>6} | {pb_str:>6} | {ma5_str:>7} | {ma10_str:>7} | {ma20_str:>7} | {trend_label:>8} | {net_income_str:>15} | {display_name}")

        print()
        print("💡 ANALIZA STRATEGICZNA:")
        print(f"   • Znaleziono {len(filtered_stocks)} spółki o potencjale inwestycyjnym")
        print(f"   • Średnie ROE: {sum(s.get('roe', 0) for s in filtered_stocks)/len(filtered_stocks):.1f}%")
        print(f"   • Średnie P/E: {sum(s.get('pe_ratio', 0) for s in filtered_stocks)/len(filtered_stocks):.1f}")
        if ENABLE_PB_FILTER:
            avg_pb = sum(s.get('pb_ratio', 0) for s in filtered_stocks if s.get('pb_ratio') is not None)
            pb_count = len([s for s in filtered_stocks if s.get('pb_ratio') is not None])
            if pb_count > 0:
                print(f"   • Średnie P/B: {avg_pb/pb_count:.2f}")
        print(f"   • Łączny zysk netto: {sum(s.get('net_income', 0) for s in filtered_stocks):,.0f} PLN")

        # Add strategic insights
        if len(filtered_stocks) >= 1:
            best_roe = max(filtered_stocks, key=lambda x: x.get('roe', 0))
            print(f"\n🏆 NAJLEPSZA SPÓŁKA Z NAJWYŻSZYM ROE:")
            print(f"   • {best_roe['ticker']} ({best_roe['name'][:30]}) - ROE: {best_roe.get('roe', 0):.1f}%")

        if len(filtered_stocks) >= 1:
            lowest_pe = min(filtered_stocks, key=lambda x: x.get('pe_ratio', float('inf')))
            print(f"💰 NAJTANIEJSZA WYCENA (NAJNIŻSZE P/E):")
            print(f"   • {lowest_pe['ticker']} ({lowest_pe['name'][:30]}) - P/E: {lowest_pe.get('pe_ratio', 0):.1f}")

        if ENABLE_PB_FILTER and len(filtered_stocks) >= 1:
            valid_pb_stocks = [s for s in filtered_stocks if s.get('pb_ratio') is not None]
            if valid_pb_stocks:
                lowest_pb = min(valid_pb_stocks, key=lambda x: x.get('pb_ratio', float('inf')))
                print(f"📚 NAJNIŻSZY P/B:")
                print(f"   • {lowest_pb['ticker']} ({lowest_pb['name'][:30]}) - P/B: {lowest_pb.get('pb_ratio', 0):.2f}")

        # Generate charts if enabled
        if ENABLE_CHART_GENERATION:
            print(f"\n📊 GENEROWANIE WYKRESÓW:")

            # Generate basic charts
            chart_files = visualization.generate_charts_for_stocks(filtered_stocks, max_charts=CHART_MAX_STOCKS)

            # Generate professional advanced charts
            professional_chart_files = advanced_charts.generate_charts_for_recommendations(filtered_stocks, max_stocks=CHART_MAX_STOCKS)

            total_charts = len(chart_files) + len(professional_chart_files)

            if total_charts > 0:
                print(f"   ✅ Wygenerowano {total_charts} wykresów Chart.js")
                print(f"   📁 Katalog podstawowy: {visualization.CHART_OUTPUT_DIR}")
                print(f"   📁 Katalog profesjonalny: professional_charts")

                # Generate sample professional chart display
                if filtered_stocks and filtered_stocks[0].get('historical_prices') is not None:
                    first_stock = filtered_stocks[0]

                    # Basic chart for console
                    basic_chart = visualization.generate_price_ma_chart(
                        first_stock['ticker'],
                        first_stock['name'],
                        first_stock['historical_prices'],
                        first_stock.get('ma5'),
                        first_stock.get('ma10'),
                        first_stock.get('ma20'),
                        first_stock.get('trend_label', 'Nieznany')
                    )

                    # Professional candlestick chart for console
                    professional_chart = advanced_charts.generate_candlestick_chart(
                        first_stock['ticker'],
                        first_stock['name'],
                        first_stock['historical_prices'],
                        first_stock.get('ma5'),
                        first_stock.get('ma10'),
                        first_stock.get('ma20'),
                        first_stock.get('trend_label', 'Nieznany')
                    )

                    if professional_chart:
                        print(f"\n🎯 PROFESJONALNY WYKRES ŚWIECZKOWY DLA {first_stock['ticker']}:")
                        print("=" * 80)
                        print(json.dumps(professional_chart, indent=2, ensure_ascii=False))
                        print("=" * 80)

                # Print charts summary
                summary = visualization.create_summary_charts_summary(filtered_stocks)
                visualization.print_charts_summary(summary)

                # Professional charts summary
                print(f"\n🏆 PROFESJONALNE WYKRESY:")
                print(f"   • Wykresy świeczkowe z MA i Bollinger Bands")
                print(f"   • Wskaźniki RSI (overbought/oversold)")
                print(f"   • MACD z histogramem")
                print(f"   • Analiza wolumenu z MA")
                print(f"   • Stylizacja TradingView")

                # Export summary
                visualization.export_charts_summary_to_json(summary)
            else:
                print(f"   ⚠️  Nie udało się wygenerować wykresów (brak danych historycznych)")
    else:
        print("⚠️  BRAK REKOMENDACJI INWESTYCYJNYCH")
        if ENABLE_TRIPLE_FILTER:
            criteria_desc = f"ROE ≥ {ROE_THRESHOLD}%, P/E ≤ {PE_THRESHOLD}"
            if ENABLE_PB_FILTER:
                criteria_desc += f", P/B ≤ {PB_THRESHOLD}"
            print(f"   • Żadna spółka nie spełnia kryteriów {criteria_desc}")
        elif ENABLE_DUAL_FILTER:
            print(f"   • Żadna spółka nie spełnia kryteriów ROE ≥ {ROE_THRESHOLD}% i P/E ≤ {PE_THRESHOLD}")
        else:
            print("   • Brak spółek rentownych w ostatnim kwartale")
        print("   • Zalecamy rozważenie mniej restrykcyjnych kryteriów analizy")

        print("\n💡 REKOMENDACJE STRATEGICZNE:")
        print("   • Rozważ rozszerzenie kryteriów (ROE ≥ 8% lub P/E ≤ 20)")
        if ENABLE_PB_FILTER:
            print(f"   • Rozważ wyższy próg P/B (np. {PB_THRESHOLD * 1.5:.1f})")
        print("   • Analizuj spółki z innych sektorów GPW")
        print("   • Monitoruj zmiany kwartalne w wynikach finansowych")

    print()

    # Export data for further analysis
    if all_results:
        df = pd.DataFrame(all_results)

        # Sort by ROE (descending) for CSV export to match console output
        df_sorted = df.sort_values('roe', ascending=False, na_position='last')

        print("📁 EKSPORTOWANIE DANYCH:")

        # Export filtered results (main recommendations)
        if filtered_stocks:
            filtered_df = pd.DataFrame(filtered_stocks)
            filtered_df.to_csv(output_file, index=False)
            print(f"   ✅ Rekomendacje inwestycyjne: '{output_file}'")

            # Export only filtered companies (for easy reference)
            filtered_only_filename = output_file.replace('.csv', '_rekomendacje.csv')
            filtered_df.to_csv(filtered_only_filename, index=False)
            print(f"   ✅ Plik rekomendacji: '{filtered_only_filename}'")

        # Export all profitable companies (before P/E filtering)
        if profitable_stocks:
            all_profitable_df = pd.DataFrame(profitable_stocks)
            all_profitable_filename = output_file.replace('.csv', '_wszystkie_rentowne.csv')
            all_profitable_df.to_csv(all_profitable_filename, index=False)
            print(f"   ✅ Wszystkie spółki rentowne: '{all_profitable_filename}'")

        # Export ROE-sorted complete dataset (primary export)
        df_sorted.to_csv(output_file, index=False)
        print(f"   ✅ Kompletna analiza ROE: '{output_file}'")

        # Export ROE-sorted data to legacy file for backward compatibility
        df_sorted.to_csv(LEGACY_CSV_FILE, index=False)
        print(f"   ✅ Kompletny zbiór danych (legacy): '{LEGACY_CSV_FILE}'")

        # Export deprecated filename for compatibility during transition
        if ENABLE_DUAL_FILTER:
            deprecated_filename = DEPRECATED_PE_CSV_FILE if ACTIVE_INDEX == 'WIG30' else 'wig20_analysis_pe_threshold.csv'
            df_sorted.to_csv(deprecated_filename, index=False)
            print(f"   ✅ Kompatybilny eksport: '{deprecated_filename}'")

        print(f"\n📊 PODSUMOWANIE EKSPORTU:")
        print(f"   • Wygenerowano {len(filtered_stocks)} rekomendacji inwestycyjnych")
        print(f"   • Główny plik eksportu: ROE-posortowany '{output_file}'")
        print(f"   • Analiza obejmuje dane z ostatniego kwartału finansowego")
        print(f"   • Plik gotowy do dalszej analizy w Excel/PowerBI")
    else:
        print("❌ Brak danych do eksportu")
        print("   • Sprawdź połączenie internetowe i dostęp do API Yahoo Finance")

    # Professional closing
    print("\n" + "="*60)
    print("🏛️  KONIEC RAPORTU INWESTYCYJNEGO")
    print("="*60)
    print("⚠️  DISCLAIMER:")
    print("   • Niniejszy raport ma charakter informacyjny i edukacyjny")
    print("   • Nie stanowi porady inwestycyjnej w rozumieniu prawa")
    print("   • Inwestycje giełdowe wiążą się z ryzykiem utraty kapitału")
    print("   • Przed podjęciem decyzji inwestycyjnej zalecamy konsultację")
    print("     z licencjonowanym doradcą inwestycyjnym")
    print("   • Historyczne wyniki nie gwarantują przyszłych zysków")
    print(f"\n📅 Raport wygenerowano: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print("🔄 Aktualizacja danych: comiesięczna (po wynikach kwartalnych)")
    print("="*60)


if __name__ == '__main__':
    main()