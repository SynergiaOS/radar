import pandas as pd
from datetime import datetime
from analyzer import analyze_company_fundamentals
from config import (
    WIG30_TICKERS, WIG20_TICKERS, OUTPUT_CSV_FILE, WIG20_OUTPUT_CSV_FILE,
    LEGACY_CSV_FILE, TOP_N_DISPLAY, ACTIVE_INDEX, ROE_THRESHOLD,
    PE_THRESHOLD, ENABLE_DUAL_FILTER
)


def meets_dual_filter_criteria(stock):
    """Check if a stock meets both ROE and P/E filtering criteria."""
    # Check ROE criteria
    roe_ok = stock.get('roe') is not None and stock['roe'] >= ROE_THRESHOLD

    # Check P/E criteria
    pe_ok = stock.get('pe_ratio') is not None and stock['pe_ratio'] <= PE_THRESHOLD

    return roe_ok and pe_ok


def main():
    """Main function to analyze WIG30/WIG20 companies for profitability and valuation."""
    # Select tickers and output file based on active index
    tickers = WIG20_TICKERS if ACTIVE_INDEX == 'WIG20' else WIG30_TICKERS
    output_file = WIG20_OUTPUT_CSV_FILE if ACTIVE_INDEX == 'WIG20' else OUTPUT_CSV_FILE

    print(f"📊 RAPORT INWESTYCYJNY - ANALIZA {ACTIVE_INDEX}")
    print(f"📅 Data raportu: {datetime.now().strftime('%d.%m.%Y')}")
    print("=" * 60)
    print("🎯 Strategia: Value Investing (ROE ≥ 10% + P/E ≤ 15)")
    print("💡 Cel: Identyfikacja fundamentalnie mocnych i niedowartościowanych spółek")
    print()

    # Analyze all companies
    all_results = []

    print("🔍 PRZEBIEG ANALIZY:")
    for ticker in tickers:
        print(f"   Analizowanie {ticker}...", end=' ')
        result = analyze_company_fundamentals(ticker)
        if result:
            all_results.append(result)
            print("✅")
        else:
            print("❌")

    print()

    # Filter profitable companies
    profitable_stocks = [stock for stock in all_results if stock['profitable']]

    # Apply dual filtering if enabled
    if ENABLE_DUAL_FILTER:
        filtered_stocks = [stock for stock in profitable_stocks if meets_dual_filter_criteria(stock)]
    else:
        filtered_stocks = profitable_stocks

    # Sort filtered companies by ROE (descending), None values last
    filtered_stocks = sorted(
        filtered_stocks,
        key=lambda x: x.get('roe') or float('-inf'),
        reverse=True
    )

    # Print executive summary
    print("📈 PODSUMOWANIE DLA ZARZĄDU:")
    print(f"   • Łącznie przeanalizowano spółek: {len(all_results)}")
    print(f"   • Spółki rentowne (zysk netto > 0): {len(profitable_stocks)} ({len(profitable_stocks)/len(all_results)*100:.1f}%)")
    print(f"   • Spółki spełniające kryteria strategii: {len(filtered_stocks)} ({len(filtered_stocks)/len(all_results)*100:.1f}%)")
    print()

    # Print investment recommendations
    if filtered_stocks:
        print("🎯 REKOMENDACJE INWESTYCYJNE:")
        print("📊 Spółki spełniające kryteria Value Investing:")
        print("=" * 110)
        print(f"{'Lp.':>3} {'Ticker':<8} | {'ROE':>6} | {'P/E':>6} | {'Zysk Netto':>15} | {'Nazwa Spółki'}")
        print("=" * 110)

        for i, stock in enumerate(filtered_stocks[:TOP_N_DISPLAY], 1):
            ticker = stock['ticker']
            name = stock['name']
            roe = stock.get('roe')
            pe_ratio = stock.get('pe_ratio')

            # Format values
            roe_str = f"{roe:.1f}%" if roe is not None else "N/A"
            pe_str = f"{pe_ratio:.1f}" if pe_ratio is not None else "N/A"
            net_income_str = f"{stock['net_income']:,.0f} PLN"

            # Truncate name if too long
            display_name = name[:35] + '..' if len(name) > 35 else name

            print(f"{i:>3}. {ticker:<8} | ROE: {roe_str:>5} | P/E: {pe_str:>5} | {net_income_str:>15} | {display_name}")

        print()
        print("💡 ANALIZA STRATEGICZNA:")
        print(f"   • Znaleziono {len(filtered_stocks)} spółki o potencjale inwestycyjnym")
        print(f"   • Średnie ROE: {sum(s.get('roe', 0) for s in filtered_stocks)/len(filtered_stocks):.1f}%")
        print(f"   • Średnie P/E: {sum(s.get('pe_ratio', 0) for s in filtered_stocks)/len(filtered_stocks):.1f}")
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
    else:
        print("⚠️  BRAK REKOMENDACJI INWESTYCYJNYCH")
        if ENABLE_DUAL_FILTER:
            print(f"   • Żadna spółka nie spełnia kryteriów ROE ≥ {ROE_THRESHOLD}% i P/E ≤ {PE_THRESHOLD}")
            print("   • Zalecamy rozważenie mniej restrykcyjnych kryteriów analizy")
        else:
            print("   • Brak spółek rentownych w ostatnim kwartale")

        print("\n💡 REKOMENDACJE STRATEGICZNE:")
        print("   • Rozważ rozszerzenie kryteriów (ROE ≥ 8% lub P/E ≤ 20)")
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

        # Export complete dataset for analysis
        df.to_csv(LEGACY_CSV_FILE, index=False)
        print(f"   ✅ Kompletny zbiór danych: '{LEGACY_CSV_FILE}'")

        print(f"\n📊 PODSUMOWANIE EKSPORTU:")
        print(f"   • Wygenerowano {len(filtered_stocks)} rekomendacji inwestycyjnych")
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