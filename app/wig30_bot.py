import pandas as pd
from datetime import datetime
from analyzer import analyze_profitability
from config import WIG30_TICKERS, OUTPUT_CSV_FILE


def main():
    """Main function to analyze WIG30 companies for profitability."""
    print("WIG30 Profitability Scanner")
    print("=" * 50)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print()

    # Analyze all companies
    all_results = []

    print("Analyzing WIG30 companies...")
    for ticker in WIG30_TICKERS:
        print(f"Processing {ticker}...", end=' ')
        result = analyze_profitability(ticker)
        if result:
            all_results.append(result)
            print("✓")
        else:
            print("✗")

    print()

    # Filter profitable companies
    profitable_stocks = [stock for stock in all_results if stock['profitable']]

    # Print summary
    print(f"Analysis Summary:")
    print(f"Total companies analyzed: {len(all_results)}")
    print(f"Profitable companies: {len(profitable_stocks)}")
    print(f"Unprofitable companies: {len(all_results) - len(profitable_stocks)}")
    print()

    # Print profitable companies
    if profitable_stocks:
        print("Profitable Companies (Positive Net Income):")
        print("-" * 60)
        print(f"{'Ticker':<10} {'Company Name':<30} {'Net Income':<15} {'Quarter End':<12}")
        print("-" * 60)

        for stock in profitable_stocks:
            ticker = stock['ticker']
            name = stock['name'][:28] + '..' if len(stock['name']) > 30 else stock['name']
            net_income = f"{stock['net_income']:,.2f}"
            quarter_end = stock['quarter_end']

            print(f"{ticker:<10} {name:<30} {net_income:>15} {quarter_end:<12}")
    else:
        print("No profitable companies found in the latest quarter.")

    print()

    # Export to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Results exported to: {OUTPUT_CSV_FILE}")

        # Also export only profitable companies
        if profitable_stocks:
            profitable_df = pd.DataFrame(profitable_stocks)
            profitable_filename = OUTPUT_CSV_FILE.replace('.csv', '_profitable_only.csv')
            profitable_df.to_csv(profitable_filename, index=False)
            print(f"Profitable companies exported to: {profitable_filename}")
    else:
        print("No data to export.")


if __name__ == '__main__':
    main()