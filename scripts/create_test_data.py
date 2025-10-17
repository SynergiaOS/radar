#!/usr/bin/env python3
"""
Test Data Generator for File-Based Monitoring
Creates sample stock data files for testing the file monitoring system
"""

import pandas as pd
import json
import random
from datetime import datetime, timedelta
import os

def create_sample_stock_data():
    """Create sample stock data files for testing."""

    # Sample stock data
    stocks_data = [
        {'ticker': 'TXT.WA', 'name': 'Text S.A.', 'base_price': 51.00},
        {'ticker': 'XTB.WA', 'name': 'XTB S.A.', 'base_price': 67.44},
        {'ticker': 'PKN.WA', 'name': 'Orlen S.A.', 'base_price': 87.76},
        {'ticker': 'PKO.WA', 'name': 'PKO BP S.A.', 'base_price': 73.72},
        {'ticker': 'PZU.WA', 'name': 'PZU S.A.', 'base_price': 55.40},
        {'ticker': 'LPP.WA', 'name': 'LPP S.A.', 'base_price': 17240.00},
        {'ticker': 'CDR.WA', 'name': 'CD Projekt S.A.', 'base_price': 257.30},
        {'ticker': 'KGH.WA', 'name': 'KGHM S.A.', 'base_price': 188.95},
        {'ticker': 'PEO.WA', 'name': 'Bank PEKAO S.A.', 'base_price': 184.20},
        {'ticker': 'MBK.WA', 'name': 'mBank S.A.', 'base_price': 924.20},
    ]

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Generate current prices with small random variations
    current_data = []
    for stock in stocks_data:
        # Add random change between -2% and +2%
        change_pct = random.uniform(-2, 2)
        current_price = stock['base_price'] * (1 + change_pct / 100)

        current_data.append({
            'ticker': stock['ticker'],
            'name': stock['name'],
            'price': round(current_price, 2),
            'change': round(change_pct, 2),
            'volume': random.randint(1000, 10000),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })

    # 1. Create CSV file
    df_csv = pd.DataFrame(current_data)
    csv_file = 'data/current_prices.csv'
    df_csv.to_csv(csv_file, index=False)
    print(f"✅ Utworzono plik CSV: {csv_file}")

    # 2. Create GPW-style CSV
    gpw_data = []
    for stock in current_data:
        gpw_data.append([stock['ticker'], stock['price'], stock['change'], stock['volume']])

    gpw_file = 'data/gpw_quotes.csv'
    with open(gpw_file, 'w', encoding='utf-8') as f:
        f.write('Ticker,Cena,Zmiana,Wolumen\n')
        for row in gpw_data:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
    print(f"✅ Utworzono plik GPW: {gpw_file}")

    # 3. Create Excel file (skip if no openpyxl)
    excel_file = 'data/stock_data.xlsx'
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_csv.to_excel(writer, sheet_name='Current Prices', index=False)
        print(f"✅ Utworzono plik Excel: {excel_file}")
    except ImportError:
        print(f"⚠️  Pominięto Excel (brak openpyxl): {excel_file}")

    # 4. Create JSON file
    json_data = {}
    for stock in current_data:
        json_data[stock['ticker']] = {
            'price': stock['price'],
            'change': stock['change'],
            'name': stock['name'],
            'timestamp': stock['timestamp']
        }

    json_file = 'data/live_prices.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Utworzono plik JSON: {json_file}")

    # 5. Create simple current_prices.csv in root
    simple_file = 'current_prices.csv'
    simple_data = []
    for stock in current_data:
        simple_data.append({
            'symbol': stock['ticker'],
            'price': stock['price'],
            'change_pct': stock['change']
        })

    df_simple = pd.DataFrame(simple_data)
    df_simple.to_csv(simple_file, index=False)
    print(f"✅ Utworzono prosty plik CSV: {simple_file}")

    return current_data

def simulate_price_updates():
    """Simulate real-time price updates in files."""
    print(f"\n🔄 Symulacja aktualizacji cen w plikach...")

    # Read existing data
    if os.path.exists('data/current_prices.csv'):
        df = pd.read_csv('data/current_prices.csv')

        # Update prices with small random changes
        for idx, row in df.iterrows():
            change_pct = random.uniform(-1.5, 1.5)
            current_price = row['price']
            new_price = current_price * (1 + change_pct / 100)

            df.at[idx, 'price'] = round(new_price, 2)
            df.at[idx, 'change'] = round(change_pct, 2)
            df.at[idx, 'timestamp'] = datetime.now().strftime('%H:%M:%S')

        # Save updated data
        df.to_csv('data/current_prices.csv', index=False)

        # Update other files too
        df.to_csv('current_prices.csv', index=False)

        # Update JSON
        json_data = {}
        for _, row in df.iterrows():
            json_data[row['ticker']] = {
                'price': row['price'],
                'change': row['change'],
                'name': row['name'],
                'timestamp': row['timestamp']
            }

        with open('data/live_prices.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Zaktualizowano ceny w plikach")
        return True

    return False

def display_file_info():
    """Display information about created files."""
    print(f"\n📁 WYGENEROWANE PLIKI DANYCH:")

    files_to_check = [
        'data/current_prices.csv',
        'data/gpw_quotes.csv',
        'data/stock_data.xlsx',
        'data/live_prices.json',
        'current_prices.csv'
    ]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"   ✅ {file_path} ({size} bytes, {mtime.strftime('%H:%M:%S')})")
        else:
            print(f"   ❌ {file_path} (nie znaleziono)")

def main():
    """Main function to create test data."""
    print("🏛️  GENERATOR DANYCH TESTOWYCH DLA MONITORINGU Z PLIKÓW")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print("=" * 60)

    # Create initial data
    data = create_sample_stock_data()

    print(f"\n📊 WYGENEROWANO DANE DLA {len(data)} SPÓŁEK:")
    for stock in data[:5]:  # Show first 5
        change_emoji = "📈" if stock['change'] > 0 else "📉" if stock['change'] < 0 else "➡️"
        print(f"   {change_emoji} {stock['ticker']}: {stock['price']:.2f} PLN ({stock['change']:+.2f}%)")

    if len(data) > 5:
        print(f"   ... i {len(data) - 5} kolejnych spółek")

    display_file_info()

    print(f"\n💡 WSKAZÓWKI:")
    print(f"   • Pliki gotowe do testowania: python3 file_monitor.py")
    print(f"   • Aktualizuj dane: python3 create_test_data.py --update")
    print(f"   • Źródło danych: PLIKI LOKALNE (bez API)")

    # Check if update flag
    import sys
    if '--update' in sys.argv:
        simulate_price_updates()

if __name__ == "__main__":
    main()