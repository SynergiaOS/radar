# WIG30/WIG20 Profitability & Valuation Scanner Bot

A Python bot that analyzes the profitability and valuation of companies in the WIG30 and WIG20 indices using fundamental metrics from Yahoo Finance.

## Description

This bot automatically fetches financial data for companies in the WIG30 or WIG20 index (Poland's main stock market indices), calculates Return on Equity (ROE) and Price-to-Earnings (P/E) ratios, and applies dual filtering to identify fundamentally strong and undervalued companies. The bot filters companies based on positive net income, ROE ≥ 10% (profitability) AND P/E ≤ 15 (valuation), then displays a top 10 leaderboard sorted by ROE. Results are displayed in the console with comprehensive formatting showing ROE, P/E ratios, and exported to multiple CSV files for further analysis.

## Features

- **Automatic Data Fetching**: Retrieves quarterly financial statements, balance sheets, and current market data from Yahoo Finance
- **ROE Calculation**: Calculates Return on Equity (ROE) for profitability assessment
- **P/E Ratio Analysis**: Calculates Price-to-Earnings ratio for valuation assessment
- **Dual Filtering Strategy**: Combines ROE ≥ 10% (profitability) AND P/E ≤ 15 (valuation) for value investing
- **Index Flexibility**: Supports both WIG30 (30 companies) and WIG20 (20 blue-chip companies) analysis
- **Comprehensive Analysis**: Analyzes all companies in selected index in a single run
- **Smart Filtering**: Identifies companies with positive net income and applies fundamental criteria
- **ROE-Based Sorting**: Automatically sorts results by ROE to identify the most capital-efficient companies
- **Enhanced Console Output**: Displays top 10 leaderboard with emojis, ROE, P/E ratios, net income, and company names
- **Multiple CSV Exports**: Generates filtered results, all profitable companies, and legacy format files
- **Error Handling**: Gracefully handles missing data and API errors
- **Modular Design**: Clean separation of configuration, analysis logic, and main execution

## Installation

### Prerequisites
- Python 3.8 or higher (recommended: 3.9+ for best performance)

### Setup Steps
1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Bot
Execute the main script:
```bash
python wig30_bot.py
```

### Expected Output
The bot will:
1. Display progress for each company being analyzed
2. Show a summary with total companies analyzed and those meeting dual criteria
3. Print a top 10 leaderboard of companies meeting ROE ≥ 10% AND P/E ≤ 15 criteria, displaying:
   - Ranking number
   - Ticker symbol
   - ROE percentage
   - P/E ratio
   - Net income with thousand separators
   - Company name
4. Generate multiple CSV files in the same directory

### Generated Files
- `wig30_analysis_pe_threshold.csv` - Complete analysis of WIG30 companies meeting dual criteria
- `wig30_analysis_pe_threshold_filtered_only.csv` - Only companies meeting dual criteria
- `wig30_analysis_pe_threshold_all_profitable.csv` - All profitable companies (before P/E filtering)
- `wig20_analysis_pe_threshold.csv` - WIG20 version (when ACTIVE_INDEX = 'WIG20')
- `wig30_analysis.csv` - Legacy format for backward compatibility

### Example Output
```
🏆 Analiza WIG20 na 2025-10-13
==================================================
Rentowne z ROE ≥ 10% i P/E ≤ 15: 9/19
TOP po ROE (z P/E):
------------------------------------------------------------------------------------------------------------------------
Lp. Ticker    | ROE    | P/E    |     Zysk netto | Nazwa
------------------------------------------------------------------------------------------------------------------------
  1. KRU.WA   | ROE: 28.5% | P/E:  8.2 |    120,000,000 PLN | Kruk SA
  2. PKN.WA   | ROE: 22.1% | P/E:  6.5 |  1,570,000,000 PLN | ORLEN S.A.
  3. PZU.WA   | ROE: 18.7% | P/E:  7.8 |  1,470,000,000 PLN | PZU SA
...

📊 Rentowne z ROE ≥ 10% i P/E ≤ 15: 9/19
Pełne wyniki w 'wig20_analysis_pe_threshold.csv'
```

## Configuration

### Modifying Settings
All configuration is centralized in `config.py`:

1. **Index Selection**: Set `ACTIVE_INDEX` to 'WIG30' or 'WIG20' to choose which index to analyze
2. **Update Ticker List**: Modify `WIG30_TICKERS` or `WIG20_TICKERS` to add/remove companies
3. **Adjust ROE Threshold**: Change `ROE_THRESHOLD` (default: 10.0) - minimum ROE percentage for filtering
4. **Set P/E Threshold**: Adjust `PE_THRESHOLD` (default: 15.0) - maximum P/E ratio for value investing filter
5. **Enable/Disable Dual Filtering**: Set `ENABLE_DUAL_FILTER` (default: True) to toggle simultaneous ROE and P/E filtering
6. **Control Display Count**: Modify `TOP_N_DISPLAY` (default: 10) to change how many top companies are shown
7. **Change Output Filename**: Update `OUTPUT_CSV_FILE` constant

**Example: To analyze only profitability without P/E filtering:**
```python
ENABLE_DUAL_FILTER = False  # Disable dual filtering
```

**Example: To change filtering criteria:**
```python
ROE_THRESHOLD = 15.0  # Only companies with ROE ≥ 15%
PE_THRESHOLD = 12.0  # More conservative valuation (P/E ≤ 12)
```

### Adding New Companies
To add companies to the analysis:
1. Open `config.py`
2. Add the ticker symbol to `WIG30_TICKERS` or `WIG20_TICKERS` list
3. Save the file

## Index Selection

### WIG30 vs WIG20
- **WIG30**: 30 largest and most liquid companies on the Warsaw Stock Exchange
- **WIG20**: Blue-chip index of the 20 largest companies (subset of WIG30)

### How to Switch Between Indices
In `config.py`, change the `ACTIVE_INDEX` constant:
```python
ACTIVE_INDEX = 'WIG20'  # Analyze WIG20 companies
# or
ACTIVE_INDEX = 'WIG30'  # Analyze WIG30 companies (default)
```

## Data Source

**Source**: Yahoo Finance (via yfinance library)
**Data Type**: Quarterly financial statements, balance sheets, and current market data
**Key Metrics**: Net Income, Shareholders' Equity, Current Price, Trailing EPS, ROE, P/E

## Metrics Explained

- **Net Income**: Total profit (or loss) reported in the latest quarter, measured in PLN. A positive net income indicates the company was profitable during that quarter.

- **ROE (Return on Equity)**: Measures how efficiently a company generates profit from shareholders' equity. Calculated as `(Net Income / Shareholders' Equity) × 100%`. Higher ROE indicates better profitability relative to the capital invested by shareholders. For example, ROE of 20% means the company generates 20 PLN of profit for every 100 PLN of equity.

- **P/E Ratio (Price-to-Earnings)**: Measures stock valuation by comparing current stock price to earnings per share (EPS). Formula: P/E = Current Price / Trailing EPS (TTM). Interpretation: Lower P/E suggests the stock may be undervalued. P/E of 10 means you pay 10 PLN for every 1 PLN of annual earnings. Typical value investing threshold: P/E ≤ 15.


- **Profitability**: A company is considered profitable if its net income is positive (> 0). Only profitable companies are included in the analysis.

- **Dual Filtering Strategy**: Combining ROE ≥ 10% (profitability efficiency) with P/E ≤ 15 (reasonable valuation) identifies companies that are both fundamentally strong and potentially undervalued. This is a classic value investing approach.

## Limitations

- **Data Delays**: Financial data is updated quarterly, not in real-time
- **Data Availability**: Some companies may have missing or incomplete financial data
- **ROE Data Availability**: ROE may be unavailable (displayed as "N/A") for some companies if:
  - Balance sheet data is missing from Yahoo Finance
  - Shareholders' equity is zero or negative (can happen for companies with accumulated losses)
  - Data quality issues or reporting delays
- **P/E Ratio Limitations**: P/E may be unavailable for some companies if price or EPS data is missing
  - P/E is not meaningful for unprofitable companies (negative EPS)
  - P/E can be distorted by one-time events affecting earnings
  - Different industries have different "normal" P/E ranges (banks typically have lower P/E than tech companies)
  - The bot uses trailing P/E (based on past 12 months earnings), not forward P/E (based on estimated future earnings)
- **Currency**: Net income is shown in the company's reporting currency
- **Not Investment Advice**: This tool is for informational purposes only and should not be used as investment advice
- **Independent Verification**: Always verify financial data from multiple sources before making decisions

## Future Enhancements

Potential improvements for future versions:
- **Dividend Yield Analysis**: Add dividend yield calculation for income investing strategies
- **Debt-to-Equity Ratio**: Include financial health assessment metrics
- **Price Momentum Indicators**: Add 52-week high/low analysis and momentum indicators
- **Sector-Based Filtering**: Enable filtering and comparison by industry sectors
- **Historical Trend Analysis**: Track ROE and P/E changes over multiple quarters
- **PEG Ratio Calculation**: Add P/E to Growth ratio for growth stock valuation
- **Industry Benchmarking**: Compare company metrics with sector averages and WIG median
- **Forward P/E Analysis**: Include analyst estimates and forward-looking valuations
- **Automated Alerts**: Set up notifications when companies meet filtering criteria
- **Interactive Charts**: Visualize metric distributions using matplotlib or plotly
- **Portfolio Integration**: Import current holdings and analyze against screening criteria
- **Multi-Market Support**: Expand to other European markets or international indices

## Language Note

This bot was developed for analyzing Polish stock market data, but can be easily adapted for other markets by updating the ticker list in the configuration file.

---

# Wersja Polska (Polish Version)

## Bot Skanera Rentowności i Wyceny WIG30/WIG20

Bot Python, który analizuje rentowność i wycenę spółek z indeksów WIG30 i WIG20, wykorzystując wskaźniki fundamentalne z Yahoo Finance.

## Opis

Ten bot automatycznie pobiera dane finansowe spółek z indeksu WIG30 lub WIG20 (główne indeksy GPW), oblicza Rentowność Kapitału Własnego (ROE) oraz wskaźnik Cena/Zysk (C/Z), a następnie stosuje podwójne filtrowanie do identyfikacji fundamentalnie mocnych i niedowartościowanych spółek. Bot filtruje spółki na podstawie dodatniego zysku netto, ROE ≥ 10% (rentowność) ORAZ C/Z ≤ 15 (wycena), a następnie wyświetla top 10 ranking posortowany według ROE. Wyniki są prezentowane w konsoli z kompleksowym formatowaniem pokazującym ROE, wskaźniki C/Z, i eksportowane do wielu plików CSV do dalszej analizy.

## Funkcje

- **Automatyczne Pobieranie Danych**: Pobiera kwartalne sprawozdania finansowe, bilanse i aktualne dane rynkowe z Yahoo Finance
- **Obliczanie ROE**: Oblicza Rentowność Kapitału Własnego (ROE) do oceny rentowności
- **Analiza Wskaźnika C/Z**: Oblicza wskaźnik Cena/Zysk do oceny wyceny
- **Strategia Podwójnego Filtrowania**: Łączy ROE ≥ 10% (rentowność) ORAZ C/Z ≤ 15 (wycena) dla inwestowania wartościowego
- **Elastyczność Indeksu**: Wspiera analizę zarówno WIG30 (30 spółek) jak i WIG20 (20 spółek blue-chip)
- **Kompleksowa Analiza**: Analizuje wszystkie spółki z wybranego indeksu w jednym uruchomieniu
- **Inteligentne Filtrowanie**: Identyfikuje spółki z dodatnim zyskiem netto i stosuje kryteria fundamentalne
- **Sortowanie według ROE**: Automatycznie sortuje wyniki według ROE dla identyfikacji najbardziej kapitałowo wydajnych spółek
- **Zwiększony Wyjście Konsoli**: Wyświetla top 10 ranking z emoji, wskaźnikami ROE, C/Z, zyskami netto i nazwami spółek
- **Wiele Eksportów CSV**: Generuje przefiltrowane wyniki, wszystkie rentowne spółki i pliki w formacie legacy
- **Obsługa Błędów**: Elegancko radzi sobie z brakującymi danymi i błędami API
- **Modularny Projekt**: Czyste rozdzielenie konfiguracji, logiki analizy i głównego wykonania

## Instalacja

### Wymagania
- Python 3.8 lub wyższy (zalecany: 3.9+ dla najlepszej wydajności)

### Kroki Instalacji
1. Sklonuj lub pobierz to repozytorium
2. Zainstaluj wymagane zależności:
   ```bash
   pip install -r requirements.txt
   ```

## Użycie

### Uruchomienie Bota
Wykonaj główny skrypt:
```bash
python wig30_bot.py
```

### Oczekiwane Wyjście
Bot wyświetli:
1. Postęp dla każdej analizowanej spółki
2. Podsumowanie z łączną liczbą przeanalizowanych spółek i spełniających podwójne kryteria
3. Top 10 ranking spółek spełniających kryteria ROE ≥ 10% ORAZ C/Z ≤ 15, wyświetlający:
   - Numer rankingu
   - Symbol ticker
   - Procentowe ROE
   - Wskaźnik C/Z
   - Zysk netto z separatorami tysięcy
   - Nazwa spółki
4. Generuje wiele plików CSV w tym samym katalogu

### Przykładowe Wyjście
```
📊 RAPORT INWESTYCYJNY - ANALIZA WIG30
📅 Data raportu: 16.10.2025
============================================================
🎯 Strategia: Value Investing (ROE ≥ 10% + C/Z ≤ 15)
💡 Cel: Identyfikacja fundamentalnie mocnych i niedowartościowanych spółek

🔍 PRZEBIEG ANALIZY:
   Analizowanie PEO.WA... ✅
   Analizowanie PKN.WA... ✅
   ...

📈 PODSUMOWANIE DLA ZARZĄDU:
   • Łącznie przeanalizowano spółek: 30
   • Spółki rentowne (zysk netto > 0): 18 (60.0%)
   • Spółki spełniające kryteria strategii: 12 (40.0%)

🎯 REKOMENDACJE INWESTYCYJNE:
📊 Spółki spełniające kryteria Value Investing:
==================================================================================
Lp. Ticker    | ROE    | C/Z    |     Zysk Netto | Nazwa Spółki
==================================================================================
  1. KRU.WA   | ROE: 28.5% | C/Z:  8.2 |    120,000,000 PLN | Kruk SA
  2. PKN.WA   | ROE: 22.1% | C/Z:  6.5 |  1,570,000,000 PLN | ORLEN S.A.
  3. PZU.WA   | ROE: 18.7% | C/Z:  7.8 |  1,470,000,000 PLN | PZU SA
...
```

## Język

Bot wyświetla wszystkie komunikaty w języku polskim, co czyni go idealnym dla polskiego rynku inwestycyjnego.

## License

This project is open source and available under the MIT License.