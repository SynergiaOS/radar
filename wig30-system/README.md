# 🏆 WIG30 Trading System v2.0

**Zunifikowany system analizy i tradingu na GPW WIG30**

## 📋 Opis Systemu

Kompleksowy system do analizy fundamentalnej, technicznej i future scoring spółek z indeksu WIG30. Łączy w sobie funkcjonalność trzech poprzednich analyzerów w jedno spójne rozwiązanie.

### 🎯 Główne Funkcjonalności

- **Analiza Fundamentalna**: ROE, P/E, P/B, rentowność, wskaźniki zadłużenia
- **Analiza Techniczna**: MA, RSI, MACD, Bollinger Bands, analiza trendu
- **Future Scoring**: Algorytm oceny potencjału przyszłego wzrostu (0-100)
- **Rekomendacje AI**: System rekomendacji kup/trzymaj/sprzedaj
- **Ryzyko Management**: Zarządzanie pozycjami i portfolio heat
- **Dashboard**: Webowy interfejs z profesjonalnymi wykresami

## 🏗️ Architektura Systemu

```
wig30-system/
├── backend/                    # Backend Python Flask
│   ├── api/                   # Endpointy API
│   ├── services/              # Logika biznesowa
│   ├── models/                # Modele danych
│   └── utils/                 # Helper functions
├── frontend/                   # Frontend Next.js
│   ├── components/            # Komponenty React
│   ├── pages/                 # Strony aplikacji
│   └── utils/                 # Utility functions
├── unified_system/             # Zunifikowany analyzer
│   ├── core/                  # Główna logika analizy
│   ├── config/                # Konfiguracja systemu
│   └── services/              # Serwisy zewnętrzne
├── data/                       # Dane i eksporty
│   ├── exports/               # Wyniki analiz
│   ├── legacy/                # Archiwalne dane
│   └── charts/                # Wykresy
├── docker/                     # Konfiguracje Docker
├── scripts/                    # Skrypty startowe
└── docs/                       # Dokumentacja
```

## 🚀 Szybki Start

### 1. Uruchomienie Backend (Flask Dashboard)
```bash
cd /home/marcin/windsurf/Projects/radar
python web_dashboard.py
```
Dostępne na: http://localhost:5000

### 2. Uruchomienie Frontend (Next.js)
```bash
cd /home/marcin/windsurf/Projects/radar/gpw-smart-analyzer
npm run dev
```
Dostępne na: http://localhost:3001

### 3. Uruchomienie Zunifikowanego Analyzer
```bash
cd /home/marcin/windsurf/Projects/radar
python -c "from unified_system import UnifiedWIG30Analyzer; analyzer = UnifiedWIG30Analyzer(); analyzer.run_full_analysis(); analyzer.generate_comprehensive_report()"
```

## 📊 Funkcje Kluczowe

### 🔍 Zunifikowany Analyzer
- Łączy funkcjonalności: `wig30_bot.py`, `simple_wig30_analyzer.py`, `dynamic_wig30_analyzer.py`
- Analiza 30 spółek WIG30 w czasie rzeczywistym
- Future Score (0-100) z uwzględnieniem sektorów
- Rekomendacje DM (mBank, DM BOŚ)

### 📈 Analiza Fundamentalna
- **ROE**: Rentowność kapitału własnego
- **P/E**: Cena do zysku
- **P/B**: Cena do wartości księgowej
- **Rentowność**: Zysk netto, przychody
- **Wskaźniki zadłużenia**: Debt-to-Equity, Current Ratio

### 📉 Analiza Techniczna
- **Moving Averages**: 5, 10, 20, 50 dni
- **RSI**: Relative Strength Index (14)
- **MACD**: z sygnałami i histogramem
- **Bollinger Bands**: z odchyleniem standardowym
- **Analiza trendu**: wzrostowy/spadkowy/boczny

### 🎯 Future Score Algorithm
- **Sektorowe bonusy**: Technologia/Gaming (+25), Banki (+20), E-commerce (+22), Energia (+18)
- **Trend techniczny**: Wzrostowy (+15), Boczny (+10)
- **Rekomendacje DM**: +12 punktów
- **RSI w zakresie**: +8 punktów
- **Przychody wzrostowe**: +5 punktów

## 🔧 Konfiguracja

System wykorzystuje zunifikowany plik konfiguracyjny:
```python
# unified_system/config/settings.py
config = Config()
```

### Główne progi:
- **ROE ≥ 10.0%**: Minimalna rentowność
- **P/E ≤ 20.0**: Maksymalna wycena
- **P/B ≤ 3.0**: Maksymalna wartość księgowa
- **Future Score ≥ 45**: Próg spółek przyszłościowych

## 📋 Przykładowe Wyniki

**Top 3 Przyszłościowe Spółki (październik 2025):**

1. **CDR.WA** (CD Projekt) - Future Score: 60/100
   - Sektor: Technologia/Gaming
   - Czynniki: Stabilizacja, Rekomendacje DM, RSI w zakresie

2. **MBK.WA** (mBank) - Future Score: 60/100
   - Sektor: Banki
   - Czynniki: Trend wzrostowy, Rekomendacje DM, Atrakcyjna wycena

3. **PKN.WA** (Orlen) - Future Score: 58/100
   - Sektor: Energia
   - Czynniki: Trend wzrostowy, Rekomendacje DM, Atrakcyjna wycena

## 🛡️ Security

System wykorzystuje zabezpieczenia:
- ✅ Usunięto luki w dynamicznym wykonywaniu kodu (ProfessionalChart.tsx)
- ✅ Bezpieczne deserializacja callback functions
- ✅ Input validation i sanitization
- ✅ Rate limiting na API zewnętrzne

## 📚 Dokumentacja

- [API Documentation](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Współpraca

Projekt stworzony przez:
- **WIG30 Radar Team**
- **License**: MIT

## 📞 Kontakt

- GitHub Issues: [Report Bug](https://github.com/wig30-radar/issues)
- Dokumentacja: [Wiki](https://github.com/wig30-radar/wiki)

---

**⚠️ Disclaimer**: Niniejszy system ma charakter informacyjny i edukacyjny. Nie stanowi porady inwestycyjnej. Inwestycje giełdowe wiążą się z ryzykiem utraty kapitału.