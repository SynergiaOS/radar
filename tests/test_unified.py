#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test zunifikowanego analyzera
"""

from unified_system import UnifiedWIG30Analyzer
import time

def test_unified_analyzer():
    """Testuje zunifikowany analyzer na wybranych spółkach"""

    print("🧪 Testowanie Zunifikowanego Analyzer WIG30 v2.0")
    print("=" * 60)

    # Zmniejsz listę tickerów do testu
    analyzer = UnifiedWIG30Analyzer()
    analyzer.tickers = ['CDR.WA', 'MBK.WA', 'PKN.WA']  # Tylko 3 spółki do testu

    start_time = time.time()

    try:
        # Uruchom analizę
        analyzer.run_full_analysis()

        # Generuj raport
        analyzer.generate_comprehensive_report()

        end_time = time.time()
        print(f"\n⏱️ Test zakończony w {end_time - start_time:.1f} sekund")
        print("✅ Zunifikowany analyzer działa poprawnie!")

    except Exception as e:
        print(f"❌ Błąd testu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unified_analyzer()