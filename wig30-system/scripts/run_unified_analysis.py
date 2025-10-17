#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Unified WIG30 Analysis Script
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run unified WIG30 analysis"""
    print("🎯 Zunifikowany System Analizy WIG30 v2.0")
    print("=" * 50)

    try:
        # Import unified analyzer
        from unified_system import UnifiedWIG30Analyzer

        # Initialize analyzer
        analyzer = UnifiedWIG30Analyzer()

        # Run full analysis
        analyzer.run_full_analysis()

        # Generate comprehensive report
        analyzer.generate_comprehensive_report()

        print("\n✅ Analiza zakończona pomyślnie!")
        print(f"📁 Wyniki zapisane w: {project_root}/unified_system/data/exports/")

        # Show summary
        if analyzer.future_stocks:
            print(f"\n🚀 Znaleziono {len(analyzer.future_stocks)} spółek przyszłościowych:")
            for i, stock in enumerate(analyzer.future_stocks[:5], 1):
                print(f"   {i}. {stock['ticker']} - Future Score: {stock['future_score']}")
        else:
            print("\n⚠️ Brak spółek spełniających kryteria przyszłościowe")

    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        print("Upewnij się, że jesteś w głównym katalogu projektu")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd analizy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()