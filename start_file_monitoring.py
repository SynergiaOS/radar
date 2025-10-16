#!/usr/bin/env python3
"""
File-Based Stock Monitoring System Launcher
Quick start script for file-based stock monitoring
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def print_banner():
    """Print startup banner."""
    print("🏛️" * 60)
    print("📂 SYSTEM MONITORINGU SPÓŁEK Z PLIKÓW - FILE-BASED")
    print("🏛️" * 60)
    print(f"🚅 Start systemu: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print("📂 Źródło danych: PLIKI LOKALNE (bez API)")
    print("=" * 60)

def check_data_files():
    """Check if data files exist."""
    required_files = [
        'wig30_analysis_pe_threshold.csv',
        'data/current_prices.csv',
        'file_monitor.py',
        'create_test_data.py'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Brakuje wymaganych plików:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Rozwiązanie:")
        if 'wig30_analysis_pe_threshold.csv' in missing_files:
            print("   • Uruchom najpierw: python3 wig30_bot.py")
        if 'data/current_prices.csv' in missing_files:
            print("   • Wygeneruj dane: python3 create_test_data.py")
        return False

    return True

def setup_data_files():
    """Setup data files for monitoring."""
    print("🔧 Konfiguracja plików danych...")

    if not os.path.exists('data/current_prices.csv'):
        print("📊 Tworzenie danych testowych...")
        try:
            subprocess.run([sys.executable, 'create_test_data.py'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Błąd tworzenia danych testowych")
            return False
    else:
        print("✅ Pliki danych już istnieją")
        return True

def show_data_status():
    """Show current data status."""
    print(f"\n📊 STATUS PLIKÓW DANYCH:")

    data_files = [
        'data/current_prices.csv',
        'data/gpw_quotes.csv',
        'data/live_prices.json',
        'current_prices.csv'
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"   ✅ {file_path} ({size} bytes, {mtime.strftime('%H:%M:%S')})")
        else:
            print(f"   ❌ {file_path} (brak)")

def start_file_monitoring():
    """Start the file-based monitoring system."""
    print("\n🚀 Uruchamianie monitoringu z plików...")
    print("💡 Wskazówki:")
    print("   • Monitorowanie co 15 sekund")
    print("   • Alerty przy zmianach ≥ ±1.5%")
    print("   • Dane czytane z plików CSV/JSON")
    print("   • Naciśnij Ctrl+C aby zatrzymać")
    print("   • Dane zapisywane automatycznie")
    print("\n" + "="*60)

    try:
        subprocess.run([sys.executable, 'file_monitor.py'])
    except KeyboardInterrupt:
        print("\n🛑 Monitoring zatrzymany przez użytkownika")

def show_instructions():
    """Show usage instructions."""
    print("\n📖 INSTRUKCJA OBSŁUGI:")
    print("=" * 40)
    print("1. 📊 Generowanie danych:")
    print("   python3 create_test_data.py")
    print("")
    print("2. 🔄 Aktualizacja danych:")
    print("   python3 create_test_data.py --update")
    print("")
    print("3. 📂 Uruchomienie monitoringu:")
    print("   python3 file_monitor.py")
    print("")
    print("4. 📁 Struktura plików:")
    print("   /data/")
    print("   ├── current_prices.csv")
    print("   ├── gpw_quotes.csv")
    print("   └── live_prices.json")
    print("")
    print("5. 🚨 Alerty zapisywane w:")
    print("   file_monitoring_log.csv")

def show_summary():
    """Show system summary."""
    print("\n📈 PODSUMOWANIE SYSTEMU PLIKOWEGO:")
    print("✅ Źródło danych: PLIKI LOKALNE")
    print("✅ Formaty: CSV, JSON, GPW")
    print("✅ Aktualizacje: co 15 sekund")
    print("✅ Alerty cenowe: automatyczne")
    print("✅ Logowanie: CSV + JSON")
    print("✅ Brak zależności API")

def main():
    """Main launcher function."""
    print_banner()

    # Check requirements
    if not check_data_files():
        input("\nNaciśnij Enter aby zakończyć...")
        return

    # Setup data files
    if not setup_data_files():
        input("\nNaciśnij Enter aby zakończyć...")
        return

    # Show data status
    show_data_status()

    # Show instructions
    show_instructions()

    # Ask user if they want to start monitoring
    response = input("\n❓ Czy chcesz uruchomić monitoring z plików? (t/n): ").lower().strip()

    if response in ['t', 'tak', 'yes', 'y']:
        start_file_monitoring()
        show_summary()
    else:
        print("\n💡 Monitoring nie uruchomiony.")
        print("   • Uruchom później: python3 file_monitor.py")
        print("   • Aktualizuj dane: python3 create_test_data.py --update")

    print(f"\n🏁 Zakończono: {datetime.now().strftime('%H:%M:%S')}")
    input("Naciśnij Enter aby zakończyć...")

if __name__ == "__main__":
    main()