#!/usr/bin/env python3
"""
Stock Monitoring System Launcher
Quick start script for real-time stock monitoring
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def print_banner():
    """Print startup banner."""
    print("🏛️" * 60)
    print("📊 SYSTEM MONITORINGU SPÓŁEK WIG30 - REAL-TIME")
    print("🏛️" * 60)
    print(f"🚅 Start systemu: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print("=" * 60)

def check_prerequisites():
    """Check if all necessary files exist."""
    required_files = [
        'wig30_analysis_pe_threshold.csv',
        'monitor.py',
        'dashboard_generator.py'
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
        return False

    return True

def generate_dashboard():
    """Generate HTML dashboard."""
    print("📊 Generowanie dashboardu HTML...")
    try:
        subprocess.run([sys.executable, 'dashboard_generator.py'], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Błąd generowania dashboardu")
        return False

def start_monitoring():
    """Start the monitoring system."""
    print("\n🚀 Uruchamianie monitoringu w czasie rzeczywistym...")
    print("💡 Wskazówki:")
    print("   • Monitorowanie co 30 sekund")
    print("   • Alerty przy zmianach ≥ ±2%")
    print("   • Naciśnij Ctrl+C aby zatrzymać")
    print("   • Dane zapisywane automatycznie")
    print("\n" + "="*60)

    try:
        subprocess.run([sys.executable, 'monitor.py'])
    except KeyboardInterrupt:
        print("\n🛑 Monitoring zatrzymany przez użytkownika")

def show_summary():
    """Show monitoring summary."""
    print("\n📊 PODSUMOWANIE SYSTEMU:")
    print("✅ Monitorowane spółki: WIG30")
    print("✅ Aktualizacje cen: co 30 sekund")
    print("✅ Alerty cenowe: automatyczne")
    print("✅ Dashboard HTML: dostępny w przeglądarce")
    print("✅ Logowanie alertów: CSV")
    print("✅ Zapis danych: JSON")

def main():
    """Main launcher function."""
    print_banner()

    # Check prerequisites
    if not check_prerequisites():
        input("\nNaciśnij Enter aby zakończyć...")
        return

    # Generate dashboard
    if not generate_dashboard():
        input("\nNaciśnij Enter aby zakończyć...")
        return

    # Show instructions
    print("\n🌐 DASHBOARD:")
    print("   • Otwórz: dashboard.html w przeglądarce")
    print("   • Auto-odświeżanie co 30 sekund")
    print("   • Widok na komórkę i desktop")

    # Ask user if they want to start monitoring
    response = input("\n❓ Czy chcesz uruchomić monitoring w czasie rzeczywistym? (t/n): ").lower().strip()

    if response in ['t', 'tak', 'yes', 'y']:
        start_monitoring()
        show_summary()
    else:
        print("\n💡 Monitorowanie nie uruchomione.")
        print("   • Uruchom później: python3 monitor.py")
        print("   • Dashboard dostępny: dashboard.html")

    print(f"\n🏁 Zakończono: {datetime.now().strftime('%H:%M:%S')}")
    input("Naciśnij Enter aby zakończyć...")

if __name__ == "__main__":
    main()