#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start script for WIG30 Backend System
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = ['flask', 'pandas', 'numpy', 'yfinance', 'requests']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print(f"❌ Brakujące moduły: {', '.join(missing_modules)}")
        print("📦 Instalowanie brakujących zależności...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_modules)
        print("✅ Zależności zainstalowane")
    else:
        print("✅ Wszystkie zależności są zainstalowane")

def start_backend():
    """Start Flask backend server"""
    project_root = Path(__file__).parent.parent.parent
    backend_script = project_root / 'web_dashboard.py'

    if not backend_script.exists():
        print(f"❌ Nie znaleziono pliku backend: {backend_script}")
        return False

    print("🚀 Uruchamiam WIG30 Backend System...")
    print(f"📁 Project root: {project_root}")
    print(f"🌐 Backend będzie dostępny na: http://localhost:5000")
    print("=" * 50)

    try:
        # Change to project directory
        os.chdir(project_root)

        # Start Flask backend
        subprocess.run([sys.executable, str(backend_script)], check=True)

    except KeyboardInterrupt:
        print("\n🛑 Backend zatrzymany przez użytkownika")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Błąd uruchomienia backendu: {e}")
        return False
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd: {e}")
        return False

def main():
    """Main function"""
    print("🎯 WIG30 Backend Startup Script v2.0")
    print("=" * 40)

    # Check dependencies
    check_dependencies()

    # Start backend
    success = start_backend()

    if success:
        print("✅ Backend uruchomiony pomyślnie")
    else:
        print("❌ Nie udało się uruchomić backendu")
        sys.exit(1)

if __name__ == "__main__":
    main()