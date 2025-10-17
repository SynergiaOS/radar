#!/usr/bin/env python3
"""
Main entry point for the WIG30 Trading System
Starts the web dashboard with real-time data integration
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def start_dashboard():
    """Start the main web dashboard"""
    try:
        from src.backend.dashboard.web_dashboard import app
        print("🚀 Starting WIG30 Trading Dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8050")
        print("🔌 WebSocket endpoint: ws://localhost:8050/ws")
        print("⏹️  Press Ctrl+C to stop")

        # Run the dashboard
        app.run_server(
            host='0.0.0.0',
            port=8050,
            debug=False,
            use_reloader=False
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install dependencies: pip install -r src/config/requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False

def start_analysis():
    """Start standalone analysis mode"""
    try:
        from src.backend.analysis.dynamic_wig30_analyzer import main
        print("📈 Starting WIG30 Analysis Mode...")
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def start_trading():
    """Start trading bot mode"""
    try:
        from src.backend.trading.wig30_bot import main
        print("🤖 Starting Trading Bot Mode...")
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def show_help():
    """Show available options"""
    print("""
🎯 WIG30 Trading System - Available Modes:

📊 dashboard    - Web dashboard with real-time data (default)
📈 analysis     - Standalone analysis mode
🤖 trading      - Trading bot mode
❓ help         - Show this help

Usage:
    python main.py [mode]

Examples:
    python main.py dashboard
    python main.py analysis
    python main.py trading
    """)

if __name__ == "__main__":
    # Parse command line arguments
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "dashboard"

    if mode in ["dashboard", "-d", "--dashboard"]:
        start_dashboard()
    elif mode in ["analysis", "-a", "--analysis"]:
        start_analysis()
    elif mode in ["trading", "-t", "--trading"]:
        start_trading()
    elif mode in ["help", "-h", "--help", ""]:
        show_help()
    else:
        print(f"❌ Unknown mode: {mode}")
        show_help()