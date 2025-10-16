#!/usr/bin/env python3
"""
Advanced Trading System Launcher
Main application that integrates all components: ML/RL, GUI, Alerts, Real-time data
"""

import tkinter as tk
from tkinter import messagebox
import threading
import asyncio
import time
import json
import os
from datetime import datetime

# Import all system components
from ml_trading_system import AdvancedMLTradingSystem
from trading_gui import TradingSystemGUI
from alert_system import TradingAlertSystem
from realtime_integration import RealTimeDataIntegration
from file_monitor import FileStockMonitor

class AdvancedTradingApp:
    """Main application that integrates all trading system components."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 Advanced ML/RL Trading System")
        self.root.geometry("1500x950")
        self.root.configure(bg='#1e1e1e')

        # Initialize system components
        self.ml_system = None
        self.gui = None
        self.alert_system = None
        self.realtime_system = None
        self.file_monitor = None

        # State variables
        self.systems_running = False
        self.monitoring_active = False

        # Create main interface
        self.create_main_interface()

        # Initialize systems
        self.initialize_systems()

    def create_main_interface(self):
        """Create the main application interface."""
        # Title
        title_frame = tk.Frame(self.root, bg='#1e1e1e')
        title_frame.pack(fill=tk.X, padx=20, pady=10)

        title_label = tk.Label(title_frame,
                               text="🚀 Advanced ML/RL Trading System",
                               font=('Arial', 20, 'bold'),
                               fg='white',
                               bg='#1e1e1e')
        title_label.pack()

        subtitle_label = tk.Label(title_frame,
                                 text="Machine Learning + Reinforcement Learning + Real-time Data",
                                 font=('Arial', 12),
                                 fg='#cccccc',
                                 bg='#1e1e1e')
        subtitle_label.pack()

        # Control panel
        control_frame = tk.Frame(self.root, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        # System status
        status_frame = tk.Frame(control_frame, bg='#2d2d2d')
        status_frame.pack(fill=tk.X, padx=10, pady=10)

        self.status_label = tk.Label(status_frame,
                                    text="🔴 Systems Offline",
                                    font=('Arial', 12, 'bold'),
                                    fg='#ff4444',
                                    bg='#2d2d2d')
        self.status_label.pack(side=tk.LEFT)

        self.timestamp_label = tk.Label(status_frame,
                                       text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                       font=('Arial', 10),
                                       fg='#cccccc',
                                       bg='#2d2d2d')
        self.timestamp_label.pack(side=tk.RIGHT)

        # Main buttons
        buttons_frame = tk.Frame(control_frame, bg='#2d2d2d')
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        # System controls
        sys_frame = tk.LabelFrame(buttons_frame,
                                 text="🤖 System Controls",
                                 font=('Arial', 11, 'bold'),
                                 fg='white',
                                 bg='#2d2d2d')
        sys_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        button_style = {'font': ('Arial', 10),
                        'bg': '#007acc',
                        'fg': 'white',
                        'activebackground': '#005c9f',
                        'relief': tk.RAISED,
                        'bd': 2,
                        'width': 15}

        tk.Button(sys_frame, text="🧠 Train ML/RL",
                 command=self.train_ml_rl_models,
                 **button_style).pack(pady=5)

        tk.Button(sys_frame, text="📊 Run Analysis",
                 command=self.run_complete_analysis,
                 **button_style).pack(pady=5)

        tk.Button(sys_frame, text="📈 Start Monitoring",
                 command=self.start_monitoring,
                 **button_style).pack(pady=5)

        # Monitoring controls
        monitor_frame = tk.LabelFrame(buttons_frame,
                                    text="📡 Monitoring",
                                    font=('Arial', 11, 'bold'),
                                    fg='white',
                                    bg='#2d2d2d')
        monitor_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        tk.Button(monitor_frame, text="📁 File Monitor",
                 command=self.start_file_monitoring,
                 **button_style).pack(pady=5)

        tk.Button(monitor_frame, text="⚡ Real-time Data",
                 command=self.start_realtime_data,
                 **button_style).pack(pady=5)

        tk.Button(monitor_frame, text="🚨 Send Alerts",
                 command=self.send_alerts,
                 **button_style).pack(pady=5)

        # GUI controls
        gui_frame = tk.LabelFrame(buttons_frame,
                                text="🖥️ Interface",
                                font=('Arial', 11, 'bold'),
                                fg='white',
                                bg='#2d2d2d')
        gui_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        tk.Button(gui_frame, text="📊 Open GUI",
                 command=self.open_trading_gui,
                 **button_style).pack(pady=5)

        tk.Button(gui_frame, text="📋 View Results",
                 command=self.view_results,
                 **button_style).pack(pady=5)

        tk.Button(gui_frame, text="💾 Export Data",
                 command=self.export_data,
                 **button_style).pack(pady=5)

        # Log panel
        log_frame = tk.LabelFrame(self.root,
                                 text="📝 System Log",
                                 font=('Arial', 11, 'bold'),
                                 fg='white',
                                 bg='#2d2d2d')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create scrolled text for logs
        from tkinter import scrolledtext
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                  height=10,
                                                  bg='#1a1a1a',
                                                  fg='#00ff00',
                                                  font=('Consolas', 9),
                                                  insertbackground='#00ff00')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = tk.Label(self.root,
                                textvariable=self.progress_var,
                                font=('Arial', 10),
                                fg='white',
                                bg='#1e1e1e')
        progress_label.pack(pady=5)

    def initialize_systems(self):
        """Initialize all system components."""
        self.log("🚀 Initializing Advanced Trading System...")

        try:
            # Initialize ML system
            self.ml_system = AdvancedMLTradingSystem()
            self.log("✅ ML/RL Trading System initialized")

            # Initialize alert system
            self.alert_system = TradingAlertSystem()
            self.log("✅ Alert System initialized")

            # Initialize real-time system
            self.realtime_system = RealTimeDataIntegration()
            self.realtime_system.load_config()
            self.log("✅ Real-time Data System initialized")

            # Initialize file monitor
            self.file_monitor = FileStockMonitor()
            self.log("✅ File Monitor initialized")

            self.log("🎯 All systems initialized successfully!")
            self.update_status("🟢 Systems Ready", 'success')

        except Exception as e:
            self.log(f"❌ Error initializing systems: {str(e)}", 'error')
            self.update_status("🔴 System Error", 'error')

    def log(self, message, level='info'):
        """Add message to log panel."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == 'success':
            prefix = "✅"
            color = '#00ff00'
        elif level == 'warning':
            prefix = "⚠️"
            color = '#ffff00'
        elif level == 'error':
            prefix = "❌"
            color = '#ff4444'
        else:
            prefix = "ℹ️"
            color = '#ffffff'

        log_message = f"[{timestamp}] {prefix} {message}\n"

        self.log_text.insert(tk.END, log_message)
        self.log_text.tag_add(level, f"{self.log_text.index('end - 2c linestart')}",
                             f"{self.log_text.index('end - 1c linestart')}")
        self.log_text.tag_config(level, foreground=color)

        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, status, status_type='info'):
        """Update status label."""
        self.status_label.config(text=status)

        colors = {'success': '#00ff00', 'warning': '#ffff00', 'error': '#ff4444', 'info': '#ffffff'}
        self.status_label.config(fg=colors.get(status_type, '#ffffff'))

    def train_ml_rl_models(self):
        """Train ML and RL models in background."""
        def training_thread():
            try:
                self.log("🧠 Starting ML/RL model training...", 'info')
                self.update_status("🟡 Training Models...", 'warning')
                self.progress_var.set("Training models...")

                if not self.ml_system:
                    self.log("❌ ML system not initialized", 'error')
                    return

                # Load data
                if self.ml_system.load_historical_data():
                    self.log("✅ Historical data loaded", 'success')

                    # Train ML models
                    self.ml_system.train_ml_models()
                    self.log("✅ ML models trained successfully", 'success')

                    # Train RL agent
                    self.ml_system.train_rl_agent(episodes=300)
                    self.log("✅ RL agent trained successfully", 'success')

                    self.ml_system.save_models()

                    self.log("🎯 ML/RL training completed!", 'success')
                    self.update_status("🟢 Models Ready", 'success')
                    self.progress_var.set("Models trained and ready")
                else:
                    self.log("❌ Failed to load training data", 'error')
                    self.update_status("🔴 Training Failed", 'error')
                    self.progress_var.set("Training failed")

            except Exception as e:
                self.log(f"❌ Training error: {str(e)}", 'error')
                self.update_status("🔴 Training Error", 'error')
                self.progress_var.set("Training error")

        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()

    def run_complete_analysis(self):
        """Run complete trading analysis."""
        def analysis_thread():
            try:
                self.log("📊 Starting complete trading analysis...", 'info')
                self.update_status("🟡 Analyzing...", 'warning')
                self.progress_var.set("Running analysis...")

                if not self.ml_system:
                    self.log("❌ ML system not initialized", 'error')
                    return

                # Run analysis
                results = self.ml_system.generate_comprehensive_signals()

                # Save results
                with open('latest_trading_signals.json', 'w') as f:
                    json.dump(results, f, indent=2)

                self.log(f"✅ Analysis complete! {len(results['signals'])} signals generated", 'success')

                # Display top signals
                buy_signals = [s for s in results['signals'] if 'BUY' in s.get('final_action', '')]
                if buy_signals:
                    self.log(f"🟢 Top Buy Signal: {buy_signals[0]['ticker']} @ {buy_signals[0]['current_price']:.2f} PLN", 'success')
                    self.log(f"   Score: {buy_signals[0]['combined_score']:.1f} | ML: {buy_signals[0]['ml_prediction']} | RL: {buy_signals[0]['rl_action']}", 'info')

                self.update_status("🟢 Analysis Complete", 'success')
                self.progress_var.set("Analysis complete")

                # Send alerts if significant signals found
                if buy_signals:
                    self.alert_system.send_trading_alerts(buy_signals[:3])

            except Exception as e:
                self.log(f"❌ Analysis error: {str(e)}", 'error')
                self.update_status("🔴 Analysis Error", 'error')
                self.progress_var.set("Analysis error")

        thread = threading.Thread(target=analysis_thread, daemon=True)
        thread.start()

    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.log("📈 Starting comprehensive monitoring...", 'info')
            self.update_status("🟡 Monitoring Active", 'warning')

            # Start monitoring loop
            self.monitoring_loop()
        else:
            self.monitoring_active = False
            self.log("⏹️ Monitoring stopped", 'warning')
            self.update_status("🟢 Systems Ready", 'success')

    def monitoring_loop(self):
        """Background monitoring loop."""
        if self.monitoring_active:
            try:
                # Run analysis
                self.run_complete_analysis()

                # Schedule next update (every 60 seconds)
                self.root.after(60000, self.monitoring_loop)

            except Exception as e:
                self.log(f"❌ Monitoring error: {str(e)}", 'error')

    def start_file_monitoring(self):
        """Start file-based monitoring."""
        def file_monitor_thread():
            try:
                self.log("📁 Starting file-based monitoring...", 'info')

                if not self.file_monitor:
                    self.file_monitor = FileStockMonitor()

                if self.file_monitor.load_previous_analysis():
                    self.file_monitor.start_monitoring()
                else:
                    self.log("❌ Failed to load file monitor data", 'error')

            except Exception as e:
                self.log(f"❌ File monitor error: {str(e)}", 'error')

        thread = threading.Thread(target=file_monitor_thread, daemon=True)
        thread.start()

    def start_realtime_data(self):
        """Start real-time data collection."""
        def realtime_thread():
            try:
                self.log("⚡ Starting real-time data collection...", 'info')

                if not self.realtime_system:
                    self.realtime_system = RealTimeDataIntegration()

                # Run real-time collection
                asyncio.run(self.realtime_system.start_real_time_collection())

            except Exception as e:
                self.log(f"❌ Real-time data error: {str(e)}", 'error')

        thread = threading.Thread(target=realtime_thread, daemon=True)
        thread.start()

    def send_alerts(self):
        """Send trading alerts."""
        try:
            self.log("🚨 Preparing to send alerts...", 'info')

            if os.path.exists('latest_trading_signals.json'):
                with open('latest_trading_signals.json', 'r') as f:
                    data = json.load(f)

                if 'signals' in data:
                    self.alert_system.send_trading_alerts(data['signals'])
                    self.log(f"✅ Alerts sent for {len(data['signals'])} signals", 'success')
                else:
                    self.log("⚠️ No signals found to alert", 'warning')
            else:
                self.log("⚠️ No signal data available. Run analysis first.", 'warning')

        except Exception as e:
            self.log(f"❌ Alert error: {str(e)}", 'error')

    def open_trading_gui(self):
        """Open the advanced trading GUI."""
        try:
            self.log("🖥️ Opening Advanced Trading GUI...", 'info')

            # Create new window for GUI
            gui_window = tk.Toplevel(self.root)
            gui_window.title("📊 Advanced Trading GUI")
            gui_window.geometry("1400x900")

            # Import and create GUI
            from trading_gui import TradingSystemGUI
            gui = TradingSystemGUI()
            gui.root = gui_window
            gui.run()

            self.log("✅ Trading GUI opened in new window", 'success')

        except Exception as e:
            self.log(f"❌ GUI error: {str(e)}", 'error')
            messagebox.showerror("GUI Error", f"Failed to open GUI: {str(e)}")

    def view_results(self):
        """View analysis results."""
        try:
            self.log("📋 Loading analysis results...", 'info')

            if os.path.exists('latest_trading_signals.json'):
                with open('latest_trading_signals.json', 'r') as f:
                    data = json.load(f)

                # Create results window
                results_window = tk.Toplevel(self.root)
                results_window.title("📊 Trading Analysis Results")
                results_window.geometry("800x600")

                # Display results
                text_widget = tk.Text(results_window, bg='#1a1a1a', fg='white', font=('Consolas', 10))
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Format and display results
                results_text = f"Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                results_text += "=" * 60 + "\n\n"

                if 'signals' in data:
                    results_text += f"Total Signals: {len(data['signals'])}\n\n"

                    for signal in data['signals'][:10]:
                        results_text += f"{signal['ticker']} - {signal['name']}\n"
                        results_text += f"  Price: {signal['current_price']:.2f} PLN\n"
                        results_text += f"  Action: {signal['final_action']}\n"
                        results_text += f"  Score: {signal['combined_score']:.1f}\n"
                        results_text += f"  ML: {signal['ml_prediction']} ({signal['ml_confidence']:.3f})\n"
                        results_text += f"  RL: {signal['rl_action']} ({signal['rl_confidence']:.3f})\n"
                        results_text += "-" * 40 + "\n"

                text_widget.insert('1.0', results_text)
                text_widget.config(state='disabled')

                self.log("✅ Results displayed in new window", 'success')

            else:
                self.log("⚠️ No results found. Run analysis first.", 'warning')

        except Exception as e:
            self.log(f"❌ Results error: {str(e)}", 'error')

    def export_data(self):
        """Export all system data."""
        try:
            self.log("💾 Exporting system data...", 'info')

            # Create export directory
            export_dir = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(export_dir, exist_ok=True)

            # Export ML signals
            if os.path.exists('latest_trading_signals.json'):
                import shutil
                shutil.copy('latest_trading_signals.json', f"{export_dir}/trading_signals.json")

            # Export real-time data
            if self.realtime_system:
                self.realtime_system.export_data(f"{export_dir}/realtime_data.csv")

            # Export alert logs
            if os.path.exists('alerts_log.csv'):
                import shutil
                shutil.copy('alerts_log.csv', f"{export_dir}/alerts_log.csv")

            self.log(f"✅ Data exported to {export_dir}/", 'success')
            messagebox.showinfo("Export Complete", f"Data exported to {export_dir}/")

        except Exception as e:
            self.log(f"❌ Export error: {str(e)}", 'error')
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def update_timestamp(self):
        """Update timestamp display."""
        self.timestamp_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_timestamp)

    def run(self):
        """Start the main application."""
        # Start timestamp updates
        self.update_timestamp()

        # Main loop
        self.root.mainloop()


def main():
    """Main function to run the advanced trading system."""
    print("🚀 ADVANCED TRADING SYSTEM LAUNCHER")
    print("=" * 50)
    print("🤖 Machine Learning + Reinforcement Learning")
    print("📊 Real-time Data + Advanced GUI + Alerts")
    print("=" * 50)

    try:
        app = AdvancedTradingApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Advanced Trading System stopped by user")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()