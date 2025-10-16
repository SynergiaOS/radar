#!/usr/bin/env python3
"""
Advanced Trading System GUI
Professional graphical interface for ML/RL trading system
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import threading
import os
from PIL import Image, ImageTk
import pandas as pd
import numpy as np

class TradingSystemGUI:
    """Advanced GUI for the trading system."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Advanced ML/RL Trading System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')

        # Style configuration
        self.setup_styles()

        # Initialize variables
        self.signals_data = []
        self.monitoring_active = False
        self.selected_stock = None

        # Create GUI components
        self.create_widgets()

        # Load initial data
        self.load_initial_data()

    def setup_styles(self):
        """Setup modern dark theme styles."""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure colors
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'select_bg': '#404040',
            'select_fg': '#ffffff',
            'button_bg': '#007acc',
            'button_fg': '#ffffff',
            'success': '#4caf50',
            'warning': '#ff9800',
            'danger': '#f44336',
            'info': '#2196f3'
        }

        # Configure ttk styles
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('TButton',
                           background=self.colors['button_bg'],
                           foreground=self.colors['button_fg'])
        self.style.map('TButton',
                      background=[('active', '#005c9f')])
        self.style.configure('Header.TLabel',
                           background=self.colors['bg'],
                           foreground=self.colors['fg'],
                           font=('Arial', 16, 'bold'))
        self.style.configure('Success.TLabel',
                           background=self.colors['bg'],
                           foreground=self.colors['success'])
        self.style.configure('Warning.TLabel',
                           background=self.colors['bg'],
                           foreground=self.colors['warning'])
        self.style.configure('Danger.TLabel',
                           background=self.colors['bg'],
                           foreground=self.colors['danger'])

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self.create_header(main_frame)

        # Content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Left panel - Controls and info
        left_panel = ttk.Frame(content_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel - Charts and tables
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create left panel components
        self.create_control_panel(left_panel)
        self.create_info_panel(left_panel)

        # Create right panel components
        self.create_charts_panel(right_panel)
        self.create_signals_table(right_panel)

    def create_header(self, parent):
        """Create header section."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_label = ttk.Label(header_frame,
                              text="🤖 Advanced ML/RL Trading System",
                              style='Header.TLabel')
        title_label.pack(side=tk.LEFT)

        # Status indicator
        self.status_label = ttk.Label(header_frame,
                                    text="● Ready",
                                    style='Success.TLabel')
        self.status_label.pack(side=tk.RIGHT)

        # Timestamp
        self.timestamp_label = ttk.Label(header_frame,
                                       text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.timestamp_label.pack(side=tk.RIGHT, padx=(0, 20))

    def create_control_panel(self, parent):
        """Create control panel."""
        control_frame = ttk.LabelFrame(parent, text="🎛️ Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Analysis buttons
        ttk.Button(control_frame,
                  text="🔍 Run Analysis",
                  command=self.run_analysis,
                  width=20).pack(pady=5)

        ttk.Button(control_frame,
                  text="🤖 Train ML Models",
                  command=self.train_models,
                  width=20).pack(pady=5)

        ttk.Button(control_frame,
                  text="🧠 Train RL Agent",
                  command=self.train_rl_agent,
                  width=20).pack(pady=5)

        ttk.Button(control_frame,
                  text="📊 Update Signals",
                  command=self.update_signals,
                  width=20).pack(pady=5)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Monitoring controls
        self.monitor_button = ttk.Button(control_frame,
                                       text="▶️ Start Monitoring",
                                       command=self.toggle_monitoring,
                                       width=20)
        self.monitor_button.pack(pady=5)

        ttk.Button(control_frame,
                  text="📧 Send Alerts",
                  command=self.send_alerts,
                  width=20).pack(pady=5)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Risk profile selection
        ttk.Label(control_frame, text="Risk Profile:").pack(anchor=tk.W)
        self.risk_profile = tk.StringVar(value="MODERATE")
        risk_frame = ttk.Frame(control_frame)
        risk_frame.pack(fill=tk.X, pady=5)

        profiles = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]
        for profile in profiles:
            ttk.Radiobutton(risk_frame,
                           text=profile,
                           variable=self.risk_profile,
                           value=profile).pack(anchor=tk.W)

        # Refresh interval
        ttk.Label(control_frame, text="Refresh Interval (seconds):").pack(anchor=tk.W, pady=(10, 0))
        self.refresh_interval = tk.StringVar(value="30")
        ttk.Spinbox(control_frame,
                   from_=5,
                   to=300,
                   textvariable=self.refresh_interval,
                   width=18).pack(anchor=tk.W)

    def create_info_panel(self, parent):
        """Create information panel."""
        info_frame = ttk.LabelFrame(parent, text="📊 System Status", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)

        # System stats
        self.info_text = scrolledtext.ScrolledText(info_frame,
                                                  height=15,
                                                  width=40,
                                                  bg='#2d2d2d',
                                                  fg='#ffffff',
                                                  font=('Consolas', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(info_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))

    def create_charts_panel(self, parent):
        """Create charts panel."""
        charts_frame = ttk.LabelFrame(parent, text="📈 Charts & Analysis", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.patch.set_facecolor('#2d2d2d')
        self.ax1.set_facecolor('#2d2d2d')
        self.ax2.set_facecolor('#2d2d2d')

        # Configure chart appearance
        for ax in [self.ax1, self.ax2]:
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        self.ax1.set_title('Stock Prices', color='white', fontsize=12)
        self.ax2.set_title('Signal Strength', color='white', fontsize=12)

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_signals_table(self, parent):
        """Create signals table."""
        table_frame = ttk.LabelFrame(parent, text="🎯 Trading Signals", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview
        columns = ('Ticker', 'Name', 'Price', 'Action', 'ML Score', 'RL Score', 'Combined', 'Confidence')
        self.signals_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)

        # Configure columns
        self.signals_tree.heading('Ticker', text='Ticker')
        self.signals_tree.heading('Name', text='Name')
        self.signals_tree.heading('Price', text='Price (PLN)')
        self.signals_tree.heading('Action', text='Action')
        self.signals_tree.heading('ML Score', text='ML Score')
        self.signals_tree.heading('RL Score', text='RL Score')
        self.signals_tree.heading('Combined', text='Combined')
        self.signals_tree.heading('Confidence', text='Confidence')

        # Column widths
        self.signals_tree.column('Ticker', width=80)
        self.signals_tree.column('Name', width=200)
        self.signals_tree.column('Price', width=100)
        self.signals_tree.column('Action', width=100)
        self.signals_tree.column('ML Score', width=80)
        self.signals_tree.column('RL Score', width=80)
        self.signals_tree.column('Combined', width=100)
        self.signals_tree.column('Confidence', width=100)

        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.signals_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.signals_tree.xview)
        self.signals_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout
        self.signals_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Bind double-click for details
        self.signals_tree.bind('<Double-Button-1>', self.show_stock_details)

    def log_info(self, message, level='info'):
        """Log message to info panel."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == 'success':
            prefix = "✅"
        elif level == 'warning':
            prefix = "⚠️"
        elif level == 'error':
            prefix = "❌"
        else:
            prefix = "ℹ️"

        log_message = f"[{timestamp}] {prefix} {message}\n"

        self.info_text.insert(tk.END, log_message)
        self.info_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, status, status_type='info'):
        """Update status label."""
        self.status_label.config(text=f"● {status}")

        if status_type == 'success':
            self.status_label.config(style='Success.TLabel')
        elif status_type == 'warning':
            self.status_label.config(style='Warning.TLabel')
        elif status_type == 'error':
            self.status_label.config(style='Danger.TLabel')
        else:
            self.status_label.config(style='TLabel')

    def update_timestamp(self):
        """Update timestamp display."""
        self.timestamp_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def load_initial_data(self):
        """Load initial data and display."""
        self.log_info("Loading initial data...")
        self.update_status("Loading...", 'info')

        try:
            # Check if signals file exists
            if os.path.exists('ml_rl_trading_signals.json'):
                with open('ml_rl_trading_signals.json', 'r') as f:
                    self.signals_data = json.load(f)

                if 'signals' in self.signals_data:
                    self.populate_signals_table(self.signals_data['signals'])
                    self.update_charts()
                    self.log_info(f"Loaded {len(self.signals_data['signals'])} signals", 'success')
                    self.update_status("Ready", 'success')
                else:
                    self.log_info("No signals found in data file", 'warning')
                    self.update_status("No data", 'warning')
            else:
                self.log_info("No signals data file found. Run analysis first.", 'warning')
                self.update_status("No data", 'warning')

        except Exception as e:
            self.log_info(f"Error loading data: {str(e)}", 'error')
            self.update_status("Error", 'error')

    def populate_signals_table(self, signals):
        """Populate signals table with data."""
        # Clear existing data
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)

        # Add new data
        for signal in signals:
            action = signal.get('final_action', 'N/A')

            # Color coding for actions
            if 'BUY' in action:
                tags = ('buy',)
            elif 'SELL' in action:
                tags = ('sell',)
            else:
                tags = ('hold',)

            self.signals_tree.insert('', 'end', values=(
                signal.get('ticker', 'N/A'),
                signal.get('name', 'N/A')[:30],
                f"{signal.get('current_price', 0):.2f}",
                action,
                f"{signal.get('ml_confidence', 0):.3f}",
                f"{signal.get('rl_confidence', 0):.3f}",
                f"{signal.get('combined_score', 0):.1f}",
                f"{signal.get('ml_accuracy', 0):.3f}"
            ), tags=tags)

        # Configure tags
        self.signals_tree.tag_configure('buy', foreground='#4caf50')
        self.signals_tree.tag_configure('sell', foreground='#f44336')
        self.signals_tree.tag_configure('hold', foreground='#ff9800')

    def update_charts(self):
        """Update charts with current data."""
        if not self.signals_data or 'signals' not in self.signals_data:
            return

        signals = self.signals_data['signals']

        # Clear charts
        self.ax1.clear()
        self.ax2.clear()

        # Chart 1: Stock prices
        tickers = [s['ticker'] for s in signals[:10]]
        prices = [s.get('current_price', 0) for s in signals[:10]]

        colors = ['#4caf50' if 'BUY' in s.get('final_action', '')
                  else '#f44336' if 'SELL' in s.get('final_action', '')
                  else '#ff9800' for s in signals[:10]]

        bars1 = self.ax1.bar(tickers, prices, color=colors)
        self.ax1.set_title('Stock Prices & Actions', color='white', fontsize=12)
        self.ax1.set_ylabel('Price (PLN)', color='white')
        self.ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, price in zip(bars1, prices):
            height = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{price:.1f}', ha='center', va='bottom', color='white')

        # Chart 2: Signal strength
        combined_scores = [s.get('combined_score', 0) for s in signals[:10]]
        ml_scores = [s.get('signal_strength', 0) for s in signals[:10]]

        x = np.arange(len(tickers))
        width = 0.35

        bars2 = self.ax2.bar(x - width/2, combined_scores, width, label='Combined Score', color='#2196f3')
        bars3 = self.ax2.bar(x + width/2, ml_scores, width, label='ML Score', color='#4caf50')

        self.ax2.set_title('Signal Strength Comparison', color='white', fontsize=12)
        self.ax2.set_ylabel('Score', color='white')
        self.ax2.set_xticks(x)
        self.ax2.set_xticklabels(tickers, rotation=45)
        self.ax2.legend()

        # Add value labels
        for bars in [bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', color='white')

        # Refresh canvas
        self.canvas.draw()

    def run_analysis(self):
        """Run complete analysis in background thread."""
        def analysis_thread():
            try:
                self.log_info("Starting complete analysis...")
                self.update_status("Analyzing...", 'info')
                self.progress.start()

                # Import and run ML system
                from ml_trading_system import AdvancedMLTradingSystem

                system = AdvancedMLTradingSystem()

                if system.load_historical_data():
                    system.train_ml_models()
                    system.train_rl_agent(episodes=200)
                    results = system.generate_comprehensive_signals()

                    # Update data
                    self.signals_data = results
                    self.populate_signals_table(results['signals'])
                    self.update_charts()

                    # Save results
                    with open('ml_rl_trading_signals.json', 'w') as f:
                        json.dump(results, f, indent=2)

                    self.log_info(f"Analysis complete! {len(results['signals'])} signals generated", 'success')
                    self.update_status("Ready", 'success')
                else:
                    self.log_info("Failed to load data for analysis", 'error')
                    self.update_status("Error", 'error')

            except Exception as e:
                self.log_info(f"Analysis failed: {str(e)}", 'error')
                self.update_status("Error", 'error')
            finally:
                self.progress.stop()

        # Run in background thread
        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()

    def train_models(self):
        """Train ML models."""
        self.log_info("Training ML models...")
        # This would trigger model training
        messagebox.showinfo("ML Training", "ML models training initiated. This may take a few minutes.")

    def train_rl_agent(self):
        """Train RL agent."""
        self.log_info("Training RL agent...")
        # This would trigger RL training
        messagebox.showinfo("RL Training", "RL agent training initiated. This may take several minutes.")

    def update_signals(self):
        """Update trading signals."""
        self.log_info("Updating signals...")
        self.update_status("Updating...", 'info')

        try:
            # Refresh data
            self.load_initial_data()
            self.log_info("Signals updated successfully", 'success')
            self.update_status("Ready", 'success')
        except Exception as e:
            self.log_info(f"Failed to update signals: {str(e)}", 'error')
            self.update_status("Error", 'error')

    def toggle_monitoring(self):
        """Toggle monitoring on/off."""
        if self.monitoring_active:
            self.monitoring_active = False
            self.monitor_button.config(text="▶️ Start Monitoring")
            self.log_info("Monitoring stopped")
            self.update_status("Ready", 'success')
        else:
            self.monitoring_active = True
            self.monitor_button.config(text="⏸️ Stop Monitoring")
            self.log_info("Monitoring started")
            self.update_status("Monitoring", 'info')
            # Start monitoring loop
            self.monitoring_loop()

    def monitoring_loop(self):
        """Background monitoring loop."""
        if self.monitoring_active:
            try:
                self.update_signals()
                self.update_timestamp()

                # Schedule next update
                interval = int(self.refresh_interval.get()) * 1000
                self.root.after(interval, self.monitoring_loop)
            except Exception as e:
                self.log_info(f"Monitoring error: {str(e)}", 'error')

    def send_alerts(self):
        """Send email/push alerts."""
        self.log_info("Preparing alerts...")
        messagebox.showinfo("Alerts", "Alert system would send notifications for significant signals.")

    def show_stock_details(self, event):
        """Show detailed information for selected stock."""
        selection = self.signals_tree.selection()
        if not selection:
            return

        item = self.signals_tree.item(selection[0])
        ticker = item['values'][0]

        # Find signal data
        signal = None
        if self.signals_data and 'signals' in self.signals_data:
            for s in self.signals_data['signals']:
                if s.get('ticker') == ticker:
                    signal = s
                    break

        if signal:
            details = f"""
📊 {ticker} - {signal.get('name', 'N/A')}

💰 Current Price: {signal.get('current_price', 0):.2f} PLN
🎯 Final Action: {signal.get('final_action', 'N/A')}
📈 Combined Score: {signal.get('combined_score', 0):.1f}

🤖 ML Analysis:
   Prediction: {signal.get('ml_prediction', 'N/A')}
   Confidence: {signal.get('ml_confidence', 0):.3f}
   Accuracy: {signal.get('ml_accuracy', 0):.3f}

🧠 RL Analysis:
   Action: {signal.get('rl_action', 'N/A')}
   Confidence: {signal.get('rl_confidence', 0):.3f}

📊 Technical:
   RSI: {signal.get('rsi', 0):.1f}
   Volume Ratio: {signal.get('volume_ratio', 0):.2f}
            """
            messagebox.showinfo(f"{ticker} Details", details)

    def run(self):
        """Start the GUI application."""
        # Update timestamp every second
        self.update_timestamp()
        self.root.after(1000, self.update_timestamp)

        # Start main loop
        self.root.mainloop()


def main():
    """Main function to run the trading GUI."""
    app = TradingSystemGUI()
    app.run()


if __name__ == "__main__":
    main()