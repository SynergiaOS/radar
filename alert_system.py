#!/usr/bin/env python3
"""
Advanced Alert System
Email and push notifications for trading signals
"""

import smtplib
import requests
import json
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict
import ssl

class TradingAlertSystem:
    """Advanced alert system for trading notifications."""

    def __init__(self):
        self.config = self.load_config()
        self.email_enabled = self.config.get('email', {}).get('enabled', False)
        self.push_enabled = self.config.get('push', {}).get('enabled', False)
        self.webhook_enabled = self.config.get('webhook', {}).get('enabled', False)

    def load_config(self) -> Dict:
        """Load alert configuration."""
        config_file = 'alert_config.json'

        # Default configuration
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'recipients': []
            },
            'push': {
                'enabled': False,
                'pushbullet_token': '',
                'pushover_user_key': '',
                'pushover_app_key': ''
            },
            'webhook': {
                'enabled': False,
                'discord_webhook': '',
                'slack_webhook': '',
                'teams_webhook': ''
            },
            'filters': {
                'min_signal_strength': 50,
                'only_buy_signals': False,
                'only_top_n': 5
            }
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in user_config:
                        user_config[key] = default_config[key]
                    elif isinstance(default_config[key], dict):
                        for subkey in default_config[key]:
                            if subkey not in user_config[key]:
                                user_config[key][subkey] = default_config[key][subkey]
                return user_config
            except:
                pass

        # Create default config file
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)

        print(f"📝 Created default config file: {config_file}")
        return default_config

    def send_email_alert(self, subject: str, body: str):
        """Send email alert."""
        if not self.email_enabled:
            print("⚠️  Email alerts disabled")
            return False

        try:
            email_config = self.config['email']
            sender_email = email_config['sender_email']
            sender_password = email_config['sender_password']
            recipients = email_config['recipients']

            if not sender_email or not sender_password or not recipients:
                print("❌ Email configuration incomplete")
                return False

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject

            # Add body
            msg.attach(MIMEText(body, 'html'))

            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipients, msg.as_string())

            print(f"✅ Email alert sent to {len(recipients)} recipients")
            return True

        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}")
            return False

    def send_pushbullet_notification(self, title: str, message: str):
        """Send Pushbullet notification."""
        if not self.push_enabled:
            return False

        try:
            token = self.config['push'].get('pushbullet_token')
            if not token:
                return False

            url = "https://api.pushbullet.com/v2/pushes"
            headers = {
                "Access-Token": token,
                "Content-Type": "application/json"
            }
            data = {
                "type": "note",
                "title": title,
                "body": message
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print("✅ Pushbullet notification sent")
                return True
            else:
                print(f"❌ Pushbullet error: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Pushbullet error: {str(e)}")
            return False

    def send_pushover_notification(self, title: str, message: str):
        """Send Pushover notification."""
        if not self.push_enabled:
            return False

        try:
            user_key = self.config['push'].get('pushover_user_key')
            app_key = self.config['push'].get('pushover_app_key')

            if not user_key or not app_key:
                return False

            url = "https://api.pushover.net/1/messages.json"
            data = {
                "user": user_key,
                "token": app_key,
                "title": title,
                "message": message,
                "priority": 1
            }

            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("✅ Pushover notification sent")
                return True
            else:
                print(f"❌ Pushover error: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Pushover error: {str(e)}")
            return False

    def send_discord_webhook(self, content: str, embed_data: Dict = None):
        """Send Discord webhook notification."""
        if not self.webhook_enabled:
            return False

        try:
            webhook_url = self.config['webhook'].get('discord_webhook')
            if not webhook_url:
                return False

            data = {"content": content}

            if embed_data:
                data["embeds"] = [embed_data]

            response = requests.post(webhook_url, json=data)
            if response.status_code == 204:
                print("✅ Discord notification sent")
                return True
            else:
                print(f"❌ Discord error: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Discord error: {str(e)}")
            return False

    def send_slack_webhook(self, text: str, blocks: List = None):
        """Send Slack webhook notification."""
        if not self.webhook_enabled:
            return False

        try:
            webhook_url = self.config['webhook'].get('slack_webhook')
            if not webhook_url:
                return False

            data = {"text": text}

            if blocks:
                data["blocks"] = blocks

            response = requests.post(webhook_url, json=data)
            if response.status_code == 200:
                print("✅ Slack notification sent")
                return True
            else:
                print(f"❌ Slack error: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Slack error: {str(e)}")
            return False

    def format_email_alert(self, signals: List[Dict]) -> str:
        """Format signals into HTML email."""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .signal { background-color: white; margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }
                .buy-signal { border-left-color: #27ae60; }
                .sell-signal { border-left-color: #e74c3c; }
                .hold-signal { border-left-color: #f39c12; }
                .ticker { font-size: 18px; font-weight: bold; margin-bottom: 5px; }
                .details { color: #555; margin: 5px 0; }
                .score { font-weight: bold; font-size: 14px; }
                .high-score { color: #27ae60; }
                .medium-score { color: #f39c12; }
                .low-score { color: #e74c3c; }
                .footer { margin-top: 20px; color: #7f8c8d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 ML/RL Trading Signals Alert</h1>
                <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        """

        for signal in signals:
            action = signal.get('final_action', 'HOLD')
            score = signal.get('combined_score', 0)
            score_class = 'high-score' if score > 60 else 'medium-score' if score > 30 else 'low-score'
            signal_class = f"{action.lower().replace(' ', '-')}-signal"

            html += f"""
            <div class="signal {signal_class}">
                <div class="ticker">{signal.get('ticker', 'N/A')} - {signal.get('name', 'N/A')}</div>
                <div class="details">
                    💰 Price: {signal.get('current_price', 0):.2f} PLN |
                    🎯 Action: {action} |
                    <span class="score {score_class}">Score: {score:.1f}</span>
                </div>
                <div class="details">
                    🤖 ML: {signal.get('ml_prediction', 'N/A')} ({signal.get('ml_confidence', 0):.3f}) |
                    🧠 RL: {signal.get('rl_action', 'N/A')} ({signal.get('rl_confidence', 0):.3f})
                </div>
            </div>
            """

        html += """
            <div class="footer">
                <p>⚠️ This is an automated alert. Please verify information before making trading decisions.</p>
                <p>📊 Advanced ML/RL Trading System | Not financial advice</p>
            </div>
        </body>
        </html>
        """

        return html

    def format_discord_embed(self, signal: Dict) -> Dict:
        """Format signal as Discord embed."""
        action = signal.get('final_action', 'HOLD')
        score = signal.get('combined_score', 0)

        # Color based on action
        if 'BUY' in action:
            color = 0x27ae60  # Green
        elif 'SELL' in action:
            color = 0xe74c3c  # Red
        else:
            color = 0xf39c12  # Orange

        embed = {
            "title": f"🤖 {signal.get('ticker', 'N/A')} - {signal.get('name', 'N/A')}",
            "description": f"**Price:** {signal.get('current_price', 0):.2f} PLN\n"
                          f"**Action:** {action}\n"
                          f"**Score:** {score:.1f}\n"
                          f"**ML Prediction:** {signal.get('ml_prediction', 'N/A')} ({signal.get('ml_confidence', 0):.3f})\n"
                          f"**RL Action:** {signal.get('rl_action', 'N/A')} ({signal.get('rl_confidence', 0):.3f})",
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "Advanced ML/RL Trading System"
            }
        }

        return embed

    def filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter signals based on configuration."""
        filters = self.config.get('filters', {})

        filtered = signals.copy()

        # Filter by signal strength
        min_strength = filters.get('min_signal_strength', 50)
        filtered = [s for s in filtered if s.get('combined_score', 0) >= min_strength]

        # Filter by action type
        if filters.get('only_buy_signals', False):
            filtered = [s for s in filtered if 'BUY' in s.get('final_action', '')]

        # Limit to top N
        top_n = filters.get('only_top_n', 5)
        filtered = sorted(filtered, key=lambda x: x.get('combined_score', 0), reverse=True)[:top_n]

        return filtered

    def send_trading_alerts(self, signals: List[Dict]):
        """Send alerts for trading signals."""
        if not signals:
            print("⚠️  No signals to alert")
            return

        # Apply filters
        filtered_signals = self.filter_signals(signals)

        if not filtered_signals:
            print("⚠️  No signals pass filter criteria")
            return

        print(f"🚨 Sending alerts for {len(filtered_signals)} filtered signals")

        # Prepare email
        email_subject = f"🤖 Trading Signals Alert - {len(filtered_signals)} Opportunities"
        email_body = self.format_email_alert(filtered_signals)

        # Send email
        if self.email_enabled:
            self.send_email_alert(email_subject, email_body)

        # Send push notifications
        if self.push_enabled:
            for signal in filtered_signals[:3]:  # Top 3 signals
                title = f"{signal.get('ticker')} - {signal.get('final_action')}"
                message = f"Price: {signal.get('current_price', 0):.2f} PLN | Score: {signal.get('combined_score', 0):.1f}"

                self.send_pushbullet_notification(title, message)
                self.send_pushover_notification(title, message)

        # Send Discord notifications
        if self.webhook_enabled:
            for signal in filtered_signals:
                embed = self.format_discord_embed(signal)
                self.send_discord_webhook(f"🎯 **{signal.get('final_action')} Signal**", embed)

        # Log alerts
        self.log_alerts(filtered_signals)

    def log_alerts(self, signals: List[Dict]):
        """Log sent alerts to file."""
        try:
            log_file = 'alerts_log.csv'

            # Check if file exists
            file_exists = os.path.exists(log_file)

            with open(log_file, 'a', newline='') as f:
                import csv
                writer = csv.writer(f)

                # Write header if new file
                if not file_exists:
                    writer.writerow(['timestamp', 'ticker', 'action', 'price', 'score', 'ml_confidence', 'rl_confidence'])

                # Write alerts
                for signal in signals:
                    writer.writerow([
                        datetime.now().isoformat(),
                        signal.get('ticker'),
                        signal.get('final_action'),
                        signal.get('current_price'),
                        signal.get('combined_score'),
                        signal.get('ml_confidence'),
                        signal.get('rl_confidence')
                    ])

            print(f"💾 Alerts logged to {log_file}")

        except Exception as e:
            print(f"❌ Error logging alerts: {str(e)}")

    def test_alert_system(self):
        """Test the alert system."""
        print("🧪 Testing alert system...")

        test_signal = {
            'ticker': 'TEST.WA',
            'name': 'Test Company',
            'current_price': 100.50,
            'final_action': 'BUY',
            'combined_score': 85.5,
            'ml_prediction': 'UP',
            'ml_confidence': 0.85,
            'rl_action': 'BUY',
            'rl_confidence': 0.90
        }

        self.send_trading_alerts([test_signal])
        print("✅ Alert system test completed")


def main():
    """Main function to test alert system."""
    print("🚨 TRADING ALERT SYSTEM")
    print("=" * 40)

    alert_system = TradingAlertSystem()

    # Load and display configuration
    config = alert_system.config
    print(f"📧 Email alerts: {'✅ Enabled' if config['email']['enabled'] else '❌ Disabled'}")
    print(f"📱 Push alerts: {'✅ Enabled' if config['push']['enabled'] else '❌ Disabled'}")
    print(f"🔗 Webhook alerts: {'✅ Enabled' if config['webhook']['enabled'] else '❌ Disabled'}")

    # Test with sample signals if available
    if os.path.exists('ml_rl_trading_signals.json'):
        print("\n📊 Loading signals from file...")
        with open('ml_rl_trading_signals.json', 'r') as f:
            data = json.load(f)

        if 'signals' in data:
            print(f"Found {len(data['signals'])} signals")
            alert_system.send_trading_alerts(data['signals'])
        else:
            print("No signals found in file")
    else:
        print("\n🧪 Running test alert...")
        alert_system.test_alert_system()

    print(f"\n📝 Configure alerts in: alert_config.json")
    print("🏁 Alert system test complete!")


if __name__ == "__main__":
    main()