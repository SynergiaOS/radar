# -*- coding: utf-8 -*-
"""
Ujednoliconiony konfigurator systemu WIG30
Łączy wszystkie konfiguracje w jednym miejscu
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

# Podstawowe ustawienia systemu
SYSTEM_CONFIG = {
    'name': 'WIG30 Trading System',
    'version': '2.0.0',
    'author': 'WIG30 Radar Team',
    'created_at': datetime.now().isoformat(),
    'environment': os.getenv('ENVIRONMENT', 'development')
}

# Konfiguracja indeksów
INDICES_CONFIG = {
    'WIG30': {
        'name': 'WIG30',
        'tickers': [
            'PKN.WA', 'LPP.WA', 'CDR.WA', 'KGH.WA', 'PKO.WA', 'PZU.WA', 'PEO.WA', 'MBK.WA',
            'SPL.WA', 'DNP.WA', 'ALE.WA', 'PGE.WA', 'BDX.WA', 'OPL.WA', 'CPS.WA', 'JSW.WA',
            'CCC.WA', 'KTY.WA', 'KRU.WA', 'XTB.WA', 'MIL.WA', 'ACP.WA', 'TPE.WA', 'ALR.WA',
            'PCO.WA', 'ZAB.WA', 'TXT.WA', 'SNT.WA', 'RBW.WA', '11B.WA'
        ],
        'description': '30 największych spółek na GPW'
    },
    'WIG20': {
        'name': 'WIG20',
        'tickers': [
            'PKN.WA', 'LPP.WA', 'CDR.WA', 'KGH.WA', 'PKO.WA', 'PZU.WA', 'PEO.WA', 'MBK.WA',
            'SPL.WA', 'DNP.WA', 'ALE.WA', 'PGE.WA', 'BDX.WA', 'OPL.WA', 'CPS.WA', 'JSW.WA',
            'CCC.WA', 'KTY.WA', 'KRU.WA', 'XTB.WA'
        ],
        'description': '20 największych spółek na GPW'
    }
}

# Aktywny indeks
ACTIVE_INDEX = os.getenv('ACTIVE_INDEX', 'WIG30')

# Progi analizy fundamentalnej
FUNDAMENTAL_THRESHOLDS = {
    'roe_min': float(os.getenv('ROE_THRESHOLD', 10.0)),
    'pe_max': float(os.getenv('PE_THRESHOLD', 20.0)),
    'pb_max': float(os.getenv('PB_THRESHOLD', 3.0)),
    'debt_to_equity_max': float(os.getenv('DEBT_TO_EQUITY_MAX', 2.0)),
    'current_ratio_min': float(os.getenv('CURRENT_RATIO_MIN', 1.0))
}

# Konfiguracja analizy technicznej
TECHNICAL_CONFIG = {
    'ma_periods': [5, 10, 20, 50],
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2
}

# Konfiguracja Future Score
FUTURE_SCORE_CONFIG = {
    'sectors': {
        'technology': {
            'tickers': ['CDR.WA', 'TXT.WA', '11B.WA'],
            'score': 25,
            'name': 'Technologia/Gaming'
        },
        'banking': {
            'tickers': ['PEO.WA', 'MBK.WA', 'PKO.WA', 'SPL.WA'],
            'score': 20,
            'name': 'Sektor bankowy'
        },
        'ecommerce': {
            'tickers': ['ALE.WA', 'CCC.WA'],
            'score': 22,
            'name': 'E-commerce'
        },
        'energy': {
            'tickers': ['PKN.WA', 'PGE.WA', 'TPE.WA'],
            'score': 18,
            'name': 'Energia'
        }
    },
    'trend_bonus': {
        'uptrend': 15,
        'sideways': 10,
        'downtrend': 0
    },
    'recommendation_bonus': 12,
    'rsi_bonus': 8,
    'revenue_bonus': 5,
    'min_score_threshold': 45
}

# Rekomendacje DM (2025)
DM_RECOMMENDATIONS = {
    'ALE.WA': ['mBank'],
    'ALR.WA': ['mBank'],
    'CCC.WA': ['DM BOŚ'],
    'CDR.WA': ['mBank', 'DM BOŚ'],
    'CPS.WA': ['mBank'],
    'KRU.WA': ['mBank'],
    'LPP.WA': ['mBank', 'DM BOŚ'],
    'MBK.WA': ['DM BOŚ'],
    'MIL.WA': ['mBank'],
    'PKN.WA': ['mBank'],
    'SNT.WA': ['mBank'],
    'TXT.WA': ['mBank', 'DM BOŚ']
}

# Konfiguracja API
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 5000)),
    'debug': os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
    'cors_origins': os.getenv('CORS_ORIGINS', '*').split(',')
}

# Konfiguracja danych zewnętrznych
DATA_SOURCES = {
    'yahoo_finance': {
        'enabled': True,
        'rate_limit': 0.3,
        'timeout': 30
    },
    'stooq': {
        'enabled': True,
        'rate_limit': 0.5
    },
    'cqg_api': {
        'enabled': os.getenv('CQG_API_ENABLED', 'False').lower() == 'true',
        'url': os.getenv('CQG_API_URL', ''),
        'token': os.getenv('CQG_API_TOKEN', '')
    }
}

# Konfiguracja eksportu
EXPORT_CONFIG = {
    'output_dir': 'data/exports',
    'chart_dir': 'data/charts',
    'formats': ['csv', 'json', 'excel'],
    'max_charts': 12,
    'include_technical': True,
    'include_fundamentals': True
}

# Konfiguracja risk management
RISK_CONFIG = {
    'max_position_size': 0.02,  # 2% kapitału na pozycję
    'max_portfolio_heat': 0.20,  # 20% maksymalne zaangażowanie
    'stop_loss_pct': 0.05,  # 5% stop loss
    'take_profit_pct': 0.10,  # 10% take profit
    'rebalance_threshold': 0.05  # 5% próg rebalancingu
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/wig30_system.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

class Config:
    """Główna klasa konfiguracyjna"""

    def __init__(self):
        self.system = SYSTEM_CONFIG
        self.indices = INDICES_CONFIG
        self.active_index = ACTIVE_INDEX
        self.fundamental = FUNDAMENTAL_THRESHOLDS
        self.technical = TECHNICAL_CONFIG
        self.future_score = FUTURE_SCORE_CONFIG
        self.recommendations = DM_RECOMMENDATIONS
        self.api = API_CONFIG
        self.data_sources = DATA_SOURCES
        self.export = EXPORT_CONFIG
        self.risk = RISK_CONFIG
        self.logging = LOGGING_CONFIG

    def get_active_tickers(self) -> List[str]:
        """Pobiera aktywne tickery dla wybranego indeksu"""
        return self.indices[self.active_index]['tickers']

    def get_index_info(self) -> Dict:
        """Pobiera informacje o aktywnym indeksie"""
        return self.indices[self.active_index]

    def is_future_score_stock(self, ticker: str) -> bool:
        """Sprawdza czy spółka należy do sektorów z future score"""
        for sector_config in self.future_score['sectors'].values():
            if ticker in sector_config['tickers']:
                return True
        return False

    def get_sector_bonus(self, ticker: str) -> int:
        """Pobiera punkty za sektor dla danego tickera"""
        for sector_config in self.future_score['sectors'].values():
            if ticker in sector_config['tickers']:
                return sector_config['score']
        return 0

    def get_sector_name(self, ticker: str) -> str:
        """Pobiera nazwę sektora dla danego tickera"""
        for sector_config in self.future_score['sectors'].values():
            if ticker in sector_config['tickers']:
                return sector_config['name']
        return None

    def validate_config(self) -> bool:
        """Waliduje konfigurację"""
        required_dirs = [self.export['output_dir'], self.export['chart_dir']]
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)

        return True

# Globalna instancja konfiguracji
config = Config()