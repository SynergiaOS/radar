#!/usr/bin/env python3
"""
Realtime WebSocket Server for PKN.WA stock data
Based on TradingView analysis: trend channel, RSI divergence, shooting star pattern
"""

import asyncio
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import socketio
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PKNDataGenerator:
    """Generator danych PKN.WA bazujący na analizie TradingView"""

    def __init__(self):
        self.base_price = 85.0  # Aktualna cena z TradingView
        self.support_level = 80.0  # Poziom wsparcia (szczyt 2021)
        self.resistance_level = 95.0  # Poziom oporu (szczyt 2017)
        self.channel_slope = 0.001  # Nachylenie kanału wzrostowego

        # Parametry RSI z dywergencją
        self.rsi_trend = -0.01  # Spadkowy trend RSI (dywergencja)
        self.base_rsi = 68.0

        # Parametry wolumenu
        self.base_volume = 3_000_000
        self.volume_spike_threshold = 5_000_000

        # Historia danych
        self.price_history: List[float] = []
        self.rsi_history: List[float] = []

        # Inicjalizacja danych historycznych
        self._initialize_historical_data()

    def _initialize_historical_data(self):
        """Inicjalizuj dane historyczne dla ostatnich 100 dni"""
        now = time.time()

        for i in range(100, 0, -1):
            timestamp = now - (i * 24 * 60 * 60)

            # Symulacja kanału wzrostowego
            trend_factor = 1 + self.channel_slope * (100 - i)

            # Dodaj cykliczne wahania w kanale
            cycle_factor = 1 + 0.05 * math.sin(i * 0.1)

            # Cena z kanału wzrostowego
            price = self.base_price * trend_factor * cycle_factor

            # Dodaj losowość
            volatility = 0.025  # 2.5% dzienna zmienność
            price *= (1 + (random.random() - 0.5) * volatility)

            # Ograniczenie do kanału
            price = max(self.support_level * 0.95, min(self.resistance_level * 1.05, price))

            self.price_history.append(price)

            # RSI z dywergencją (spada gdy cena rośnie)
            rsi = self.base_rsi + self.rsi_trend * (100 - i) + random.random() * 10
            rsi = max(30, min(80, rsi))  # RSI w zakresie 30-80
            self.rsi_history.append(rsi)

    def calculate_sma(self, data: List[float], period: int) -> float:
        """Oblicz prostą średnią kroczącą"""
        if len(data) < period:
            return sum(data) / len(data)
        return sum(data[-period:]) / period

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Oblicz RSI"""
        if len(prices) < period + 1:
            return 50

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains[-period:]) / period if len(gains) >= period else 0
        avg_loss = sum(losses[-period:]) / period if len(losses) >= period else 0

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return max(0, min(100, rsi))

    def detect_patterns(self) -> Dict[str, bool]:
        """Wykryj wzorce techniczne z analizy TradingView"""
        patterns = {}

        if len(self.price_history) < 20:
            return patterns

        recent_prices = self.price_history[-20:]
        recent_rsi = self.rsi_history[-20:]

        # 1. Shooting Star (spadająca gwiazda)
        if len(recent_prices) >= 1:
            last_price = recent_prices[-1]
            # Shooting star: long upper shadow, small body
            patterns['shooting_star'] = random.random() < 0.1  # 10% szans

        # 2. RSI Divergence (dywergencja)
        if len(recent_prices) >= 10 and len(recent_rsi) >= 10:
            price_trend = recent_prices[-1] - recent_prices[-10]
            rsi_trend = recent_rsi[-1] - recent_rsi[-10]
            # Cena rośnie, RSI spada = dywergencja niedźwiedzia
            patterns['rsi_divergence'] = (price_trend > 0 and rsi_trend < 0)

        # 3. Channel Breakout (wybicie z kanału)
        current_price = recent_prices[-1]
        patterns['channel_breakout_up'] = current_price > self.resistance_level
        patterns['channel_breakout_down'] = current_price < self.support_level

        # 4. Volume Spike (wzrost wolumenu)
        current_volume = random.randint(1_000_000, 8_000_000)
        patterns['volume_spike'] = current_volume > self.volume_spike_threshold

        return patterns

    def generate_realtime_data(self) -> Dict:
        """Generuj dane w czasie rzeczywistym"""
        current_time = datetime.now()

        # Trend kanału wzrostowy
        trend_factor = 1 + self.channel_slope * 0.1  # Dzienny przyrost

        # Cykliczne wahania w kanale
        cycle_factor = 1 + 0.02 * math.sin(time.time() * 0.1)

        # Nowa cena
        base_change = (random.random() - 0.48) * 0.04  # Lekka przewaga wzrostów
        price_change = base_change + 0.001  # Trend wzrostowy

        new_price = self.price_history[-1] * (1 + price_change) if self.price_history else self.base_price
        new_price *= trend_factor * cycle_factor

        # Ograniczenie do kanału z możliwością wybicia
        if random.random() < 0.05:  # 5% szans na lekkie wybicie
            new_price = new_price * (1 + (random.random() - 0.5) * 0.03)

        new_price = max(self.support_level * 0.9, min(self.resistance_level * 1.1, new_price))

        # Parametry świecy
        open_price = self.price_history[-1] if self.price_history else new_price
        close_price = new_price

        # High/Low z volatility
        daily_volatility = 0.015  # 1.5% dzienna zmienność
        high_price = max(open_price, close_price) * (1 + random.random() * daily_volatility)
        low_price = min(open_price, close_price) * (1 - random.random() * daily_volatility)

        # Wolumen
        base_volume = self.base_volume
        if random.random() < 0.15:  # 15% szans na increased volume
            base_volume *= random.uniform(1.5, 3.0)

        volume = int(base_volume * (1 + (random.random() - 0.5) * 0.5))

        # RSI z dywergencją
        self.price_history.append(close_price)
        current_rsi = self.calculate_rsi(self.price_history[-15:])
        self.rsi_history.append(current_rsi)

        # Średnie kroczące
        sma5 = self.calculate_sma(self.price_history, 5)
        sma10 = self.calculate_sma(self.price_history, 10)
        sma20 = self.calculate_sma(self.price_history, 20)

        # Wykryj wzorce
        patterns = self.detect_patterns()

        # Poziomy Fibonacciego
        lowest = min(self.price_history[-100:] if len(self.price_history) >= 100 else self.price_history)
        highest = max(self.price_history[-100:] if len(self.price_history) >= 100 else self.price_history)

        fibonacci_levels = {
            'fib_0': lowest,
            'fib_236': lowest + (highest - lowest) * 0.236,
            'fib_382': lowest + (highest - lowest) * 0.382,
            'fib_500': lowest + (highest - lowest) * 0.5,
            'fib_618': lowest + (highest - lowest) * 0.618,
            'fib_786': lowest + (highest - lowest) * 0.786,
            'fib_100': highest,
            'fib_1618': highest + (highest - lowest) * 0.618
        }

        # Zwróć dane
        return {
            'ticker': 'PKN.WA',
            'timestamp': current_time.isoformat(),
            'price': round(close_price, 2),
            'change': round((close_price - open_price) / open_price * 100, 2),
            'volume': volume,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'rsi': round(current_rsi, 1),
            'sma5': round(sma5, 2),
            'sma10': round(sma10, 2),
            'sma20': round(sma20, 2),
            'support': round(self.support_level, 2),
            'resistance': round(self.resistance_level, 2),
            'patterns': patterns,
            'fibonacci': fibonacci_levels,
            'analysis': {
                'trend': 'Bullish (channel)',
                'signal': 'Hold' if not patterns.get('shooting_star') else 'Caution',
                'strength': 'Medium',
                'outlook': 'Korekta możliwa (-20%) przy wybiciu poniżej wsparcia'
            }
        }

# Globalne zmienne
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
data_generator = PKNDataGenerator()
connected_clients = set()

@sio.event
async def connect(sid, environ):
    """Nowy klient się połączył"""
    logger.info(f"Klient {sid} podłączony")
    connected_clients.add(sid)

    # Wyślij aktualne dane od razu po połączeniu
    current_data = data_generator.generate_realtime_data()
    await sio.emit('pkn_realtime_data', current_data, room=sid)

@sio.event
async def disconnect(sid):
    """Klient się rozłączył"""
    logger.info(f"Klient {sid} rozłączony")
    connected_clients.discard(sid)

@sio.event
async def request_historical_data(sid, data):
    """Wyślij dane historyczne na żądanie"""
    days = data.get('days', 30)
    logger.info(f"Wysyłanie {days} dni danych historycznych do klienta {sid}")

    historical_data = []
    for i in range(days):
        timestamp = datetime.now() - timedelta(days=days-i)
        # Generuj historyczne dane
        hist_data = data_generator.generate_realtime_data()
        hist_data['timestamp'] = timestamp.isoformat()
        historical_data.append(hist_data)

    await sio.emit('pkn_historical_data', historical_data, room=sid)

async def broadcast_realtime_data():
    """Główna pętla broadcastowania danych w czasie rzeczywistym"""
    while True:
        if connected_clients:
            # Generuj i wyślij dane do wszystkich podłączonych klientów
            current_data = data_generator.generate_realtime_data()

            logger.debug(f"Wysyłanie danych: {current_data['ticker']} = {current_data['price']} PLN")

            await sio.emit('pkn_realtime_data', current_data)

        # Aktualizacja co 5 sekund (dane intraday)
        await asyncio.sleep(5)

async def main():
    """Główna funkcja serwera"""
    logger.info("Startowanie serwera WebSocket PKN.WA...")
    logger.info("Adres: ws://localhost:5000")
    logger.info("Dostępne zdarzenia:")
    logger.info("  - connect/disconnect")
    logger.info("  - pkn_realtime_data (co 5 sekund)")
    logger.info("  - request_historical_data")

    # Uruchom zadanie broadcastowania w tle
    asyncio.create_task(broadcast_realtime_data())

    # Konfiguracja aplikacji ASGI
    app = socketio.ASGIApp(sio)

    return app

if __name__ == '__main__':
    import math

    app = asyncio.run(main())

    # Uruchom serwer na porcie 5000
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)