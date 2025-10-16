#!/usr/bin/env python3
"""
xtB Integration Module
Real-time API integration for Polish market data from xStation 5 API
Provides WebSocket streaming, real-time quotes, and order management
"""

import asyncio
import websockets
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import XTB_API_URL, XTB_LOGIN, XTB_PASSWORD, GPW_TICKERS
from trading_chart_service import chart_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class XTBTicker:
    """xtB ticker information"""
    symbol: str
    description: str
    category_name: str
    min_lot: float
    max_lot: float
    lot_step: float
    min_trailing_distance: float
    spread_raw: float
    spread_table: float
    high: float
    low: float
    change_pct: float
    volume: int
    time: datetime

@dataclass
class XTBQuote:
    """Real-time quote from xtB"""
    symbol: str
    ask: float
    bid: float
    ask_volume: int
    bid_volume: int
    high: float
    low: float
    change_pct: float
    volume: int
    time: datetime

@dataclass
class XTBCandle:
    """Candle data from xtB"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    period: str  # 'm1', 'm5', 'm15', 'h1', 'd1', etc.

class XTBClient:
    """xtB xStation 5 API client"""

    def __init__(self, demo: bool = True):
        self.demo = demo
        self.base_url = XTB_API_URL
        self.login = XTB_LOGIN
        self.password = XTB_PASSWORD
        self.session_id = None
        self.websocket = None
        self.stream_task = None
        self.quote_callbacks: List[Callable[[XTBQuote], None]] = []
        self.candle_callbacks: List[Callable[[XTBCandle], None]] = []
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_heartbeat = None

        # Cache for ticker data
        self.tickers_cache: Dict[str, XTBTicker] = {}
        self.quotes_cache: Dict[str, XTBQuote] = {}
        self.candles_cache: Dict[str, pd.DataFrame] = {}

    async def connect(self) -> bool:
        """
        Connect to xtB API

        Returns:
            True if connection successful
        """
        try:
            ws_url = f"wss://{self.base_url}/websocket"
            self.websocket = await websockets.connect(ws_url)

            # Login
            login_payload = {
                "command": "login",
                "arguments": {
                    "userId": self.login,
                    "password": self.password,
                    "appName": "RadarTradingSystem"
                }
            }

            await self.websocket.send(json.dumps(login_payload))
            response = await self.websocket.recv()
            result = json.loads(response)

            if result.get('status') == True:
                self.session_id = result.get('streamSessionId')
                self.is_connected = True
                self.reconnect_attempts = 0
                self.last_heartbeat = datetime.now()

                # Start streaming task
                self.stream_task = asyncio.create_task(self._stream_messages())

                logger.info(f"Successfully connected to xtB API (Demo: {self.demo})")
                return True
            else:
                logger.error(f"Login failed: {result.get('errorDescr', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from xtB API"""
        try:
            self.is_connected = False

            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass

            if self.websocket:
                await self.websocket.close()

            logger.info("Disconnected from xtB API")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def _send_command(self, command: str, arguments: Dict = None) -> Dict:
        """Send command to xtB API"""
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to xtB API")

        payload = {
            "command": command,
            "arguments": arguments or {}
        }

        try:
            await self.websocket.send(json.dumps(payload))
            response = await self.websocket.recv()
            result = json.loads(response)

            if result.get('status') != True:
                error_msg = result.get('errorDescr', 'Unknown error')
                logger.error(f"Command {command} failed: {error_msg}")
                raise Exception(f"xtB API error: {error_msg}")

            return result.get('returnData', {})

        except Exception as e:
            logger.error(f"Error sending command {command}: {e}")
            raise

    async def _stream_messages(self):
        """Handle streaming messages"""
        try:
            while self.is_connected and self.websocket:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=60.0  # Timeout for heartbeat
                )

                self.last_heartbeat = datetime.now()
                data = json.loads(message)

                # Handle different message types
                if 'command' in data:
                    if data['command'] == 'tickPrices':
                        await self._handle_quote(data)
                    elif data['command'] == 'candlePrices':
                        await self._handle_candle(data)
                    elif data['command'] == 'ping':
                        await self._handle_ping(data)
                    else:
                        logger.debug(f"Unknown command: {data['command']}")

        except asyncio.TimeoutError:
            logger.warning("WebSocket timeout - attempting reconnection")
            await self._reconnect()
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed - attempting reconnection")
            await self._reconnect()
        except Exception as e:
            logger.error(f"Error in stream handler: {e}")
            await self._reconnect()

    async def _handle_quote(self, data: Dict):
        """Handle real-time quote data"""
        try:
            quote_data = data['data'][0] if data.get('data') else None
            if not quote_data:
                return

            quote = XTBQuote(
                symbol=quote_data['symbol'],
                ask=float(quote_data['ask']),
                bid=float(quote_data['bid']),
                ask_volume=int(quote_data['askVolume']),
                bid_volume=int(quote_data['bidVolume']),
                high=float(quote_data['high']),
                low=float(quote_data['low']),
                change_pct=float(quote_data['percentageChange']),
                volume=int(quote_data['ex']),
                time=datetime.fromtimestamp(quote_data['timestamp'] / 1000)
            )

            # Update cache
            self.quotes_cache[quote.symbol] = quote

            # Notify callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(quote)
                except Exception as e:
                    logger.error(f"Error in quote callback: {e}")

        except Exception as e:
            logger.error(f"Error handling quote: {e}")

    async def _handle_candle(self, data: Dict):
        """Handle candle data"""
        try:
            candle_data = data['data'][0] if data.get('data') else None
            if not candle_data:
                return

            candle = XTBCandle(
                symbol=candle_data['symbol'],
                open=float(candle_data['open']),
                high=float(candle_data['high']),
                low=float(candle_data['low']),
                close=float(candle_data['close']),
                volume=int(candle_data['vol']),
                time=datetime.fromtimestamp(candle_data['timestamp'] / 1000),
                period=candle_data['period']
            )

            # Update cache
            if candle.symbol not in self.candles_cache:
                self.candles_cache[candle.symbol] = pd.DataFrame()

            # Add new candle to dataframe
            new_row = pd.DataFrame([{
                'time': candle.time,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            }])

            self.candles_cache[candle.symbol] = pd.concat([
                self.candles_cache[candle.symbol],
                new_row
            ]).drop_duplicates(subset=['time']).sort_values('time')

            # Notify callbacks
            for callback in self.candle_callbacks:
                try:
                    callback(candle)
                except Exception as e:
                    logger.error(f"Error in candle callback: {e}")

        except Exception as e:
            logger.error(f"Error handling candle: {e}")

    async def _handle_ping(self, data: Dict):
        """Handle ping message"""
        # Respond with pong to keep connection alive
        try:
            pong_payload = {"command": "pong"}
            await self.websocket.send(json.dumps(pong_payload))
        except Exception as e:
            logger.error(f"Error sending pong: {e}")

    async def _reconnect(self):
        """Attempt to reconnect"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.is_connected = False
            return

        self.reconnect_attempts += 1
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")

        await asyncio.sleep(min(2 ** self.reconnect_attempts, 30))  # Exponential backoff

        try:
            if self.websocket:
                await self.websocket.close()

            await self.connect()

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def get_all_symbols(self) -> List[XTBTicker]:
        """Get all available symbols"""
        try:
            symbols_data = await self._send_command("getAllSymbols")

            tickers = []
            for symbol_data in symbols_data:
                if symbol_data.get('currency') == 'PLN':  # Focus on Polish stocks
                    ticker = XTBTicker(
                        symbol=symbol_data['symbol'],
                        description=symbol_data['description'],
                        category_name=symbol_data['categoryName'],
                        min_lot=float(symbol_data['lotMin']),
                        max_lot=float(symbol_data['lotMax']),
                        lot_step=float(symbol_data['lotStep']),
                        min_trailing_distance=float(symbol_data['trailingStep']),
                        spread_raw=float(symbol_data['spreadRaw']),
                        spread_table=float(symbol_data['spreadTable']),
                        high=float(symbol_data['high']),
                        low=float(symbol_data['low']),
                        change_pct=float(symbol_data['percentageChange']),
                        volume=int(symbol_data['ex']),
                        time=datetime.now()
                    )
                    tickers.append(ticker)
                    self.tickers_cache[ticker.symbol] = ticker

            logger.info(f"Retrieved {len(tickers)} Polish stocks")
            return tickers

        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []

    async def subscribe_quotes(self, symbols: List[str]):
        """Subscribe to real-time quotes for symbols"""
        try:
            for symbol in symbols:
                payload = {
                    "command": "subscribePrices",
                    "arguments": {
                        "symbol": symbol
                    }
                }
                await self.websocket.send(json.dumps(payload))

            logger.info(f"Subscribed to quotes for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Error subscribing to quotes: {e}")

    async def subscribe_candles(self, symbol: str, period: str = 'm5'):
        """Subscribe to candle data for a symbol"""
        try:
            payload = {
                "command": "subscribeCandles",
                "arguments": {
                    "symbol": symbol,
                    "period": period
                }
            }
            await self.websocket.send(json.dumps(payload))

            logger.info(f"Subscribed to {period} candles for {symbol}")

        except Exception as e:
            logger.error(f"Error subscribing to candles: {e}")

    async def get_historical_data(self, symbol: str, period: str,
                                 start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical candle data"""
        try:
            payload = {
                "command": "getChartLastRequest",
                "arguments": {
                    "symbol": symbol,
                    "period": period,
                    "start": int(start.timestamp() * 1000),
                    "end": int(end.timestamp() * 1000),
                    "ticks": 0
                }
            }

            data = await self._send_command("getChartLastRequest", payload['arguments'])

            if not data.get('rateInfos'):
                return pd.DataFrame()

            candles = []
            for rate_info in data['rateInfos']:
                candle = {
                    'time': datetime.fromtimestamp(rate_info['timestamp'] / 1000),
                    'open': rate_info['open'],
                    'high': rate_info['high'],
                    'low': rate_info['low'],
                    'close': rate_info['close'],
                    'volume': rate_info.get('vol', 0)
                }
                candles.append(candle)

            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def add_quote_callback(self, callback: Callable[[XTBQuote], None]):
        """Add callback for real-time quotes"""
        self.quote_callbacks.append(callback)

    def add_candle_callback(self, callback: Callable[[XTBCandle], None]):
        """Add callback for candle data"""
        self.candle_callbacks.append(callback)

    def get_latest_quote(self, symbol: str) -> Optional[XTBQuote]:
        """Get latest quote from cache"""
        return self.quotes_cache.get(symbol)

    def get_candles_df(self, symbol: str) -> pd.DataFrame:
        """Get candle data from cache"""
        return self.candles_cache.get(symbol, pd.DataFrame())

    def get_ticker_info(self, symbol: str) -> Optional[XTBTicker]:
        """Get ticker information from cache"""
        return self.tickers_cache.get(symbol)


class XTBDataManager:
    """Manager for xtB data operations"""

    def __init__(self):
        self.client = XTBClient(demo=True)
        self.subscribed_symbols: List[str] = []
        self.data_callbacks: Dict[str, List[Callable]] = {}
        self.gpw_symbols = GPW_TICKERS

    async def initialize(self) -> bool:
        """Initialize xtB connection and subscribe to GPW symbols"""
        try:
            # Connect to xtB
            if not await self.client.connect():
                return False

            # Get available symbols and filter for GPW
            all_symbols = await self.client.get_all_symbols()

            # Subscribe to GPW symbols
            await self.subscribe_to_gpw_symbols()

            logger.info("xtB data manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing xtB data manager: {e}")
            return False

    async def subscribe_to_gpw_symbols(self):
        """Subscribe to all GPW symbols"""
        try:
            # Get available symbols and match with our GPW list
            available_symbols = list(self.client.tickers_cache.keys())

            # Subscribe to quotes for available GPW symbols
            symbols_to_subscribe = []
            for gpw_symbol in self.gpw_symbols:
                for available in available_symbols:
                    if gpw_symbol in available:
                        symbols_to_subscribe.append(available)
                        break

            if symbols_to_subscribe:
                await self.client.subscribe_quotes(symbols_to_subscribe)
                self.subscribed_symbols = symbols_to_subscribe
                logger.info(f"Subscribed to {len(symbols_to_subscribe)} GPW symbols")
            else:
                logger.warning("No matching GPW symbols found")

        except Exception as e:
            logger.error(f"Error subscribing to GPW symbols: {e}")

    async def get_real_time_price(self, symbol: str) -> Optional[float]:
        """Get real-time price for a symbol"""
        quote = self.client.get_latest_quote(symbol)
        if quote:
            return (quote.ask + quote.bid) / 2
        return None

    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview with current prices and changes"""
        try:
            overview = {
                'timestamp': datetime.now(),
                'symbols': {},
                'market_stats': {
                    'total_symbols': len(self.subscribed_symbols),
                    'gainers': 0,
                    'losers': 0,
                    'unchanged': 0
                }
            }

            for symbol in self.subscribed_symbols:
                quote = self.client.get_latest_quote(symbol)
                if quote:
                    overview['symbols'][symbol] = {
                        'price': (quote.ask + quote.bid) / 2,
                        'change_pct': quote.change_pct,
                        'volume': quote.volume,
                        'high': quote.high,
                        'low': quote.low,
                        'time': quote.time
                    }

                    if quote.change_pct > 0:
                        overview['market_stats']['gainers'] += 1
                    elif quote.change_pct < 0:
                        overview['market_stats']['losers'] += 1
                    else:
                        overview['market_stats']['unchanged'] += 1

            return overview

        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}

    def add_data_callback(self, symbol: str, callback: Callable[[XTBQuote], None]):
        """Add callback for specific symbol data"""
        if symbol not in self.data_callbacks:
            self.data_callbacks[symbol] = []
        self.data_callbacks[symbol].append(callback)

    async def shutdown(self):
        """Shutdown xtB connection"""
        await self.client.disconnect()


# Global xtB data manager instance
xtb_manager = XTBDataManager()


async def initialize_xtb() -> bool:
    """Initialize xtB integration"""
    return await xtb_manager.initialize()


async def shutdown_xtb():
    """Shutdown xtB integration"""
    await xtb_manager.shutdown()


# Utility functions
async def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol"""
    return await xtb_manager.get_real_time_price(symbol)


async def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    return await xtb_manager.get_market_overview()


if __name__ == "__main__":
    async def main():
        """Example usage"""
        # Initialize xtB
        if await initialize_xtb():
            print("xtB initialized successfully")

            # Wait for some data
            await asyncio.sleep(10)

            # Get market overview
            overview = await get_market_status()
            print(f"Market overview: {overview}")

            # Get price for specific symbol
            price = await get_current_price('PLNKGHM')
            print(f"KGHM current price: {price}")

            # Shutdown
            await shutdown_xtb()
        else:
            print("Failed to initialize xtB")

    asyncio.run(main())