#!/usr/bin/env python3
"""
xtB Real-time Integration Module

Provides real-time market data integration with xtB (xStation 5) API
for Polish stock market trading and analysis.

Features:
- Real-time price streaming via WebSocket
- Account information and portfolio tracking
- Order management and execution
- Market depth and order book data
- Trade confirmation and position monitoring
- Real-time risk metrics calculation
- Advanced order types (stop loss, take profit, trailing stops)
- Portfolio heat monitoring and risk management
"""

import asyncio
import websockets
import json
import logging
import time
import ssl
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Callable, Any, Tuple, Literal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from config import (
    XTB_CLIENT_ID, XTB_CLIENT_SECRET, XTB_DEMO_MODE,
    XTB_WEBSOCKET_URL_DEMO, XTB_WEBSOCKET_URL_LIVE,
    ENABLE_XTB_INTEGRATION, XTB_RETRY_ATTEMPTS, XTB_TIMEOUT_SECONDS,
    POLISH_STOCK_TICKERS, RISK_PER_TRADE_PCT, MAX_POSITION_SIZE_PCT,
    BACKTEST_COMMISSION_RATE, BACKTEST_SLIPPAGE_PCT
)
from risk_management import RiskManager
from market_regime import MarketRegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class XTBAccountInfo:
    """xtB account information structure"""
    account_id: str
    account_type: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float
    currency: str
    leverage: float
    credit: float
    bonus: float
    account_locked: bool = False
    market_closed: bool = False


@dataclass
class XTBPriceInfo:
    """Real-time price information"""
    symbol: str
    bid: float
    ask: float
    high: float
    low: float
    volume: int
    timestamp: datetime
    change: float
    change_pct: float
    spread: float = 0.0

    def __post_init__(self):
        self.spread = self.ask - self.bid


@dataclass
class XTBOrderInfo:
    """Order information structure"""
    order_id: int
    symbol: str
    type: Literal['BUY', 'SELL']
    volume: float
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    status: str
    profit: float
    comment: str
    commission: float = 0.0
    swap: float = 0.0
    margin: float = 0.0


@dataclass
class XTBTradeTransaction:
    """Trade transaction for order execution"""
    symbol: str
    cmd: int  # 0=BUY, 1=SELL
    volume: float
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    order: Optional[int] = None
    comment: Optional[str] = None
    type: int = 0  # 0=OPEN, 1=CLOSE, 2=MODIFY
    expiry: Optional[int] = None
    custom_comment: Optional[str] = None
    stop_loss_limit: Optional[float] = None
    take_profit_limit: Optional[float] = None


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
    currency: str = 'PLN'
    market_closed: bool = False


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
    timestamp: int = 0


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

class XTBAPIClient:
    """
    xtB xStation 5 API Client for real-time trading and market data
    """

    # Command constants
    CMD_LOGIN = "login"
    CMD_LOGOUT = "logout"
    CMD_GET_SYMBOLS = "getSymbols"
    CMD_GET_TICK_PRICES = "getTickPrices"
    CMD_GET_TRADE_RECORDS = "getTradeRecords"
    CMD_OPEN_TRADE = "openTrade"
    CMD_CLOSE_TRADE = "closeTrade"
    CMD_MODIFY_TRADE = "modifyTrade"
    CMD_GET_MARGIN_LEVEL = "getMarginLevel"
    CMD_GET_TRADES = "getTrades"
    CMD_GET_CALENDAR = "getCalendar"

    def __init__(self):
        self.client_id = XTB_CLIENT_ID
        self.client_secret = XTB_CLIENT_SECRET
        self.demo_mode = XTB_DEMO_MODE
        self.websocket_url = XTB_WEBSOCKET_URL_DEMO if self.demo_mode else XTB_WEBSOCKET_URL_LIVE
        self.retry_attempts = XTB_RETRY_ATTEMPTS
        self.timeout_seconds = XTB_TIMEOUT_SECONDS

        self.websocket = None
        self.session_id = None
        self.logged_in = False

        # Callbacks for real-time data
        self.price_callbacks: List[Callable[[XTBPriceInfo], None]] = []
        self.trade_callbacks: List[Callable[[XTBOrderInfo], None]] = []
        self.account_callbacks: List[Callable[[XTBAccountInfo], None]] = []

        # Data storage
        self.current_prices: Dict[str, XTBPriceInfo] = {}
        self.open_positions: Dict[int, XTBOrderInfo] = {}
        self.account_info: Optional[XTBAccountInfo] = None

        # Event management
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.price_stream_task = None

        # Risk management integration
        self.risk_manager = RiskManager()
        self.market_regime_detector = MarketRegimeDetector()

        logger.info(f"XTB API Client initialized (demo_mode={self.demo_mode})")

    async def connect(self) -> bool:
        """Establish WebSocket connection and login"""
        try:
            # Create SSL context for secure connection
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.websocket_url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )

            logger.info("WebSocket connection established")

            # Login
            success = await self.login()
            if success:
                self.running = True
                # Start price streaming task
                self.price_stream_task = asyncio.create_task(self._price_stream_loop())
                logger.info("XTB API connection successful")
                return True
            else:
                await self.disconnect()
                return False

        except Exception as e:
            logger.error(f"Failed to connect to XTB API: {e}")
            return False

    async def disconnect(self):
        """Close WebSocket connection and cleanup"""
        try:
            self.running = False

            if self.price_stream_task:
                self.price_stream_task.cancel()

            if self.logged_in:
                await self.logout()

            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            logger.info("XTB API disconnected")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def login(self) -> bool:
        """Login to xStation 5 API"""
        try:
            login_command = {
                "command": self.CMD_LOGIN,
                "arguments": {
                    "userId": self.client_id,
                    "password": self.client_secret
                }
            }

            response = await self._send_command(login_command)

            if response and response.get('status'):
                self.session_id = response.get('streamSessionId')
                self.logged_in = True
                logger.info("Successfully logged in to xStation 5")
                return True
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                logger.error(f"Login failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    async def logout(self) -> bool:
        """Logout from xStation 5 API"""
        try:
            if self.logged_in:
                logout_command = {"command": self.CMD_LOGOUT}
                response = await self._send_command(logout_command)
                self.logged_in = False
                self.session_id = None
                logger.info("Successfully logged out")
                return True
            return False

        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

    async def _send_command(self, command: Dict, command_id: int = None) -> Optional[Dict]:
        """Send command to xStation 5 API"""
        if not self.websocket:
            logger.error("WebSocket connection not established")
            return None

        try:
            # Add timestamp and command ID if not provided
            if command_id is None:
                command_id = int(time.time() * 1000)

            command['customTag'] = command_id

            # Send command
            await self.websocket.send(json.dumps(command))

            # Wait for response with timeout
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.timeout_seconds
            )

            response_data = json.loads(response)

            # Check for error response
            if response_data.get('status') is False:
                error_code = response_data.get('errorCode', 'UNKNOWN')
                error_desc = response_data.get('errorDesc', 'Unknown error')
                logger.error(f"API Error {error_code}: {error_desc}")
                return None

            return response_data

        except asyncio.TimeoutError:
            logger.error(f"Command timeout: {command.get('command')}")
            return None
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return None

  async def get_account_info(self) -> Optional[XTBAccountInfo]:
        """Get current account information"""
        try:
            command = {"command": self.CMD_GET_MARGIN_LEVEL}
            response = await self._send_command(command)

            if response and response.get('status'):
                data = response.get('return_data', {})

                account_info = XTBAccountInfo(
                    account_id=self.client_id,
                    account_type="DEMO" if self.demo_mode else "LIVE",
                    balance=data.get('balance', 0.0),
                    equity=data.get('equity', 0.0),
                    margin=data.get('margin', 0.0),
                    free_margin=data.get('margin_free', 0.0),
                    margin_level=data.get('margin_level', 0.0),
                    profit=data.get('profit', 0.0),
                    currency=data.get('currency', 'PLN'),
                    leverage=data.get('leverage', 1.0),
                    credit=data.get('credit', 0.0),
                    bonus=data.get('bonus', 0.0)
                )

                self.account_info = account_info

                # Notify callbacks
                for callback in self.account_callbacks:
                    try:
                        self.executor.submit(callback, account_info)
                    except Exception as e:
                        logger.error(f"Account callback error: {e}")

                return account_info

            return None

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, XTBPriceInfo]:
        """Get current prices for specified symbols"""
        try:
            command = {
                "command": self.CMD_GET_TICK_PRICES,
                "arguments": {
                    "symbols": symbols
                }
            }

            response = await self._send_command(command)

            if response and response.get('status'):
                prices = {}
                for price_data in response.get('return_data', []):
                    price_info = XTBPriceInfo(
                        symbol=price_data['symbol'],
                        bid=float(price_data['bid']),
                        ask=float(price_data['ask']),
                        high=float(price_data['high']),
                        low=float(price_data['low']),
                        volume=int(price_data['volume']),
                        timestamp=datetime.fromtimestamp(price_data['timestamp'], tz=timezone.utc),
                        change=float(price_data['change']),
                        change_pct=float(price_data['changePercentage'])
                    )
                    prices[price_info.symbol] = price_info
                    self.current_prices[price_info.symbol] = price_info

                return prices

            return {}

        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}

    async def get_open_positions(self) -> Dict[int, XTBOrderInfo]:
        """Get all open positions"""
        try:
            command = {"command": self.CMD_GET_TRADES}
            response = await self._send_command(command)

            if response and response.get('status'):
                positions = {}
                for trade_data in response.get('return_data', []):
                    if trade_data.get('cmd') in [0, 1]:  # BUY or SELL positions
                        order_info = XTBOrderInfo(
                            order_id=trade_data['order'],
                            symbol=trade_data['symbol'],
                            type='BUY' if trade_data['cmd'] == 0 else 'SELL',
                            volume=float(trade_data['volume']),
                            price=float(trade_data['open_price']),
                            stop_loss=float(trade_data['sl']) if trade_data['sl'] > 0 else None,
                            take_profit=float(trade_data['tp']) if trade_data['tp'] > 0 else None,
                            timestamp=datetime.fromtimestamp(trade_data['open_time'], tz=timezone.utc),
                            status=trade_data['state'],
                            profit=float(trade_data['profit']),
                            comment=trade_data.get('comment', '')
                        )
                        positions[order_info.order_id] = order_info
                        self.open_positions[order_info.order_id] = order_info

                return positions

            return {}

        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return {}

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
        self.client = XTBAPIClient()
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