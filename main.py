#!/usr/bin/env python3
"""
Bybit VWAP Strategy Monitor with Trading
Monitors top 20 OI USDT perpetual futures, sends Telegram signals,
and supports market order execution with TP/SL
"""

import os
import time
import asyncio
import threading
import http.server
import socketserver
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path
import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import ccxt
import requests
import matplotlib

# Import backtest modules for SQLite data management
try:
    from backtest.database import init_db, get_ohlcv, upsert_ohlcv, get_latest_timestamp
    from backtest.collector import DataCollector, get_collector, start_collector
    USE_SQLITE = True
except ImportError:
    USE_SQLITE = False
    print("Warning: backtest module not found, using direct API calls")
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Load environment variables
load_dotenv()

# API Keys from .env (secrets stay in .env)
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Settings file path
SETTINGS_FILE = Path(__file__).parent / 'settings.json'

# Default settings
DEFAULT_SETTINGS = {
    "trading": {
        "leverage": 20,
        "order_size_usdt": 100,
        "position_check_interval": 10,
        "sl_buffer_atr_mult": 0.5
    },
    "strategy": {
        "band_entry_mult": 2.0,
        "exit_mode_long": "VWAP",
        "exit_mode_short": "VWAP",
        "target_long_deviation": 2.0,
        "target_short_deviation": 2.0,
        "min_strength": 0.7,
        "min_vol_ratio": 0.05
    },
    "safety_exit": {
        "enabled": True,
        "num_opposing_bars": 2
    },
    "trade_direction": {
        "allow_longs": True,
        "allow_shorts": True
    },
    "monitor": {
        "timeframes": ["3m", "5m", "15m"],
        "top_oi_count": 20,
        "check_interval": 60
    },
    "filters": {
        "no_trade_around_0900": True
    },
    "timezone": {
        "session_timezone": "UTC",
        "display_timezone": "Asia/Seoul"
    },
    "display": {
        "debug_mode": False,
        "send_chart": False,
        "chart_server_port": 8080
    }
}


class Settings:
    """Settings manager that loads from JSON and provides easy access"""

    def __init__(self):
        self._settings = {}
        self.load()

    def load(self):
        """Load settings from JSON file"""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f:
                    self._settings = json.load(f)
                print(f"✅ Settings loaded from {SETTINGS_FILE}")
            else:
                self._settings = DEFAULT_SETTINGS.copy()
                self.save()
                print(f"✅ Created default settings file")
        except Exception as e:
            print(f"⚠️ Error loading settings: {e}, using defaults")
            self._settings = DEFAULT_SETTINGS.copy()

    def save(self):
        """Save current settings to JSON file"""
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self._settings, f, indent=2)
            print(f"✅ Settings saved to {SETTINGS_FILE}")
            return True
        except Exception as e:
            print(f"❌ Error saving settings: {e}")
            return False

    def update(self, new_settings):
        """Update settings and save"""
        self._settings = new_settings
        return self.save()

    def to_dict(self):
        """Return settings as dictionary"""
        return self._settings.copy()

    # Trading settings
    @property
    def leverage(self):
        return self._settings.get('trading', {}).get('leverage', 20)

    @property
    def order_size_usdt(self):
        return self._settings.get('trading', {}).get('order_size_usdt', 100)

    @property
    def position_check_interval(self):
        return self._settings.get('trading', {}).get('position_check_interval', 10)

    @property
    def sl_buffer_atr_mult(self):
        return self._settings.get('trading', {}).get('sl_buffer_atr_mult', 0.5)

    @property
    def auto_trade(self):
        return self._settings.get('trading', {}).get('auto_trade', False)

    # Strategy settings
    @property
    def band_entry_mult(self):
        return self._settings.get('strategy', {}).get('band_entry_mult', 2.0)

    @property
    def exit_mode_long(self):
        return self._settings.get('strategy', {}).get('exit_mode_long', 'VWAP')

    @property
    def exit_mode_short(self):
        return self._settings.get('strategy', {}).get('exit_mode_short', 'VWAP')

    @property
    def target_long_deviation(self):
        return self._settings.get('strategy', {}).get('target_long_deviation', 2.0)

    @property
    def target_short_deviation(self):
        return self._settings.get('strategy', {}).get('target_short_deviation', 2.0)

    @property
    def min_strength(self):
        return self._settings.get('strategy', {}).get('min_strength', 0.7)

    @property
    def min_vol_ratio(self):
        return self._settings.get('strategy', {}).get('min_vol_ratio', 0.05)

    # Safety exit
    @property
    def enable_safety_exit(self):
        return self._settings.get('safety_exit', {}).get('enabled', True)

    @property
    def num_opposing_bars(self):
        return self._settings.get('safety_exit', {}).get('num_opposing_bars', 2)

    # Trade direction
    @property
    def allow_longs(self):
        return self._settings.get('trade_direction', {}).get('allow_longs', True)

    @property
    def allow_shorts(self):
        return self._settings.get('trade_direction', {}).get('allow_shorts', True)

    # Monitor settings
    @property
    def timeframes(self):
        return self._settings.get('monitor', {}).get('timeframes', ['3m', '5m', '15m'])

    @property
    def top_oi_count(self):
        return self._settings.get('monitor', {}).get('top_oi_count', 20)

    @property
    def check_interval(self):
        return self._settings.get('monitor', {}).get('check_interval', 60)

    # Filters
    @property
    def no_trade_around_0900(self):
        return self._settings.get('filters', {}).get('no_trade_around_0900', True)

    # Timezone
    @property
    def session_timezone(self):
        return self._settings.get('timezone', {}).get('session_timezone', 'UTC')

    @property
    def display_timezone(self):
        return self._settings.get('timezone', {}).get('display_timezone', 'Asia/Seoul')

    # Display
    @property
    def debug_mode(self):
        return self._settings.get('display', {}).get('debug_mode', False)

    @property
    def send_chart(self):
        return self._settings.get('display', {}).get('send_chart', False)

    @property
    def chart_server_port(self):
        return self._settings.get('display', {}).get('chart_server_port', 8080)


# Global settings instance
settings = Settings()

# Telegram notification
def send_telegram(message):
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200:
            print(f"❌ Telegram error: {response.text}")
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")


def send_telegram_photo(photo_path, caption=""):
    """Send photo to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        with open(photo_path, 'rb') as photo:
            response = requests.post(
                url,
                data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'},
                files={'photo': photo},
                timeout=30
            )
            if response.status_code != 200:
                print(f"❌ Telegram photo error: {response.text}")
    except Exception as e:
        print(f"❌ Telegram photo send failed: {e}")


def send_telegram_with_buttons(message, buttons):
    """Send message with inline keyboard buttons to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    keyboard = {"inline_keyboard": buttons}
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML',
        'reply_markup': json.dumps(keyboard)
    }

    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get('result', {}).get('message_id')
        else:
            print(f"❌ Telegram error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")
        return None


class BybitTrader:
    """Bybit trading client with TP/SL support"""

    def __init__(self):
        if not BYBIT_API_KEY or not BYBIT_API_SECRET:
            raise ValueError("API keys not configured!")

        self.exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,  # Auto-sync with server time
                'recvWindow': 60000,  # 60 second tolerance
            }
        })
        # Sync time with Bybit server
        self.exchange.load_time_difference()
        self.positions = {}  # Track open positions
        self.pending_signals = {}  # Signals waiting for user action
        self._last_order_error = None  # Store last order error for debugging

    def set_leverage(self, symbol, leverage=None):
        """Set leverage for a symbol"""
        if leverage is None:
            leverage = settings.leverage
        try:
            # Convert symbol format (BTC/USDT:USDT -> BTCUSDT)
            market_symbol = symbol.replace('/USDT:USDT', 'USDT').replace(':USDT', '')

            self.exchange.set_leverage(leverage, symbol)
            print(f"✅ Leverage set to {leverage}x for {market_symbol}")
            return True
        except Exception as e:
            # Leverage might already be set
            if 'leverage not modified' in str(e).lower():
                return True
            print(f"⚠️ Leverage setting: {e}")
            return True  # Continue anyway

    def get_ticker_price(self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None

    def calculate_quantity(self, symbol, usdt_amount):
        """Calculate order quantity based on USDT amount"""
        try:
            price = self.get_ticker_price(symbol)
            if not price:
                return None

            # Get market info for precision
            market = self.exchange.market(symbol)
            min_qty = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            precision = market.get('precision', {}).get('amount', 3)

            # Ensure precision is an integer
            if isinstance(precision, float):
                precision = int(precision) if precision > 0 else 3

            # Calculate quantity: margin * leverage = position size
            # e.g., $100 margin * 20x = $2000 position
            position_value = usdt_amount * settings.leverage
            quantity = position_value / price

            # Round to precision
            quantity = round(quantity, precision)

            # Ensure minimum quantity
            if min_qty and quantity < min_qty:
                quantity = min_qty

            print(f"📊 Quantity calc: ${usdt_amount} / ${price:.4f} = {quantity} (min: {min_qty})")
            return quantity
        except Exception as e:
            print(f"❌ Error calculating quantity: {e}")
            return None

    def place_market_order(self, symbol, side, quantity):
        """Place market order"""
        try:
            self.set_leverage(symbol)

            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,  # 'buy' or 'sell'
                amount=quantity
            )
            print(f"✅ Market {side.upper()} order placed: {symbol} qty={quantity}")
            return order
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Order failed: {error_msg}")
            # Store error for caller to access
            self._last_order_error = error_msg
            return None

    def place_tp_sl(self, symbol, side, entry_price, tp_price=None, sl_price=None):
        """
        Place Take Profit and Stop Loss using Bybit's position TP/SL
        - tp_price: Signal's VWAP target (uses fallback if None)
        - sl_price: Signal bar high/low ± buffer (required)
        """
        try:
            print(f"🔧 place_tp_sl called: side={side}, entry={entry_price}, tp={tp_price}, sl={sl_price}")

            # SL must be provided (signal bar based)
            if sl_price is None:
                print(f"⚠️ SL not provided, using entry ± 1% fallback")
                if side == 'buy':
                    sl_price = entry_price * 0.99
                else:
                    sl_price = entry_price * 1.01

            # Fallback: calculate TP from percentage if not provided
            if tp_price is None:
                print(f"🔧 TP is None, using 3% fallback...")
                if side == 'buy':
                    tp_price = entry_price * 1.03
                else:
                    tp_price = entry_price * 0.97
                print(f"🔧 Fallback TP: {tp_price}")

            print(f"📊 Setting TP=${tp_price:.4f} / SL=${sl_price:.4f} (signal bar)")

            # Round prices
            market = self.exchange.market(symbol)
            price_precision = market.get('precision', {}).get('price', 2)
            if isinstance(price_precision, float):
                price_precision = int(price_precision) if price_precision > 0 else 2
            tp_price = round(tp_price, price_precision)
            sl_price = round(sl_price, price_precision)

            # Get the Bybit symbol format
            bybit_symbol = symbol.replace('/USDT:USDT', 'USDT').replace(':USDT', '')

            print(f"📊 Setting TP: ${tp_price} (VWAP) / SL: ${sl_price} for {bybit_symbol}")

            # Use Bybit's set_trading_stop API
            response = self.exchange.private_post_v5_position_trading_stop({
                'category': 'linear',
                'symbol': bybit_symbol,
                'takeProfit': str(tp_price),
                'stopLoss': str(sl_price),
                'tpTriggerBy': 'LastPrice',
                'slTriggerBy': 'LastPrice',
                'tpslMode': 'Full',
                'positionIdx': 0,  # One-way mode
            })

            if response.get('retCode') == 0:
                print(f"✅ TP/SL set: TP=${tp_price} | SL=${sl_price}")
                return {'tp_price': tp_price, 'sl_price': sl_price}
            else:
                print(f"⚠️ TP/SL response: {response}")
                return {'tp_price': tp_price, 'sl_price': sl_price}

        except Exception as e:
            print(f"❌ TP/SL placement failed: {e}")
            return {'tp_price': tp_price if 'tp_price' in locals() else None,
                    'sl_price': sl_price if 'sl_price' in locals() else None}

    def execute_signal_trade(self, symbol, signal_type, tp_target=None, sl_initial=None, timeframe=None):
        """
        Execute a trade based on signal with TP/SL
        - tp_target: VWAP target from signal
        - sl_initial: Signal bar high/low ± buffer
        - timeframe: Timeframe for Safety Exit checking
        """
        try:
            side = 'buy' if signal_type == 'LONG' else 'sell'

            print(f"🔄 Executing {signal_type} for {symbol}")
            print(f"📊 Order size: ${settings.order_size_usdt} | Leverage: {settings.leverage}x")

            # Get current price first (for fallback and logging)
            current_price = self.get_ticker_price(symbol)
            if not current_price:
                return None, "Could not get current price"
            print(f"💰 Current price: ${current_price:.4f}")

            quantity = self.calculate_quantity(symbol, settings.order_size_usdt)
            if not quantity:
                return None, "Failed to calculate quantity"

            # Place market order
            order = self.place_market_order(symbol, side, quantity)
            if not order:
                error_detail = getattr(self, '_last_order_error', 'Unknown error')
                return None, f"Market order failed: {error_detail}"

            # Get fill price (try multiple sources)
            entry_price = current_price  # Default fallback
            try:
                if order.get('average') and order['average'] is not None:
                    entry_price = float(order['average'])
                elif order.get('price') and order['price'] is not None:
                    entry_price = float(order['price'])
            except (TypeError, ValueError) as e:
                print(f"⚠️ Price conversion issue: {e}, using current price")
                entry_price = current_price

            print(f"✅ Order filled at ${entry_price:.4f}")

            # Place TP/SL (uses fallback if signal values not available)
            if tp_target:
                print(f"🎯 TP Target (VWAP): ${tp_target:.4f}")
            if sl_initial:
                print(f"🛑 SL Initial: ${sl_initial:.4f}")

            tp_sl = self.place_tp_sl(symbol, side, entry_price, tp_target, sl_initial)

            # Track position
            self.positions[symbol] = {
                'side': signal_type,
                'entry_price': entry_price,
                'quantity': quantity,
                'entry_time': datetime.now(timezone.utc),
                'tp_price': tp_sl.get('tp_price') if tp_sl else tp_target,
                'sl_price': tp_sl.get('sl_price') if tp_sl else sl_initial,
                'timeframe': timeframe or settings.timeframes[0]  # For Safety Exit checking
            }

            return {
                'order': order,
                'entry_price': entry_price,
                'quantity': quantity,
                'tp_sl': tp_sl
            }, None

        except Exception as e:
            return None, str(e)

    def get_positions(self):
        """Get all open positions"""
        try:
            positions = self.exchange.fetch_positions()
            open_positions = []
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': pos['contracts'],
                        'entry_price': pos['entryPrice'],
                        'mark_price': pos['markPrice'],
                        'unrealized_pnl': pos['unrealizedPnl'],
                        'leverage': pos['leverage'],
                        'liquidation_price': pos.get('liquidationPrice')
                    })
            return open_positions
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return []

    def close_position(self, symbol, side=None):
        """Close a position"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    close_side = 'sell' if pos['side'] == 'long' else 'buy'
                    order = self.exchange.create_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=pos['contracts'],
                        params={'reduceOnly': True}
                    )
                    if symbol in self.positions:
                        del self.positions[symbol]
                    return order
            return None
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return None

    # ========== Safety Exit Methods ==========

    def fetch_ohlcv_for_safety(self, symbol, timeframe, limit=10):
        """Fetch OHLCV data to check for opposing bars"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            print(f"⚠️ Error fetching {timeframe} data for safety exit: {e}")
            return None

    def check_safety_exit(self, symbol, side, timeframe):
        """
        Check if we should exit due to consecutive opposing bars
        - Long: exit if N consecutive bearish bars (close < open)
        - Short: exit if N consecutive bullish bars (close > open)
        """
        df = self.fetch_ohlcv_for_safety(symbol, timeframe, limit=settings.num_opposing_bars + 2)
        if df is None or len(df) < settings.num_opposing_bars:
            return False

        # Get last N completed bars (exclude current incomplete bar)
        recent = df.tail(settings.num_opposing_bars + 1).head(settings.num_opposing_bars)

        if side == 'long':
            # Check for N consecutive bearish bars
            bearish_count = sum(1 for _, row in recent.iterrows() if row['close'] < row['open'])
            if bearish_count == settings.num_opposing_bars:
                print(f"⚠️ Safety Exit triggered: {settings.num_opposing_bars} consecutive bearish bars for LONG")
                return True
        else:  # short
            # Check for N consecutive bullish bars
            bullish_count = sum(1 for _, row in recent.iterrows() if row['close'] > row['open'])
            if bullish_count == settings.num_opposing_bars:
                print(f"⚠️ Safety Exit triggered: {settings.num_opposing_bars} consecutive bullish bars for SHORT")
                return True

        return False


# Global trader instance
trader = None

# Global live data for API access (shared with chart viewer)
api_live_data = {
    'symbols': [],
    'signals': {},
    'signal_history': [],
    'timeframe': '1h'
}

# Global exchange instance for API endpoints (public data only)
api_exchange = None


def init_api_exchange():
    """Initialize exchange for API endpoints (public data)"""
    global api_exchange
    api_exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })


def init_trader():
    """Initialize global trader instance"""
    global trader
    try:
        trader = BybitTrader()
        print(f"✅ Trader initialized with {settings.leverage}x leverage")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize trader: {e}")
        return False


def start_chart_server():
    """Start HTTP server for chart viewer and settings API"""

    class APIHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress request logs
            pass

        def do_GET(self):
            """Handle GET requests including API"""
            from urllib.parse import urlparse, parse_qs

            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == '/api/settings':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(settings.to_dict()).encode())

            elif path == '/api/symbols':
                # Return list of monitored symbols
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                symbols = api_live_data.get('symbols', [])
                symbol_list = [s['symbol'] if isinstance(s, dict) else s for s in symbols]

                # If no symbols yet, fetch top OI symbols directly
                if not symbol_list:
                    try:
                        if api_exchange is None:
                            init_api_exchange()
                        tickers = api_exchange.fetch_tickers()
                        usdt_symbols = []
                        for symbol, ticker in tickers.items():
                            if ':USDT' in symbol and ticker.get('info', {}).get('openInterest'):
                                oi = float(ticker['info']['openInterest'])
                                usdt_symbols.append({'symbol': symbol, 'oi_value': oi * (ticker['last'] or 0)})
                        usdt_symbols.sort(key=lambda x: x['oi_value'], reverse=True)
                        symbol_list = [s['symbol'] for s in usdt_symbols[:settings.top_oi_count]]
                    except Exception as e:
                        print(f"Error fetching symbols: {e}")

                self.wfile.write(json.dumps({'symbols': symbol_list}).encode())

            elif path == '/api/ohlcv':
                # Return OHLCV data with VWAP/bands for chart viewer
                symbol = query.get('symbol', [None])[0]
                timeframe = query.get('timeframe', [settings.timeframes[0]])[0]

                if not symbol:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'symbol required'}).encode())
                    return

                try:
                    if api_exchange is None:
                        init_api_exchange()

                    # Fetch OHLCV data
                    ohlcv = api_exchange.fetch_ohlcv(symbol, timeframe, limit=500)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                    # Calculate VWAP (for chart display only)
                    strategy = VWAPStrategy()
                    vwap_df = strategy.calculate_vwap(df.copy(), symbol, timeframe)
                    df_with_vwap = pd.concat([df.reset_index(drop=True), vwap_df], axis=1)

                    # Check for signals on each CLOSED bar (exclude last incomplete candle)
                    signals = []
                    for i in range(50, len(df) - 1):  # -1 to exclude incomplete current candle
                        bar_df = df.iloc[:i+1].copy()
                        result = strategy.check_signal(bar_df, symbol, timeframe)
                        if result and result.get('type'):
                            signals.append({
                                'time': int(df.iloc[i]['timestamp'] / 1000),
                                'type': result['type'],
                                'price': float(df.iloc[i]['close'])
                            })

                    # Format data for TradingView lightweight-charts
                    candles = []
                    vwap_data = []
                    upper_band_data = []
                    lower_band_data = []

                    for _, row in df_with_vwap.iterrows():
                        time_sec = int(row['timestamp'] / 1000)
                        candles.append({
                            'time': time_sec,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close'])
                        })
                        if pd.notna(row.get('vwap')):
                            vwap_data.append({'time': time_sec, 'value': float(row['vwap'])})
                        if pd.notna(row.get('upper_band')):
                            upper_band_data.append({'time': time_sec, 'value': float(row['upper_band'])})
                        if pd.notna(row.get('lower_band')):
                            lower_band_data.append({'time': time_sec, 'value': float(row['lower_band'])})

                    response = {
                        'candles': candles,
                        'vwap': vwap_data,
                        'upper_band': upper_band_data,
                        'lower_band': lower_band_data,
                        'signals': signals
                    }

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())

            elif path == '/api/signals':
                # Return signal history
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'signals': api_live_data.get('signal_history', []),
                    'current': api_live_data.get('signals', {})
                }).encode())

            else:
                # Serve static files
                super().do_GET()

        def do_POST(self):
            """Handle POST requests for settings API"""
            if self.path == '/api/settings':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)

                try:
                    new_settings = json.loads(post_data.decode('utf-8'))
                    success = settings.update(new_settings)

                    self.send_response(200 if success else 500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    response = {'success': success, 'message': 'Settings saved' if success else 'Failed to save'}
                    self.wfile.write(json.dumps(response).encode())
                except Exception as e:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_OPTIONS(self):
            """Handle CORS preflight"""
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

    try:
        port = settings.chart_server_port

        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        with ReusableTCPServer(("", port), APIHandler) as httpd:
            print(f"📊 Dashboard: http://localhost:{port}/chart_viewer.html")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"⚠️ Port {port} already in use, server not started")
        else:
            print(f"⚠️ Server error: {e}")


class VWAPStrategy:
    """VWAP Mean Reversion Strategy Implementation"""

    def __init__(self):
        pass

    def calculate_vwap(self, df, symbol, timeframe):
        """Calculate VWAP and standard deviation bands - matching Pine Script logic"""
        df = df.copy()

        # Convert timestamp to datetime (UTC)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Convert to session timezone (must match TradingView chart timezone)
        # VWAP resets at 00:00 in this timezone
        try:
            tz = pytz.timezone(settings.session_timezone)
            df['datetime_local'] = df['datetime'].dt.tz_convert(tz)
            df['date'] = df['datetime_local'].dt.date
        except:
            # Fallback to UTC if timezone invalid
            df['date'] = df['datetime'].dt.date

        # Calculate hlc3 (typical price)
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

        # Identify session starts (new day in local timezone)
        df['new_session'] = df['date'] != df['date'].shift(1)

        # Create session groups
        df['session_id'] = df['new_session'].cumsum()

        # Calculate cumulative sums within each session
        df['src_vol'] = df['hlc3'] * df['volume']
        df['src_sq_vol'] = (df['hlc3'] ** 2) * df['volume']

        df['cum_src_vol'] = df.groupby('session_id')['src_vol'].cumsum()
        df['cum_vol'] = df.groupby('session_id')['volume'].cumsum()
        df['cum_src_sq_vol'] = df.groupby('session_id')['src_sq_vol'].cumsum()

        # Calculate VWAP
        df['vwap'] = df['cum_src_vol'] / df['cum_vol']

        # Calculate standard deviation
        df['variance'] = (df['cum_src_sq_vol'] / df['cum_vol']) - (df['vwap'] ** 2)
        df['stdev'] = np.sqrt(df['variance'].clip(lower=0))

        # Calculate bands
        df['upper_band'] = df['vwap'] + df['stdev'] * settings.band_entry_mult
        df['lower_band'] = df['vwap'] - df['stdev'] * settings.band_entry_mult

        return df[['vwap', 'stdev', 'upper_band', 'lower_band']].reset_index(drop=True)
    
    def check_signal(self, df, symbol, timeframe):
        """Check for entry signals"""
        if len(df) < 15:  # Need enough data for ATR
            return None

        # Check for 09:00 KST no-trade window (±30 minutes)
        if settings.no_trade_around_0900:
            from datetime import timezone, timedelta
            kst = timezone(timedelta(hours=9))
            now_kst = datetime.now(kst)
            # Check if within ±30 min of 09:00
            minutes_from_0900 = (now_kst.hour * 60 + now_kst.minute) - (9 * 60)
            if abs(minutes_from_0900) <= 30:
                return None

        # Reset index to avoid concat issues
        df = df.reset_index(drop=True)

        # Calculate VWAP
        vwap_df = self.calculate_vwap(df.copy(), symbol, timeframe)
        df = pd.concat([df, vwap_df], axis=1)

        # Calculate ATR for volatility filter
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # Latest bar
        current = df.iloc[-1]

        # Volatility filter
        if current['stdev'] > 0 and current['atr'] > 0:
            vol_ratio = current['stdev'] / current['atr']
            if vol_ratio < settings.min_vol_ratio:
                return None
        else:
            return None

        # Signal strength
        bar_range = current['high'] - current['low']
        if bar_range <= 0:
            return None

        bull_strength = (current['close'] - current['low']) / bar_range
        bear_strength = (current['high'] - current['close']) / bar_range

        # Debug info for comparison with TradingView
        debug_info = {
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'vwap': current['vwap'],
            'stdev': current['stdev'],
            'upper_band': current['upper_band'],
            'lower_band': current['lower_band'],
            'atr': current['atr'],
            'vol_ratio': vol_ratio,
            'bull_strength': bull_strength,
            'bear_strength': bear_strength,
        }

        # Check H1/H2 pattern (Long signal)
        # Pine: open < entryLower and close > entryLower
        is_h1h2 = (
            settings.allow_longs and
            current['open'] < current['lower_band'] and
            current['close'] > current['lower_band'] and
            bull_strength >= settings.min_strength
        )

        # Check L1/L2 pattern (Short signal)
        # Pine: open > entryUpper and close < entryUpper
        is_l1l2 = (
            settings.allow_shorts and
            current['open'] > current['upper_band'] and
            current['close'] < current['upper_band'] and
            bear_strength >= settings.min_strength
        )

        # Debug: Show why signal conditions fail
        debug_info['long_conditions'] = {
            'open < lower_band': current['open'] < current['lower_band'],
            'close > lower_band': current['close'] > current['lower_band'],
            'bull_strength >= min': bull_strength >= settings.min_strength,
        }
        debug_info['short_conditions'] = {
            'open > upper_band': current['open'] > current['upper_band'],
            'close < upper_band': current['close'] < current['upper_band'],
            'bear_strength >= min': bear_strength >= settings.min_strength,
        }

        if is_h1h2:
            # Target based on exit mode
            if settings.exit_mode_long == 'VWAP':
                target = current['vwap']
            elif settings.exit_mode_long == 'Deviation Band':
                target = current['vwap'] + current['stdev'] * settings.target_long_deviation
            else:
                target = None
            # SL = signal bar low - ATR buffer (scales with volatility)
            atr_buffer = current['atr'] * settings.sl_buffer_atr_mult
            stop_loss = current['low'] - atr_buffer
            return {
                'type': 'LONG',
                'price': current['close'],
                'stop_loss': stop_loss,
                'signal_low': current['low'],
                'signal_high': current['high'],
                'target': target,
                'exit_mode': settings.exit_mode_long,
                'vwap': current['vwap'],
                'atr': current['atr'],
                'atr_buffer': atr_buffer,
                'strength': bull_strength,
                'vol_ratio': vol_ratio,
                'debug': debug_info
            }

        if is_l1l2:
            # Target based on exit mode
            if settings.exit_mode_short == 'VWAP':
                target = current['vwap']
            elif settings.exit_mode_short == 'Deviation Band':
                target = current['vwap'] - current['stdev'] * settings.target_short_deviation
            else:
                target = None
            # SL = signal bar high + ATR buffer (scales with volatility)
            atr_buffer = current['atr'] * settings.sl_buffer_atr_mult
            stop_loss = current['high'] + atr_buffer
            return {
                'type': 'SHORT',
                'price': current['close'],
                'stop_loss': stop_loss,
                'signal_low': current['low'],
                'signal_high': current['high'],
                'target': target,
                'exit_mode': settings.exit_mode_short,
                'vwap': current['vwap'],
                'atr': current['atr'],
                'atr_buffer': atr_buffer,
                'strength': bear_strength,
                'vol_ratio': vol_ratio,
                'debug': debug_info
            }

        # Return debug info even when no signal (for settings.debug_mode)
        return {'type': None, 'debug': debug_info}


class BybitMonitor:
    """Bybit market monitor for top OI symbols"""

    def __init__(self):
        # No API keys needed - only using public endpoints (tickers, OHLCV)
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}  # USDT perpetual
        })
        self.strategy = VWAPStrategy()
        self.last_signals = {}  # Track last signal time to avoid spam
        self.live_data = {'symbols': [], 'signals': {}, 'signal_history': [], 'timeframe': settings.timeframes[0]}

    def generate_live_chart(self, df, symbol, timeframe, signal=None):
        """Generate live chart for a symbol"""
        os.makedirs('charts', exist_ok=True)
        clean_symbol = symbol.replace(':USDT', '').replace('/USDT', '')
        save_path = f"charts/live_{clean_symbol}.png"

        try:
            # Calculate VWAP
            df_chart = df.reset_index(drop=True)
            vwap_df = self.strategy.calculate_vwap(df_chart.copy(), symbol, timeframe)
            df_chart = pd.concat([df_chart, vwap_df], axis=1)

            # Generate chart
            generate_chart(df_chart, symbol, timeframe, signal, save_path)
            return True
        except Exception as e:
            print(f"⚠️ Chart error for {symbol}: {e}")
            return False

    def update_live_data(self, symbols_data):
        """Update live_data.json for the HTML viewer and sync with API"""
        global api_live_data

        self.live_data['symbols'] = symbols_data
        self.live_data['updated'] = datetime.now().isoformat()

        # Sync with global api_live_data for API endpoints
        api_live_data['symbols'] = symbols_data
        api_live_data['signals'] = self.live_data.get('signals', {})
        api_live_data['signal_history'] = self.live_data.get('signal_history', [])
        api_live_data['timeframe'] = self.live_data.get('timeframe', settings.timeframes[0])

        os.makedirs('charts', exist_ok=True)
        with open('charts/live_data.json', 'w') as f:
            json.dump(self.live_data, f)

    def add_signal_to_history(self, symbol, timeframe, signal_type):
        """Add signal to history"""
        self.live_data['signal_history'].insert(0, {
            'symbol': symbol,
            'timeframe': timeframe,
            'type': signal_type,
            'time': datetime.now().isoformat()
        })
        # Keep only last 50 signals
        self.live_data['signal_history'] = self.live_data['signal_history'][:50]
    
    async def get_top_oi_symbols(self):
        """Get top 20 symbols by open interest"""
        try:
            tickers = self.exchange.fetch_tickers()
            
            # Symbols to exclude
            exclude_symbols = ['1000PEPE']

            # Filter USDT perpetual futures
            usdt_symbols = []
            for symbol, ticker in tickers.items():
                if ':USDT' in symbol and ticker.get('info', {}).get('openInterest'):
                    # Skip excluded symbols
                    if any(excl in symbol for excl in exclude_symbols):
                        continue
                    oi = float(ticker['info']['openInterest'])
                    usdt_symbols.append({
                        'symbol': symbol,
                        'oi': oi,
                        'oi_value': oi * ticker['last'] if ticker['last'] else 0
                    })
            
            # Sort by OI value and get top 20
            usdt_symbols.sort(key=lambda x: x['oi_value'], reverse=True)
            top_symbols = [s['symbol'] for s in usdt_symbols[:settings.top_oi_count]]
            
            print(f"📊 Top {settings.top_oi_count} OI symbols:")
            for i, s in enumerate(usdt_symbols[:settings.top_oi_count], 1):
                print(f"  {i}. {s['symbol']}: ${s['oi_value']:,.0f}")
            
            return top_symbols
            
        except Exception as e:
            print(f"❌ Error getting OI data: {e}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        """Fetch OHLCV data from SQLite (if available) or API"""
        try:
            # Try SQLite first if available
            if USE_SQLITE:
                df = get_ohlcv(symbol, timeframe, limit=limit)
                if df is not None and len(df) >= limit * 0.8:  # At least 80% of requested
                    return df
                # Fall through to API if not enough data

            # Fallback to API
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Store in SQLite for future use
            if USE_SQLITE and ohlcv:
                upsert_ohlcv(symbol, timeframe, ohlcv)

            return df
        except Exception as e:
            print(f"❌ Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def format_signal_message(self, symbol, timeframe, signal):
        """Format signal for Telegram"""
        signal_type = signal['type']
        emoji = "🟢" if signal_type == "LONG" else "🔴"

        # Format target line based on exit mode
        exit_mode = signal.get('exit_mode', 'VWAP')
        if signal['target'] is not None:
            target_line = f"<b>Target ({exit_mode}):</b> ${signal['target']:.4f}"
        else:
            target_line = f"<b>Target:</b> None (Exit Mode: {exit_mode})"

        # Format SL line with ATR info
        if signal.get('stop_loss') is not None:
            atr_buffer = signal.get('atr_buffer', 0)
            sl_line = f"<b>SL:</b> ${signal['stop_loss']:.4f} (ATR×{settings.sl_buffer_atr_mult}={atr_buffer:.2f})"
        else:
            sl_line = f"<b>SL:</b> Signal bar ± ATR×{settings.sl_buffer_atr_mult}"

        message = f"""
{emoji} <b>{signal_type} SIGNAL</b> {emoji}

<b>Symbol:</b> {symbol.replace(':USDT', '')}
<b>Timeframe:</b> {timeframe}
<b>Entry:</b> ${signal['price']:.4f}
{target_line}
{sl_line}

<b>Signal Strength:</b> {signal['strength']:.2%}
<b>Vol Ratio:</b> {signal['vol_ratio']:.2f}

<b>Leverage:</b> {settings.leverage}x | <b>Size:</b> ${settings.order_size_usdt}
<b>Safety Exit:</b> {settings.num_opposing_bars} opposing bars

<i>Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
        return message

    def send_signal_with_buttons(self, symbol, timeframe, signal):
        """Send signal with trade execution buttons"""
        message = self.format_signal_message(symbol, timeframe, signal)

        # Create inline keyboard buttons
        clean_symbol = symbol.replace(':USDT', '').replace('/USDT', '')
        signal_type = signal['type']

        buttons = [
            [
                {"text": f"🚀 {signal_type} NOW", "callback_data": f"trade_{clean_symbol}_{signal_type}"},
                {"text": "❌ Skip", "callback_data": f"skip_{clean_symbol}"}
            ]
        ]

        send_telegram_with_buttons(message, buttons)

        # Store pending signal for callback handler (including TP/SL)
        if trader:
            trader.pending_signals[clean_symbol] = {
                'symbol': symbol,
                'signal_type': signal_type,
                'price': signal['price'],
                'target': signal.get('target'),  # VWAP target for TP
                'stop_loss': signal.get('stop_loss'),  # Signal bar SL
                'signal_low': signal.get('signal_low'),
                'signal_high': signal.get('signal_high'),
                'timeframe': timeframe,
                'timestamp': time.time()
            }

    def auto_execute_trade(self, symbol, timeframe, signal):
        """Auto-execute trade immediately when signal detected"""
        global trader

        signal_type = signal['type']
        clean_symbol = symbol.replace(':USDT', '').replace('/USDT', '')

        print(f"\n🤖 AUTO TRADE: Executing {signal_type} for {clean_symbol}")

        # Execute trade
        result, error = trader.execute_signal_trade(
            symbol,
            signal_type,
            tp_target=signal.get('target'),
            sl_initial=signal.get('stop_loss'),
            timeframe=timeframe
        )

        if result:
            entry = result['entry_price']
            qty = result['quantity']
            tp_price = result['tp_sl'].get('tp_price') if result['tp_sl'] else signal.get('target')
            sl_actual = result['tp_sl'].get('sl_price') if result['tp_sl'] else signal.get('stop_loss')

            tp_str = f"${tp_price:.4f}" if tp_price else "Not set"
            sl_str = f"${sl_actual:.4f}" if sl_actual else "Not set"

            msg = f"""🤖 <b>AUTO TRADE EXECUTED</b>

<b>Symbol:</b> {clean_symbol}
<b>Side:</b> {signal_type}
<b>Entry:</b> ${entry:.4f}
<b>Quantity:</b> {qty}
<b>Leverage:</b> {settings.leverage}x

<b>TP:</b> {tp_str} (VWAP)
<b>SL:</b> {sl_str} (ATR-based)

<b>Signal Strength:</b> {signal.get('strength', 0):.2%}
<b>Safety Exit:</b> {settings.num_opposing_bars} opposing bars

<i>Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"""

            send_telegram(msg)
            print(f"✅ AUTO TRADE: {signal_type} executed at ${entry:.4f}")
        else:
            error_msg = f"""❌ <b>AUTO TRADE FAILED</b>

<b>Symbol:</b> {clean_symbol}
<b>Timeframe:</b> {timeframe}
<b>Side:</b> {signal_type}
<b>Error:</b> {error}"""
            send_telegram(error_msg)
            print(f"❌ AUTO TRADE FAILED [{timeframe}]: {error}")

    async def check_symbol(self, symbol):
        """Check a symbol across all timeframes"""
        for timeframe in settings.timeframes:
            try:
                # Check if we recently sent signal for this symbol/timeframe
                key = f"{symbol}_{timeframe}"
                last_signal_time = self.last_signals.get(key, 0)

                # Don't spam - wait at least 5 minutes between signals
                if time.time() - last_signal_time < 300:
                    continue

                # Fetch data
                df = self.fetch_ohlcv(symbol, timeframe)
                if df is None or len(df) < 15:
                    continue

                # Exclude last candle (incomplete/current candle) - only check closed candles
                df_closed = df.iloc[:-1].copy()
                if len(df_closed) < 15:
                    continue

                # Check for signal on closed candles only
                result = self.strategy.check_signal(df_closed, symbol, timeframe)

                if result is None:
                    continue

                # Debug output
                if settings.debug_mode and 'debug' in result:
                    d = result['debug']
                    print(f"\n📊 {symbol} ({timeframe}):")
                    print(f"   O: {d['open']:.4f} | H: {d['high']:.4f} | L: {d['low']:.4f} | C: {d['close']:.4f}")
                    print(f"   VWAP: {d['vwap']:.4f} | StDev: {d['stdev']:.4f}")
                    print(f"   Upper Band: {d['upper_band']:.4f} | Lower Band: {d['lower_band']:.4f}")
                    print(f"   Vol Ratio: {d['vol_ratio']:.2f} | ATR: {d['atr']:.4f}")
                    print(f"   Bull Str: {d['bull_strength']:.2f} | Bear Str: {d['bear_strength']:.2f}")
                    print(f"   Long Cond: {d['long_conditions']}")
                    print(f"   Short Cond: {d['short_conditions']}")

                # Check if actual signal (not just debug info)
                if result.get('type') is not None:
                    signal = result
                    print(f"\n{'='*60}")
                    print(f"🎯 SIGNAL DETECTED: {symbol} ({timeframe})")
                    print(f"   Type: {signal['type']}")
                    print(f"   Entry: ${signal['price']:.4f}")
                    atr_buf = signal.get('atr_buffer', 0)
                    print(f"   SL: ${signal['stop_loss']:.4f} (ATR×{settings.sl_buffer_atr_mult}={atr_buf:.2f})")
                    if signal['target'] is not None:
                        print(f"   TP ({signal['exit_mode']}): ${signal['target']:.4f}")
                    else:
                        print(f"   TP: None (Exit Mode: {signal['exit_mode']})")
                    print(f"{'='*60}\n")

                    # Auto trade or send signal with buttons
                    if settings.auto_trade and trader:
                        # AUTO TRADE: Execute immediately
                        self.auto_execute_trade(symbol, timeframe, signal)
                    else:
                        # Manual: Send signal with buttons
                        self.send_signal_with_buttons(symbol, timeframe, signal)

                    # Add to signal history for HTML viewer
                    self.add_signal_to_history(symbol, timeframe, signal['type'])

                    # Update last signal time
                    self.last_signals[key] = time.time()

            except Exception as e:
                print(f"❌ Error checking {symbol} {timeframe}: {e}")
                continue
    
    async def monitor(self):
        """Main monitoring loop"""
        print(f"\n🚀 Starting Bybit VWAP Monitor")
        print(f"📊 Monitoring top {settings.top_oi_count} OI symbols")
        print(f"⏱️  Timeframes: {', '.join(settings.timeframes)}")
        print(f"🔄 Check interval: {settings.check_interval}s")
        print(f"\n📋 Strategy Parameters:")
        print(f"   Session Timezone: {settings.session_timezone} (VWAP resets at 00:00 {settings.session_timezone})")
        print(f"   Display Timezone: {settings.display_timezone} (chart x-axis)")
        print(f"   SL: Signal bar ± ATR×{settings.sl_buffer_atr_mult}")
        print(f"   Safety Exit: {settings.num_opposing_bars} consecutive opposing bars")
        print(f"   Band Entry Mult: {settings.band_entry_mult}")
        print(f"   Exit Mode Long: {settings.exit_mode_long}")
        print(f"   Exit Mode Short: {settings.exit_mode_short}")
        print(f"   Min Strength: {settings.min_strength}")
        print(f"   Min Vol Ratio: {settings.min_vol_ratio}")
        print(f"   Allow Longs: {settings.allow_longs}")
        print(f"   Allow Shorts: {settings.allow_shorts}")
        print(f"   No Trade 09:00 KST: {settings.no_trade_around_0900}")
        print(f"   Debug Mode: {settings.debug_mode}")
        print(f"   Send Chart: {settings.send_chart}\n")

        send_telegram(f"""🚀 <b>Bybit VWAP Monitor Started</b>

Monitoring top {settings.top_oi_count} OI symbols
Timeframes: {', '.join(settings.timeframes)}
Exit Mode Long: {settings.exit_mode_long}
Exit Mode Short: {settings.exit_mode_short}""")
        
        while True:
            try:
                # Get top OI symbols
                symbols = await self.get_top_oi_symbols()

                if not symbols:
                    print("⚠️ No symbols found, retrying...")
                    await asyncio.sleep(settings.check_interval)
                    continue

                # Process each symbol - check signals and generate live charts
                symbols_data = []
                timeframe = settings.timeframes[0]  # Use first timeframe for live charts

                for symbol in symbols:
                    # Check for signals (all timeframes)
                    await self.check_symbol(symbol)

                    # Generate live chart (first timeframe only)
                    try:
                        df = self.fetch_ohlcv(symbol, timeframe)
                        if df is not None and len(df) >= 15:
                            # Calculate VWAP for data
                            df_calc = df.reset_index(drop=True)
                            vwap_df = self.strategy.calculate_vwap(df_calc.copy(), symbol, timeframe)
                            df_calc = pd.concat([df_calc, vwap_df], axis=1)

                            # Get latest values
                            last = df_calc.iloc[-1]
                            symbol_info = {
                                'symbol': symbol,
                                'close': float(last['close']),
                                'vwap': float(last['vwap']),
                                'upper_band': float(last['upper_band']),
                                'lower_band': float(last['lower_band']),
                                'stdev': float(last['stdev'])
                            }
                            symbols_data.append(symbol_info)

                            # Check for signal on closed candles only
                            df_closed = df.iloc[:-1].copy() if len(df) > 15 else df
                            result = self.strategy.check_signal(df_closed, symbol, timeframe)
                            if result and result.get('type'):
                                self.live_data['signals'][symbol] = {
                                    'type': result['type'],
                                    'price': result['price']
                                }
                                self.generate_live_chart(df, symbol, timeframe, result)
                            else:
                                # Remove old signal if no longer valid
                                if symbol in self.live_data['signals']:
                                    del self.live_data['signals'][symbol]
                                self.generate_live_chart(df, symbol, timeframe, None)

                    except Exception as e:
                        print(f"⚠️ Error processing {symbol}: {e}")

                    await asyncio.sleep(0.5)  # Rate limiting

                # Update live data for HTML viewer
                self.update_live_data(symbols_data)

                print(f"\n✅ Scan complete. Charts updated. Next scan in {settings.check_interval}s...")
                await asyncio.sleep(settings.check_interval)

            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                send_telegram("⏹️ <b>Bybit VWAP Monitor Stopped</b>")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(settings.check_interval)


class TelegramBotHandler:
    """Handle Telegram bot callbacks and commands"""

    def __init__(self):
        self.last_update_id = 0
        self.running = True

    def get_updates(self):
        """Get updates from Telegram"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {
            'offset': self.last_update_id + 1,
            'timeout': 5,
            'allowed_updates': ['callback_query', 'message']
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('result', [])
        except Exception as e:
            print(f"⚠️ Error getting Telegram updates: {e}")
        return []

    def answer_callback(self, callback_id, text):
        """Answer callback query"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
        try:
            requests.post(url, data={'callback_query_id': callback_id, 'text': text}, timeout=5)
        except Exception as e:
            print(f"⚠️ Error answering callback: {e}")

    def handle_callback(self, callback_data, callback_id):
        """Handle button callback"""
        global trader

        if callback_data.startswith('trade_'):
            # Parse: trade_BTCUSDT_LONG
            parts = callback_data.split('_')
            if len(parts) >= 3:
                symbol_short = parts[1]
                signal_type = parts[2]

                # Find full symbol
                symbol = f"{symbol_short}/USDT:USDT"

                self.answer_callback(callback_id, f"Executing {signal_type}...")

                if trader:
                    # Get pending signal with TP target and SL
                    pending = trader.pending_signals.get(symbol_short, {})
                    tp_target = pending.get('target')  # VWAP target
                    sl_price = pending.get('stop_loss')  # Signal bar SL
                    timeframe = pending.get('timeframe', settings.timeframes[0])
                    print(f"🔍 DEBUG tp_target={tp_target}, sl_price={sl_price}")

                    # Execute trade with signal bar SL
                    result, error = trader.execute_signal_trade(
                        symbol, signal_type,
                        tp_target=tp_target,
                        sl_initial=sl_price,
                        timeframe=timeframe
                    )

                    if result:
                        entry = result['entry_price']
                        qty = result['quantity']
                        tp_price = result['tp_sl'].get('tp_price') if result['tp_sl'] else tp_target
                        sl_actual = result['tp_sl'].get('sl_price') if result['tp_sl'] else sl_price

                        tp_str = f"${tp_price:.4f} (VWAP)" if tp_price else "Not set"
                        sl_str = f"${sl_actual:.4f} (ATR-based)" if sl_actual else "Not set"

                        msg = f"""✅ <b>Order Executed!</b>

<b>Symbol:</b> {symbol_short}
<b>Side:</b> {signal_type}
<b>Entry:</b> ${entry:.4f}
<b>Quantity:</b> {qty}
<b>Leverage:</b> {settings.leverage}x

<b>TP:</b> {tp_str}
<b>SL:</b> {sl_str}

<i>Safety Exit: {settings.num_opposing_bars} consecutive opposing bars</i>"""
                        send_telegram(msg)

                        # Clear pending signal
                        if symbol_short in trader.pending_signals:
                            del trader.pending_signals[symbol_short]
                    else:
                        send_telegram(f"❌ <b>Order Failed</b>\n\n{error}")
                else:
                    send_telegram("❌ Trader not initialized!")

        elif callback_data.startswith('skip_'):
            symbol_short = callback_data.split('_')[1]
            self.answer_callback(callback_id, f"Skipped {symbol_short}")
            send_telegram(f"⏭️ Skipped signal for {symbol_short}")

        elif callback_data.startswith('close_'):
            # Close position: close_BTCUSDT
            symbol_short = callback_data.split('_')[1]
            symbol = f"{symbol_short}/USDT:USDT"

            self.answer_callback(callback_id, "Closing position...")

            if trader:
                order = trader.close_position(symbol)
                if order:
                    send_telegram(f"✅ Position closed for {symbol_short}")
                else:
                    send_telegram(f"❌ Failed to close {symbol_short}")

    def handle_message(self, text):
        """Handle text commands"""
        global trader

        text = text.strip().lower()

        if text == '/positions' or text == '/pos':
            if trader:
                positions = trader.get_positions()
                if positions:
                    msg = "📊 <b>Open Positions</b>\n\n"
                    for pos in positions:
                        pnl = float(pos['unrealized_pnl']) if pos['unrealized_pnl'] else 0
                        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                        msg += f"<b>{pos['symbol'].replace('/USDT:USDT', '')}</b>\n"
                        msg += f"  Side: {pos['side'].upper()}\n"
                        msg += f"  Size: {pos['size']}\n"
                        msg += f"  Entry: ${float(pos['entry_price']):.4f}\n"
                        msg += f"  Mark: ${float(pos['mark_price']):.4f}\n"
                        msg += f"  {pnl_emoji} PnL: ${pnl:.2f}\n\n"
                    send_telegram(msg)
                else:
                    send_telegram("📊 No open positions")
            else:
                send_telegram("❌ Trader not initialized")

        elif text == '/help':
            help_msg = """📖 <b>Commands</b>

/positions - Show open positions
/close [SYMBOL] - Close position (e.g., /close BTCUSDT)
/balance - Show account balance
/help - Show this help

<b>Button Actions:</b>
🚀 LONG/SHORT - Execute market order
❌ Skip - Ignore signal
🔴 Close - Close position"""
            send_telegram(help_msg)

        elif text.startswith('/close '):
            symbol_short = text.replace('/close ', '').upper()
            symbol = f"{symbol_short}/USDT:USDT"
            if trader:
                order = trader.close_position(symbol)
                if order:
                    send_telegram(f"✅ Closed position for {symbol_short}")
                else:
                    send_telegram(f"❌ No position found for {symbol_short}")

        elif text == '/balance':
            if trader:
                try:
                    balance = trader.exchange.fetch_balance()
                    usdt = balance.get('USDT', {})
                    total = usdt.get('total', 0)
                    free = usdt.get('free', 0)
                    used = usdt.get('used', 0)
                    send_telegram(f"""💰 <b>Account Balance</b>

<b>Total:</b> ${total:.2f}
<b>Available:</b> ${free:.2f}
<b>In Use:</b> ${used:.2f}""")
                except Exception as e:
                    send_telegram(f"❌ Error fetching balance: {e}")

    async def poll_updates(self):
        """Poll for Telegram updates"""
        print("📱 Telegram bot handler started")

        while self.running:
            try:
                updates = self.get_updates()

                for update in updates:
                    self.last_update_id = update.get('update_id', self.last_update_id)

                    # Handle callback query (button press)
                    if 'callback_query' in update:
                        callback = update['callback_query']
                        callback_data = callback.get('data', '')
                        callback_id = callback.get('id')
                        self.handle_callback(callback_data, callback_id)

                    # Handle text message
                    elif 'message' in update:
                        message = update['message']
                        text = message.get('text', '')
                        chat_id = str(message.get('chat', {}).get('id', ''))

                        # Only handle messages from configured chat
                        if chat_id == TELEGRAM_CHAT_ID and text:
                            self.handle_message(text)

                await asyncio.sleep(1)

            except Exception as e:
                print(f"⚠️ Telegram poll error: {e}")
                await asyncio.sleep(5)


async def position_monitor():
    """Monitor open positions, check Safety Exit, and send updates"""
    global trader

    print("📈 Position monitor started")
    if settings.enable_safety_exit:
        print(f"📈 Safety Exit enabled: {settings.num_opposing_bars} consecutive opposing bars")

    last_report_time = 0
    last_safety_check = {}  # {symbol: timestamp}
    safety_check_interval = 30  # Check every 30 seconds
    report_interval = 60  # Report positions every 60 seconds

    while True:
        try:
            if trader:
                positions = trader.get_positions()
                current_time = time.time()

                # ========== Safety Exit Logic ==========
                if settings.enable_safety_exit and positions:
                    for pos in positions:
                        symbol = pos['symbol']
                        side = pos['side']  # 'long' or 'short'

                        # Check if enough time passed since last safety check
                        last_check = last_safety_check.get(symbol, 0)
                        if current_time - last_check < safety_check_interval:
                            continue

                        # Get timeframe from tracked position
                        tracked = trader.positions.get(symbol, {})
                        timeframe = tracked.get('timeframe', settings.timeframes[0])

                        # Check for Safety Exit condition
                        should_exit = trader.check_safety_exit(symbol, side, timeframe)
                        last_safety_check[symbol] = current_time

                        if should_exit:
                            # Close position
                            symbol_short = symbol.replace('/USDT:USDT', '').replace(':USDT', '')
                            print(f"🚨 Safety Exit: Closing {side.upper()} position for {symbol_short}")

                            order = trader.close_position(symbol)
                            if order:
                                send_telegram(f"""🚨 <b>Safety Exit Triggered</b>

<b>Symbol:</b> {symbol_short}
<b>Side:</b> {side.upper()}
<b>Reason:</b> {settings.num_opposing_bars} consecutive opposing bars

<i>Position closed automatically</i>""")

                # ========== Position Report ==========
                if positions and (current_time - last_report_time) >= report_interval:
                    msg = "📊 <b>Position Update</b>\n\n"

                    for pos in positions:
                        pnl = float(pos['unrealized_pnl']) if pos['unrealized_pnl'] else 0
                        entry_price = float(pos['entry_price']) if pos['entry_price'] else 0
                        mark_price = float(pos['mark_price']) if pos['mark_price'] else 0

                        # Calculate PnL percentage
                        if entry_price > 0:
                            if pos['side'] == 'long':
                                pnl_pct = ((mark_price - entry_price) / entry_price) * 100 * settings.leverage
                            else:
                                pnl_pct = ((entry_price - mark_price) / entry_price) * 100 * settings.leverage
                        else:
                            pnl_pct = 0

                        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                        symbol_short = pos['symbol'].replace('/USDT:USDT', '').replace(':USDT', '')

                        msg += f"<b>{symbol_short}</b> ({pos['side'].upper()})\n"
                        msg += f"  Entry: ${entry_price:.4f}\n"
                        msg += f"  Current: ${mark_price:.4f}\n"
                        msg += f"  {pnl_emoji} PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"

                        if pos.get('liquidation_price'):
                            msg += f"  ⚠️ Liq: ${float(pos['liquidation_price']):.4f}\n"
                        msg += "\n"

                    # Add close buttons
                    buttons = []
                    for pos in positions:
                        symbol_short = pos['symbol'].replace('/USDT:USDT', '').replace(':USDT', '')
                        buttons.append([
                            {"text": f"🔴 Close {symbol_short}", "callback_data": f"close_{symbol_short}"}
                        ])

                    send_telegram_with_buttons(msg, buttons)
                    last_report_time = current_time

            await asyncio.sleep(settings.position_check_interval)

        except Exception as e:
            print(f"⚠️ Position monitor error: {e}")
            await asyncio.sleep(settings.position_check_interval)


async def main():
    """Entry point"""
    global trader

    # Check configuration
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ ERROR: Telegram configuration missing!")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        return

    # Initialize SQLite database
    if USE_SQLITE:
        init_db()
        print("📦 SQLite database initialized")

    # Initialize trader
    if BYBIT_API_KEY and BYBIT_API_SECRET:
        init_trader()
    else:
        print("⚠️ Trading disabled - no API keys configured")

    # Start chart viewer server in background
    if settings.send_chart:
        server_thread = threading.Thread(target=start_chart_server, daemon=True)
        server_thread.start()

    # Start background data collector
    if USE_SQLITE:
        collector = get_collector()
        for tf in settings.timeframes:
            collector.add_symbol('BTC/USDT:USDT', tf)  # Main symbol
        collector.start(interval=60)
        print("📡 Background data collector started")

    # Create tasks
    monitor = BybitMonitor()
    telegram_handler = TelegramBotHandler()

    # Send startup message
    safety_status = "ON" if settings.enable_safety_exit else "OFF"
    startup_msg = f"""🚀 <b>Bybit VWAP Monitor Started</b>

Monitoring top {settings.top_oi_count} OI symbols
Timeframes: {', '.join(settings.timeframes)}
Check Interval: {settings.check_interval}s

<b>Trading Settings:</b>
Leverage: {settings.leverage}x
Order Size: ${settings.order_size_usdt}
TP: VWAP target
SL: Signal bar ± ATR×{settings.sl_buffer_atr_mult}
Safety Exit: {settings.num_opposing_bars} opposing bars [{safety_status}]

<i>Send /help for commands</i>"""
    send_telegram(startup_msg)

    # Run all tasks concurrently
    await asyncio.gather(
        monitor.monitor(),
        telegram_handler.poll_updates(),
        position_monitor()
    )


if __name__ == "__main__":
    asyncio.run(main())