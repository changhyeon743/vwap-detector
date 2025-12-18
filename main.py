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
import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import ccxt
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Load environment variables
load_dotenv()

# Configuration
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading Settings
LEVERAGE = int(os.getenv('LEVERAGE', '20'))
ORDER_SIZE_USDT = float(os.getenv('ORDER_SIZE_USDT', '100'))  # Margin size in USDT
POSITION_CHECK_INTERVAL = int(os.getenv('POSITION_CHECK_INTERVAL', '10'))  # seconds

# Trailing Stop Settings (1m Swing Strategy)
# TP = Signal's VWAP target (no env needed)
# SL = Trailing based on 1m swing + buffer
TRAILING_STOP_ENABLED = os.getenv('TRAILING_STOP_ENABLED', 'true').lower() == 'true'
SWING_N = int(os.getenv('SWING_N', '20'))  # Number of 1m candles for swing high/low
TRAIL_UPDATE_INTERVAL = int(os.getenv('TRAIL_UPDATE_INTERVAL', '60'))  # seconds between SL updates
SL_BUFFER_PERCENT = float(os.getenv('SL_BUFFER_PERCENT', '0.3'))  # Buffer as % of price (0.3% default)

# Monitor Settings
TIMEFRAMES = os.getenv('TIMEFRAMES', '3m,5m,15m').split(',')
TOP_OI_COUNT = int(os.getenv('TOP_OI_COUNT', '20'))
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))

# VWAP Strategy Parameters
BAND_ENTRY_MULT = float(os.getenv('BAND_ENTRY_MULT', '2.0'))

# Exit Mode: "VWAP", "Deviation Band", or "None"
EXIT_MODE_LONG = os.getenv('EXIT_MODE_LONG', 'VWAP')
EXIT_MODE_SHORT = os.getenv('EXIT_MODE_SHORT', 'VWAP')

# Target Deviation (used when Exit Mode is "Deviation Band")
TARGET_LONG_DEVIATION = float(os.getenv('TARGET_LONG_DEVIATION', '2.0'))
TARGET_SHORT_DEVIATION = float(os.getenv('TARGET_SHORT_DEVIATION', '2.0'))

# Safety Exit
ENABLE_SAFETY_EXIT = os.getenv('ENABLE_SAFETY_EXIT', 'true').lower() == 'true'
NUM_OPPOSING_BARS = int(os.getenv('NUM_OPPOSING_BARS', '3'))

# Trade Direction
ALLOW_LONGS = os.getenv('ALLOW_LONGS', 'true').lower() == 'true'
ALLOW_SHORTS = os.getenv('ALLOW_SHORTS', 'true').lower() == 'true'

# Signal Strength & Volatility Filter
MIN_STRENGTH = float(os.getenv('MIN_STRENGTH', '0.7'))
MIN_VOL_RATIO = float(os.getenv('MIN_VOL_RATIO', '0.25'))

# No Trade Window Around 09:00 KST
NO_TRADE_AROUND_0900 = os.getenv('NO_TRADE_AROUND_0900', 'true').lower() == 'true'

# VWAP Session Reset Timezone (must match TradingView)
SESSION_TIMEZONE = os.getenv('SESSION_TIMEZONE', 'UTC')

# Chart Display Timezone (for x-axis labels)
DISPLAY_TIMEZONE = os.getenv('DISPLAY_TIMEZONE', 'Asia/Seoul')

# Debug Mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Chart Generation
SEND_CHART = os.getenv('SEND_CHART', 'false').lower() == 'true'

# Chart Viewer Server
CHART_SERVER_PORT = int(os.getenv('CHART_SERVER_PORT', '8080'))

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

    def set_leverage(self, symbol, leverage=LEVERAGE):
        """Set leverage for a symbol"""
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
            position_value = usdt_amount * LEVERAGE
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
            print(f"❌ Order failed: {e}")
            return None

    def place_tp_sl(self, symbol, side, entry_price, tp_price=None, sl_price=None):
        """
        Place Take Profit and Stop Loss using Bybit's position TP/SL
        - tp_price: Signal's VWAP target (uses fallback if None)
        - sl_price: Signal's initial stop loss (uses swing-based if None)
        """
        try:
            print(f"🔧 place_tp_sl called: side={side}, entry={entry_price}, tp={tp_price}, sl={sl_price}")

            # Fallback: calculate SL from 1m swing if not provided
            if sl_price is None:
                print(f"🔧 SL is None, calculating from swing...")
                sl_price = self.calc_trailing_sl(symbol, 'long' if side == 'buy' else 'short')
                print(f"🔧 Swing SL result: {sl_price}")

            # Fallback: calculate TP from percentage if not provided
            if tp_price is None:
                print(f"🔧 TP is None, using 3% fallback...")
                if side == 'buy':
                    tp_price = entry_price * 1.03
                else:
                    tp_price = entry_price * 0.97
                print(f"🔧 Fallback TP: {tp_price}")

            if tp_price is None or sl_price is None:
                print(f"❌ Could not calculate TP/SL: tp={tp_price}, sl={sl_price}")
                return None

            print(f"📊 Setting TP=${tp_price:.4f} / SL=${sl_price:.4f}")

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

    def execute_signal_trade(self, symbol, signal_type, tp_target=None, sl_initial=None):
        """
        Execute a trade based on signal with TP/SL
        - tp_target: VWAP target from signal
        - sl_initial: Initial stop loss from signal (will be trailed)
        """
        try:
            side = 'buy' if signal_type == 'LONG' else 'sell'

            print(f"🔄 Executing {signal_type} for {symbol}")
            print(f"📊 Order size: ${ORDER_SIZE_USDT} | Leverage: {LEVERAGE}x")

            # Get current price first (for fallback and logging)
            current_price = self.get_ticker_price(symbol)
            if not current_price:
                return None, "Could not get current price"
            print(f"💰 Current price: ${current_price:.4f}")

            quantity = self.calculate_quantity(symbol, ORDER_SIZE_USDT)
            if not quantity:
                return None, "Failed to calculate quantity"

            # Place market order
            order = self.place_market_order(symbol, side, quantity)
            if not order:
                return None, "Market order failed"

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
                'sl_price': tp_sl.get('sl_price') if tp_sl else sl_initial
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

    # ========== Trailing Stop Methods ==========

    def fetch_1m_ohlcv(self, symbol, limit=50):
        """Fetch 1-minute OHLCV data for trailing stop"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            print(f"⚠️ Error fetching 1m data: {e}")
            return None

    def calc_trailing_sl(self, symbol, side, entry_price=None):
        """
        Calculate new SL based on 1m swing high/low + buffer
        - Long: SL = swing_low - buffer (only moves UP)
        - Short: SL = swing_high + buffer (only moves DOWN)

        Buffer is percentage-based (works for any coin price)
        """
        df = self.fetch_1m_ohlcv(symbol, limit=SWING_N + 10)
        if df is None or len(df) < SWING_N:
            return None

        # Get last N candles (exclude current incomplete candle)
        recent = df.tail(SWING_N + 1).head(SWING_N)
        current_price = df.iloc[-1]['close']

        # Buffer as percentage of price (e.g., 0.3% of $100,000 = $300)
        buffer = current_price * (SL_BUFFER_PERCENT / 100)

        if side == 'long':
            swing_low = recent['low'].min()
            new_sl = swing_low - buffer

            # Safety: SL must be below current price by at least buffer
            max_sl = current_price - buffer
            if new_sl > max_sl:
                new_sl = max_sl

        else:  # short
            swing_high = recent['high'].max()
            new_sl = swing_high + buffer

            # Safety: SL must be above current price by at least buffer
            min_sl = current_price + buffer
            if new_sl < min_sl:
                new_sl = min_sl

        return new_sl

    def update_position_sl(self, symbol, new_sl, current_tp=None):
        """Update stop-loss on Bybit position"""
        try:
            bybit_symbol = symbol.replace('/USDT:USDT', 'USDT').replace(':USDT', '')

            # If no TP provided, fetch from Bybit to preserve it
            if current_tp is None:
                try:
                    response = self.exchange.private_get_v5_position_list({
                        'category': 'linear',
                        'symbol': bybit_symbol
                    })
                    if response.get('retCode') == 0:
                        pos_list = response.get('result', {}).get('list', [])
                        if pos_list:
                            current_tp = pos_list[0].get('takeProfit')
                            if current_tp:
                                current_tp = float(current_tp)
                except Exception as e:
                    print(f"⚠️ Could not fetch current TP: {e}")

            params = {
                'category': 'linear',
                'symbol': bybit_symbol,
                'stopLoss': str(new_sl),
                'slTriggerBy': 'LastPrice',
                'tpslMode': 'Full',
                'positionIdx': 0,
            }

            if current_tp is not None and float(current_tp) > 0:
                params['takeProfit'] = str(current_tp)
                params['tpTriggerBy'] = 'LastPrice'

            response = self.exchange.private_post_v5_position_trading_stop(params)
            ret_code = response.get('retCode')

            if ret_code == 0:
                print(f"✅ SL updated to ${new_sl:.4f} for {bybit_symbol}")
                return True
            elif ret_code == 34040:  # "not modified" - SL unchanged
                return True
            else:
                print(f"⚠️ SL update: {response.get('retMsg', response)}")
                return False

        except Exception as e:
            print(f"❌ SL update failed: {e}")
            return False


# Global trader instance
trader = None


def init_trader():
    """Initialize global trader instance"""
    global trader
    try:
        trader = BybitTrader()
        print(f"✅ Trader initialized with {LEVERAGE}x leverage")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize trader: {e}")
        return False


def generate_chart(df, symbol, timeframe, signal=None, save_path=None):
    """Generate candlestick chart with VWAP and bands"""
    # Use last 150 bars for wider time scale
    df = df.tail(150).copy()
    df = df.reset_index(drop=True)

    # Convert timestamp to datetime in display timezone (for chart labels)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    tz = pytz.timezone(DISPLAY_TIMEZONE)
    df['datetime'] = df['datetime'].dt.tz_convert(tz)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors
    up_color = '#26a69a'  # Green
    down_color = '#ef5350'  # Red
    vwap_color = '#2196f3'  # Blue
    band_color = '#ff9800'  # Orange

    # Plot candlesticks
    width = 0.6
    for i in range(len(df)):
        row = df.iloc[i]
        color = up_color if row['close'] >= row['open'] else down_color

        # Wick
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)

        # Body
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        if body_height == 0:
            body_height = 0.001  # Doji
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # Plot VWAP
    ax.plot(range(len(df)), df['vwap'], color=vwap_color, linewidth=2, label='VWAP')

    # Plot bands
    ax.plot(range(len(df)), df['upper_band'], color=band_color, linewidth=1.5,
            linestyle='--', label=f'Upper Band ({BAND_ENTRY_MULT}σ)')
    ax.plot(range(len(df)), df['lower_band'], color=band_color, linewidth=1.5,
            linestyle='--', label=f'Lower Band ({BAND_ENTRY_MULT}σ)')

    # Fill between bands
    ax.fill_between(range(len(df)), df['upper_band'], df['lower_band'],
                   alpha=0.1, color=band_color)

    # Mark signal on last bar if present
    if signal and signal.get('type'):
        last_idx = len(df) - 1
        last_row = df.iloc[-1]
        if signal['type'] == 'LONG':
            ax.scatter(last_idx, last_row['low'] * 0.998, marker='^',
                      s=200, color='lime', edgecolors='black', zorder=5, label='LONG Signal')
        elif signal['type'] == 'SHORT':
            ax.scatter(last_idx, last_row['high'] * 1.002, marker='v',
                      s=200, color='red', edgecolors='black', zorder=5, label='SHORT Signal')

    # X-axis labels (show every 20th bar)
    x_labels = []
    x_ticks = []
    for i in range(0, len(df), 20):
        x_ticks.append(i)
        x_labels.append(df.iloc[i]['datetime'].strftime('%H:%M'))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)

    # Labels and title
    clean_symbol = symbol.replace(':USDT', '').replace('/USDT', '')
    ax.set_title(f'{clean_symbol} - {timeframe} | VWAP Strategy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add current values as text
    last = df.iloc[-1]
    info_text = (f"Close: {last['close']:.4f}\n"
                f"VWAP: {last['vwap']:.4f}\n"
                f"Upper: {last['upper_band']:.4f}\n"
                f"Lower: {last['lower_band']:.4f}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save chart
    if save_path is None:
        os.makedirs('charts', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signal_type = signal.get('type', '').lower() if signal else ''
        save_path = f"charts/{clean_symbol}_{timeframe}_{signal_type}_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Update charts index for HTML viewer
    update_charts_index()

    print(f"📈 Chart saved: {save_path}")
    return save_path


def update_charts_index():
    """Update the charts/index.json file for the HTML viewer"""
    import json
    import glob

    charts_dir = 'charts'
    if not os.path.exists(charts_dir):
        return

    # Get all PNG files in charts directory
    chart_files = glob.glob(os.path.join(charts_dir, '*.png'))
    chart_files = [os.path.basename(f) for f in chart_files]

    # Sort by modification time (newest first)
    chart_files.sort(key=lambda f: os.path.getmtime(os.path.join(charts_dir, f)), reverse=True)

    # Write index
    index_path = os.path.join(charts_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump({'charts': chart_files, 'updated': datetime.now().isoformat()}, f)


def start_chart_server():
    """Start HTTP server for chart viewer in background thread"""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress request logs
            pass

    try:
        with socketserver.TCPServer(("", CHART_SERVER_PORT), QuietHandler) as httpd:
            print(f"📊 Chart viewer: http://localhost:{CHART_SERVER_PORT}/chart_viewer.html")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"⚠️ Port {CHART_SERVER_PORT} already in use, chart server not started")
        else:
            print(f"⚠️ Chart server error: {e}")


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
            tz = pytz.timezone(SESSION_TIMEZONE)
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
        df['upper_band'] = df['vwap'] + df['stdev'] * BAND_ENTRY_MULT
        df['lower_band'] = df['vwap'] - df['stdev'] * BAND_ENTRY_MULT

        return df[['vwap', 'stdev', 'upper_band', 'lower_band']].reset_index(drop=True)
    
    def check_signal(self, df, symbol, timeframe):
        """Check for entry signals"""
        if len(df) < 15:  # Need enough data for ATR
            return None

        # Check for 09:00 KST no-trade window (±30 minutes)
        if NO_TRADE_AROUND_0900:
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
            if vol_ratio < MIN_VOL_RATIO:
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
            ALLOW_LONGS and
            current['open'] < current['lower_band'] and
            current['close'] > current['lower_band'] and
            bull_strength >= MIN_STRENGTH
        )

        # Check L1/L2 pattern (Short signal)
        # Pine: open > entryUpper and close < entryUpper
        is_l1l2 = (
            ALLOW_SHORTS and
            current['open'] > current['upper_band'] and
            current['close'] < current['upper_band'] and
            bear_strength >= MIN_STRENGTH
        )

        # Debug: Show why signal conditions fail
        debug_info['long_conditions'] = {
            'open < lower_band': current['open'] < current['lower_band'],
            'close > lower_band': current['close'] > current['lower_band'],
            'bull_strength >= min': bull_strength >= MIN_STRENGTH,
        }
        debug_info['short_conditions'] = {
            'open > upper_band': current['open'] > current['upper_band'],
            'close < upper_band': current['close'] < current['upper_band'],
            'bear_strength >= min': bear_strength >= MIN_STRENGTH,
        }

        if is_h1h2:
            # Target based on exit mode
            if EXIT_MODE_LONG == 'VWAP':
                target = current['vwap']
            elif EXIT_MODE_LONG == 'Deviation Band':
                target = current['vwap'] + current['stdev'] * TARGET_LONG_DEVIATION
            else:
                target = None
            return {
                'type': 'LONG',
                'price': current['close'],
                'stop_loss': None,  # Will use swing-based SL
                'target': target,
                'exit_mode': EXIT_MODE_LONG,
                'vwap': current['vwap'],
                'strength': bull_strength,
                'vol_ratio': vol_ratio,
                'debug': debug_info
            }

        if is_l1l2:
            # Target based on exit mode
            if EXIT_MODE_SHORT == 'VWAP':
                target = current['vwap']
            elif EXIT_MODE_SHORT == 'Deviation Band':
                target = current['vwap'] - current['stdev'] * TARGET_SHORT_DEVIATION
            else:
                target = None
            return {
                'type': 'SHORT',
                'price': current['close'],
                'stop_loss': None,  # Will use swing-based SL
                'target': target,
                'exit_mode': EXIT_MODE_SHORT,
                'vwap': current['vwap'],
                'strength': bear_strength,
                'vol_ratio': vol_ratio,
                'debug': debug_info
            }

        # Return debug info even when no signal (for DEBUG_MODE)
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
        self.live_data = {'symbols': [], 'signals': {}, 'signal_history': [], 'timeframe': TIMEFRAMES[0]}

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
        """Update live_data.json for the HTML viewer"""
        import json

        self.live_data['symbols'] = symbols_data
        self.live_data['updated'] = datetime.now().isoformat()

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
            top_symbols = [s['symbol'] for s in usdt_symbols[:TOP_OI_COUNT]]
            
            print(f"📊 Top {TOP_OI_COUNT} OI symbols:")
            for i, s in enumerate(usdt_symbols[:TOP_OI_COUNT], 1):
                print(f"  {i}. {s['symbol']}: ${s['oi_value']:,.0f}")
            
            return top_symbols
            
        except Exception as e:
            print(f"❌ Error getting OI data: {e}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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

        message = f"""
{emoji} <b>{signal_type} SIGNAL</b> {emoji}

<b>Symbol:</b> {symbol.replace(':USDT', '')}
<b>Timeframe:</b> {timeframe}
<b>Entry:</b> ${signal['price']:.4f}
{target_line}

<b>Signal Strength:</b> {signal['strength']:.2%}
<b>Vol Ratio:</b> {signal['vol_ratio']:.2f}

<b>Leverage:</b> {LEVERAGE}x | <b>Size:</b> ${ORDER_SIZE_USDT}
<b>TP:</b> VWAP | <b>SL:</b> Swing trailing

<i>Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
        return message

    def send_signal_with_buttons(self, symbol, timeframe, signal, chart_path=None):
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

        if chart_path:
            # Send photo with caption first, then buttons separately
            send_telegram_photo(chart_path, message)
            # Send buttons as a follow-up message
            button_msg = f"👆 <b>Execute {signal_type} for {clean_symbol}?</b>"
            send_telegram_with_buttons(button_msg, buttons)
        else:
            send_telegram_with_buttons(message, buttons)

        # Store pending signal for callback handler (including TP/SL)
        if trader:
            trader.pending_signals[clean_symbol] = {
                'symbol': symbol,
                'signal_type': signal_type,
                'price': signal['price'],
                'target': signal.get('target'),  # VWAP target for TP
                'timeframe': timeframe,
                'timestamp': time.time()
            }
    
    async def check_symbol(self, symbol):
        """Check a symbol across all timeframes"""
        for timeframe in TIMEFRAMES:
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

                # Check for signal
                result = self.strategy.check_signal(df, symbol, timeframe)

                if result is None:
                    continue

                # Debug output
                if DEBUG_MODE and 'debug' in result:
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
                    print(f"   SL: Swing-based (trailing)")
                    if signal['target'] is not None:
                        print(f"   TP ({signal['exit_mode']}): ${signal['target']:.4f}")
                    else:
                        print(f"   TP: None (Exit Mode: {signal['exit_mode']})")
                    print(f"{'='*60}\n")

                    # Generate and send chart with message and trade buttons
                    chart_path = None
                    if SEND_CHART:
                        try:
                            # Calculate VWAP data for chart
                            df_chart = df.reset_index(drop=True)
                            vwap_df = self.strategy.calculate_vwap(df_chart.copy(), symbol, timeframe)
                            df_chart = pd.concat([df_chart, vwap_df], axis=1)

                            # Generate chart
                            chart_path = generate_chart(df_chart, symbol, timeframe, signal)
                        except Exception as chart_err:
                            print(f"⚠️ Chart generation failed: {chart_err}")
                            chart_path = None

                    # Send signal with trade buttons
                    self.send_signal_with_buttons(symbol, timeframe, signal, chart_path)

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
        print(f"📊 Monitoring top {TOP_OI_COUNT} OI symbols")
        print(f"⏱️  Timeframes: {', '.join(TIMEFRAMES)}")
        print(f"🔄 Check interval: {CHECK_INTERVAL}s")
        print(f"\n📋 Strategy Parameters:")
        print(f"   Session Timezone: {SESSION_TIMEZONE} (VWAP resets at 00:00 {SESSION_TIMEZONE})")
        print(f"   Display Timezone: {DISPLAY_TIMEZONE} (chart x-axis)")
        print(f"   SL: Swing-based (trailing)")
        print(f"   Band Entry Mult: {BAND_ENTRY_MULT}")
        print(f"   Exit Mode Long: {EXIT_MODE_LONG}")
        print(f"   Exit Mode Short: {EXIT_MODE_SHORT}")
        print(f"   Min Strength: {MIN_STRENGTH}")
        print(f"   Min Vol Ratio: {MIN_VOL_RATIO}")
        print(f"   Allow Longs: {ALLOW_LONGS}")
        print(f"   Allow Shorts: {ALLOW_SHORTS}")
        print(f"   No Trade 09:00 KST: {NO_TRADE_AROUND_0900}")
        print(f"   Debug Mode: {DEBUG_MODE}")
        print(f"   Send Chart: {SEND_CHART}\n")

        send_telegram(f"""🚀 <b>Bybit VWAP Monitor Started</b>

Monitoring top {TOP_OI_COUNT} OI symbols
Timeframes: {', '.join(TIMEFRAMES)}
Exit Mode Long: {EXIT_MODE_LONG}
Exit Mode Short: {EXIT_MODE_SHORT}""")
        
        while True:
            try:
                # Get top OI symbols
                symbols = await self.get_top_oi_symbols()

                if not symbols:
                    print("⚠️ No symbols found, retrying...")
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                # Process each symbol - check signals and generate live charts
                symbols_data = []
                timeframe = TIMEFRAMES[0]  # Use first timeframe for live charts

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

                            # Check for current signal
                            result = self.strategy.check_signal(df, symbol, timeframe)
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

                print(f"\n✅ Scan complete. Charts updated. Next scan in {CHECK_INTERVAL}s...")
                await asyncio.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                send_telegram("⏹️ <b>Bybit VWAP Monitor Stopped</b>")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(CHECK_INTERVAL)


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
                    # Get pending signal with TP target
                    pending = trader.pending_signals.get(symbol_short, {})
                    tp_target = pending.get('target')  # VWAP target
                    print(f"🔍 DEBUG tp_target={tp_target}")

                    # Execute trade (SL will be swing-based)
                    result, error = trader.execute_signal_trade(
                        symbol, signal_type,
                        tp_target=tp_target
                    )

                    if result:
                        entry = result['entry_price']
                        qty = result['quantity']
                        tp_price = result['tp_sl'].get('tp_price') if result['tp_sl'] else tp_target
                        sl_price = result['tp_sl'].get('sl_price') if result['tp_sl'] else None

                        tp_str = f"${tp_price:.4f} (VWAP)" if tp_price else "Not set"
                        sl_str = f"${sl_price:.4f} (swing)" if sl_price else "Swing-based"

                        msg = f"""✅ <b>Order Executed!</b>

<b>Symbol:</b> {symbol_short}
<b>Side:</b> {signal_type}
<b>Entry:</b> ${entry:.4f}
<b>Quantity:</b> {qty}
<b>Leverage:</b> {LEVERAGE}x

<b>TP:</b> {tp_str}
<b>SL:</b> {sl_str}

<i>Trailing SL active (1m swing)</i>"""
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
    """Monitor open positions, update trailing SL, and send updates"""
    global trader

    print("📈 Position monitor started")
    if TRAILING_STOP_ENABLED:
        print(f"📈 Trailing stop enabled: {SWING_N} candle swing, {SL_BUFFER_PERCENT}% buffer")

    last_report_time = 0
    last_trail_update = {}  # {symbol: timestamp}
    report_interval = 60  # Report positions every 60 seconds

    while True:
        try:
            if trader:
                positions = trader.get_positions()
                current_time = time.time()

                # ========== Trailing Stop Logic ==========
                if TRAILING_STOP_ENABLED and positions:
                    for pos in positions:
                        symbol = pos['symbol']
                        side = pos['side']  # 'long' or 'short'

                        # Check if enough time passed since last trail update
                        last_update = last_trail_update.get(symbol, 0)
                        if current_time - last_update < TRAIL_UPDATE_INTERVAL:
                            continue

                        # Calculate new trailing SL
                        new_sl = trader.calc_trailing_sl(symbol, side)
                        if new_sl is None:
                            continue

                        # Get current SL from internal tracking
                        tracked = trader.positions.get(symbol, {})
                        current_sl = tracked.get('sl_price')
                        current_tp = tracked.get('tp_price')

                        if current_sl is None:
                            # First time - set initial SL
                            current_sl = new_sl

                        # Apply clamp rule: SL only moves in favorable direction
                        if side == 'long':
                            final_sl = max(current_sl, new_sl)  # SL can only go UP
                        else:
                            final_sl = min(current_sl, new_sl)  # SL can only go DOWN

                        # Only update if SL changed meaningfully (> 0.05%)
                        if abs(final_sl - current_sl) / current_sl > 0.0005:
                            success = trader.update_position_sl(symbol, final_sl, current_tp)
                            if success:
                                # Update internal tracking
                                if symbol in trader.positions:
                                    trader.positions[symbol]['sl_price'] = final_sl
                                last_trail_update[symbol] = current_time

                                # Notify via Telegram (optional)
                                symbol_short = symbol.replace('/USDT:USDT', '').replace(':USDT', '')
                                direction = "⬆️" if side == 'long' else "⬇️"
                                send_telegram(f"{direction} <b>SL Trailed</b>: {symbol_short}\nNew SL: ${final_sl:.4f}")

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
                                pnl_pct = ((mark_price - entry_price) / entry_price) * 100 * LEVERAGE
                            else:
                                pnl_pct = ((entry_price - mark_price) / entry_price) * 100 * LEVERAGE
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

            await asyncio.sleep(POSITION_CHECK_INTERVAL)

        except Exception as e:
            print(f"⚠️ Position monitor error: {e}")
            await asyncio.sleep(POSITION_CHECK_INTERVAL)


async def main():
    """Entry point"""
    global trader

    # Check configuration
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ ERROR: Telegram configuration missing!")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        return

    # Initialize trader
    if BYBIT_API_KEY and BYBIT_API_SECRET:
        init_trader()
    else:
        print("⚠️ Trading disabled - no API keys configured")

    # Start chart viewer server in background
    if SEND_CHART:
        server_thread = threading.Thread(target=start_chart_server, daemon=True)
        server_thread.start()

    # Create tasks
    monitor = BybitMonitor()
    telegram_handler = TelegramBotHandler()

    # Send startup message
    trail_status = "ON" if TRAILING_STOP_ENABLED else "OFF"
    startup_msg = f"""🚀 <b>Bybit VWAP Monitor Started</b>

Monitoring top {TOP_OI_COUNT} OI symbols
Timeframes: {', '.join(TIMEFRAMES)}
Check Interval: {CHECK_INTERVAL}s

<b>Trading Settings:</b>
Leverage: {LEVERAGE}x
Order Size: ${ORDER_SIZE_USDT}
TP: VWAP target
SL: Trailing ({SWING_N} candle swing) [{trail_status}]

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