# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
COIN Daily ATM Weekly Options Momentum Strategy - Production Version

CRITICAL REQUIREMENTS:
======================
1. MUST open exactly ONE trade per day (no exceptions)
2. FIXED position size: $2000 (Use all to get contracts) per day
3. Indicators determine DIRECTION only (CALL vs PUT)
4. If indicators neutral/weak → default to CALL
5. Entry window: 9:30-10:00 AM EST
6. Force exit: 3:55 PM EST (before 4 PM 0DTE expiration)
7. NEVER holds overnight
8. Comprehensive logging and error tracking

STRATEGY LOGIC:
===============
- Uses EMA, RSI, MACD, Bollinger Bands, Volume for direction analysis
- Buys ATM (at-the-money) weekly options based on market direction
- Target profit: 7.5% (configurable)
- Stop loss: 25% (configurable)
- Max hold time: 30 minutes (configurable)
- Automatically exits before market close

LOGGING LEVELS:
===============
- 🚀 Startup/Shutdown (GREEN)
- 📊 Market Analysis (CYAN)
- ✅ Trade Entry (GREEN)
- 📍 Position Updates (GREEN)
- 💰 P&L Events (CYAN/RED)
- ⚠️ Warnings (YELLOW)
- ❌ Errors (RED)
- 🔍 Debug Details (BLUE)
"""

from decimal import Decimal
from datetime import time, datetime, timedelta
import pandas as pd
from typing import Optional, Dict, Any
import pytz
import traceback

from nautilus_trader.config import PositiveInt, PositiveFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.indicators import RelativeStrengthIndex
from nautilus_trader.indicators import BollingerBands
from nautilus_trader.indicators import AverageTrueRange

# Import custom MACD to replace buggy Nautilus one
import sys
sys.path.insert(0, '/var/www/thevolumeai')
from apps.api.nautilus_bridge.custom_indicators import CustomMACD

from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce, OrderType, TriggerType, ContingencyType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument, OptionContract
from nautilus_trader.model.objects import Quantity, Price, Money
from nautilus_trader.model.orders import MarketOrder, LimitOrder
from nautilus_trader.model.orders.list import OrderList
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.common.enums import LogColor


class COINDaily0DTEMomentumConfig(StrategyConfig, frozen=True):
    """
    Configuration for COIN Daily 0DTE Momentum Strategy.
    
    This strategy MUST trade every day with FIXED $2000 position size.
    Indicators only determine CALL vs PUT direction.
    
    Attributes
    ----------
    instrument_id : InstrumentId
        The underlying instrument ID (COIN stock, e.g., COIN.NASDAQ).
    bar_type : BarType
        The bar type for the strategy (recommend 1-minute bars for intraday).
    fixed_position_value : float
        FIXED dollar amount to allocate per trade (default $2000).
    target_profit_pct : Decimal
        Target profit percentage to trigger take profit (default 7.5%).
    stop_loss_pct : Decimal
        Stop loss percentage to trigger exit (default 25%).
    min_hold_minutes : int
        Minimum hold time before stop loss can trigger (default 5 minutes).
    entry_time_start : str
        Entry window start time in HH:MM:SS format EST (default "09:30:00").
    entry_time_end : str
        Entry window end time in HH:MM:SS format EST (default "10:00:00").
    max_hold_minutes : int
        Maximum hold time in minutes before forced exit (default 30).
    force_exit_time : str
        Force exit time in HH:MM:SS format EST (default "15:55:00" - before 4 PM expiration).
    fast_ema_period : int
        Fast EMA period for trend detection (default 9).
    slow_ema_period : int
        Slow EMA period for trend detection (default 20).
    rsi_period : int
        RSI period for momentum analysis (default 14).
    macd_fast_period : int
        MACD fast EMA period (default 12).
    macd_slow_period : int
        MACD slow EMA period (default 26).
    macd_signal_period : int
        MACD signal line period (default 9).
    bb_period : int
        Bollinger Bands period (default 20).
    bb_std_dev : float
        Bollinger Bands standard deviation multiplier (default 2.0).
    min_volume_ratio : float
        Minimum volume ratio vs average for confirmation (default 1.0).
    max_bid_ask_spread_pct : float
        Maximum acceptable bid-ask spread percentage (default 30%).
    min_option_premium : float
        Minimum acceptable option premium per share (default 2.0).
    max_option_premium : float
        Maximum acceptable option premium per share (default 30.0).
    request_bars : bool
        Whether to request historical bars on startup (default True).
    """

    instrument_id: InstrumentId
    bar_type: BarType
    fixed_position_value: PositiveFloat = 2000.0
    target_profit_pct: Decimal = Decimal("7.5")
    stop_loss_pct: Decimal = Decimal("25.0")
    min_hold_minutes: PositiveInt = 5
    entry_time_start: str = "09:30:00"
    entry_time_end: str = "10:00:00"
    max_hold_minutes: PositiveInt = 30
    force_exit_time: str = "15:55:00"
    fast_ema_period: PositiveInt = 9
    slow_ema_period: PositiveInt = 20
    rsi_period: PositiveInt = 14
    macd_fast_period: PositiveInt = 12
    macd_slow_period: PositiveInt = 26
    macd_signal_period: PositiveInt = 9
    bb_period: PositiveInt = 20
    bb_std_dev: PositiveFloat = 2.0
    min_volume_ratio: PositiveFloat = 1.0
    max_bid_ask_spread_pct: PositiveFloat = 30.0
    min_option_premium: PositiveFloat = 2.0
    max_option_premium: PositiveFloat = 30.0
    request_bars: bool = True


class COINDaily0DTEMomentum(Strategy):
    """
    COIN Daily 0DTE Momentum Strategy Implementation.

    EXECUTION FLOW:
    ===============
    1. on_start() - Initialize strategy and indicators
    2. on_bar() - Process each bar:
       a. Check for new trading day (reset flags)
       b. Update volume history
       c. Manage existing position (if any)
       d. Check entry window (9:30-10:00 AM EST)
       e. Analyze market direction (CALL vs PUT)
       f. Enter position (MUST trade daily)
    3. on_order_filled() - Handle order fills:
       a. Entry fill → create SL/TP orders
       b. SL fill → cancel TP
       c. TP fill → cancel SL
    4. on_position_closed() - Log P&L and cleanup
    5. on_stop() - Force close any open positions

    STATE MANAGEMENT:
    =================
    - traded_today: Flag to ensure only one trade per day
    - current_position: Reference to open position (if any)
    - current_option_instrument: Active option contract
    - entry_order_id: ID of entry order for tracking
    - active_sl_order_id: ID of stop loss order
    - active_tp_order_id: ID of take profit order
    - is_closing: Flag to prevent duplicate close attempts

    ERROR HANDLING:
    ===============
    - All critical operations wrapped in try/except
    - Detailed error logging with stack traces
    - Fallback mechanisms for missing data
    - Graceful degradation (trade anyway if possible)
    """

    def __init__(self, config: COINDaily0DTEMomentumConfig) -> None:
        """
        Initialize the COIN Daily 0DTE Momentum Strategy.
        
        Validates configuration, initializes indicators, and sets up state tracking.
        
        Parameters
        ----------
        config : COINDaily0DTEMomentumConfig
            Strategy configuration parameters.
        
        Raises
        ------
        ValueError
            If fast_ema_period >= slow_ema_period.
        """
        self.log.info("=" * 80, LogColor.BLUE)
        self.log.info("🔧 INITIALIZING COIN DAILY 0DTE MOMENTUM STRATEGY", LogColor.BLUE)
        self.log.info("=" * 80, LogColor.BLUE)
        
        # Validate EMA periods
        PyCondition.is_true(
            config.fast_ema_period < config.slow_ema_period,
            f"fast_ema_period ({config.fast_ema_period}) must be < slow_ema_period ({config.slow_ema_period})",
        )
        
        super().__init__(config)

        # Instrument reference
        self.instrument: Optional[Instrument] = None
        self.log.info(f"🔍 Target instrument: {config.instrument_id}", LogColor.BLUE)

        # Initialize technical indicators
        self.log.info("📊 Initializing technical indicators...", LogColor.BLUE)
        try:
            self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
            self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
            self.rsi = RelativeStrengthIndex(config.rsi_period)
            self.macd = CustomMACD(
                config.macd_fast_period,
                config.macd_slow_period,
                config.macd_signal_period,
            )
            self.bb = BollingerBands(period=config.bb_period, k=config.bb_std_dev)
            self.atr = AverageTrueRange(period=14)
            
            self.log.info(f"   ✓ EMA: Fast={config.fast_ema_period}, Slow={config.slow_ema_period}", LogColor.BLUE)
            self.log.info(f"   ✓ RSI: Period={config.rsi_period}", LogColor.BLUE)
            self.log.info(f"   ✓ MACD: {config.macd_fast_period}/{config.macd_slow_period}/{config.macd_signal_period}", LogColor.BLUE)
            self.log.info(f"   ✓ BB: Period={config.bb_period}, StdDev={config.bb_std_dev}", LogColor.BLUE)
            self.log.info(f"   ✓ ATR: Period=14", LogColor.BLUE)
        except Exception as e:
            self.log.error(f"❌ CRITICAL: Failed to initialize indicators: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            raise

        # Trading state variables
        self.traded_today = False
        self.last_trade_date: Optional[datetime] = None
        self.entry_price: Optional[float] = None
        self.entry_timestamp: Optional[int] = None
        self.current_position: Optional[Position] = None
        self.current_option_instrument: Optional[OptionContract] = None
        self.pending_option_instrument: Optional[OptionContract] = None
        
        # Order tracking
        self.entry_order_id = None
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.is_closing = False

        # Volume tracking for confirmation
        self.volume_history = []
        self.max_volume_samples = 20

        # Parse time strings to time objects
        try:
            self.entry_time_start_parsed = self._parse_time(config.entry_time_start)
            self.entry_time_end_parsed = self._parse_time(config.entry_time_end)
            self.force_exit_time_parsed = self._parse_time(config.force_exit_time)
            
            self.log.info(f"⏰ Entry window: {self.entry_time_start_parsed} - {self.entry_time_end_parsed} EST", LogColor.BLUE)
            self.log.info(f"⏰ Force exit: {self.force_exit_time_parsed} EST", LogColor.BLUE)
        except Exception as e:
            self.log.error(f"❌ CRITICAL: Failed to parse time strings: {e}", LogColor.RED)
            raise
        
        # EST timezone for proper time handling
        try:
            self.est_tz = pytz.timezone('America/New_York')
            self.log.info(f"🌍 Timezone: America/New_York (handles EST/EDT automatically)", LogColor.BLUE)
        except Exception as e:
            self.log.error(f"❌ CRITICAL: Failed to initialize timezone: {e}", LogColor.RED)
            raise

        # Performance tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        
        self.log.info("✅ Strategy initialization complete", LogColor.GREEN)
        self.log.info("=" * 80, LogColor.BLUE)

    def _parse_time(self, time_str: str) -> time:
        """
        Parse time string in HH:MM:SS format to Python time object.
        
        Parameters
        ----------
        time_str : str
            Time string in format "HH:MM:SS" (e.g., "09:30:00").
        
        Returns
        -------
        time
            Python time object.
        
        Raises
        ------
        ValueError
            If time string cannot be parsed.
        
        Examples
        --------
        >>> _parse_time("09:30:00")
        time(9, 30, 0)
        """
        try:
            parts = time_str.split(":")
            if len(parts) != 3:
                raise ValueError(f"Expected format HH:MM:SS, got {time_str}")
            
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2])
            
            if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                raise ValueError(f"Invalid time values in {time_str}")
            
            return time(hour, minute, second)
        except Exception as e:
            self.log.error(f"❌ Failed to parse time string '{time_str}': {e}", LogColor.RED)
            # Default to 9:30 AM as safe fallback
            self.log.warning("⚠️ Using default time 09:30:00", LogColor.YELLOW)
            return time(9, 30, 0)

    def on_start(self) -> None:
        """
        Actions performed on strategy start.
        
        Responsibilities:
        1. Resolve instrument from cache
        2. Register indicators for automatic bar updates
        3. Subscribe to bar data
        4. Request historical data for indicator warm-up
        5. Log startup configuration
        
        Called by NautilusTrader when strategy is started.
        """
        self.log.info("=" * 80, LogColor.GREEN)
        self.log.info("🚀 STARTING COIN DAILY 0DTE MOMENTUM STRATEGY", LogColor.GREEN)
        self.log.info("=" * 80, LogColor.GREEN)
        
        # Resolve instrument from cache
        try:
            self.instrument = self.cache.instrument(self.config.instrument_id)
            if self.instrument is None:
                self.log.warning(
                    f"⚠️ Instrument {self.config.instrument_id} not found in cache",
                    LogColor.YELLOW
                )
                self.log.warning("   Will be resolved when bars arrive", LogColor.YELLOW)
            else:
                self.log.info(f"✓ Instrument resolved: {self.instrument.id.symbol}", LogColor.GREEN)
        except Exception as e:
            self.log.error(f"❌ Error resolving instrument: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

        # Log strategy configuration
        self.log.info("📋 STRATEGY CONFIGURATION:", LogColor.CYAN)
        self.log.info(f"   Instrument: {self.config.instrument_id}", LogColor.CYAN)
        self.log.info(f"   Bar Type: {self.config.bar_type}", LogColor.CYAN)
        self.log.info(f"   FIXED Position: ${self.config.fixed_position_value:.2f} per day", LogColor.CYAN)
        self.log.info(f"   Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST", LogColor.CYAN)
        self.log.info(f"   Force Exit: {self.config.force_exit_time} EST", LogColor.CYAN)
        self.log.info(f"   Target Profit: +{self.config.target_profit_pct}%", LogColor.CYAN)
        self.log.info(f"   Stop Loss: -{self.config.stop_loss_pct}%", LogColor.CYAN)
        self.log.info(f"   Max Hold Time: {self.config.max_hold_minutes} minutes", LogColor.CYAN)
        self.log.info(f"   Min Hold Time: {self.config.min_hold_minutes} minutes (before SL)", LogColor.CYAN)

        # Register indicators for automatic updates
        try:
            self.log.info("📊 Registering indicators for automatic updates...", LogColor.BLUE)
            self.register_indicator_for_bars(self.config.bar_type, self.fast_ema)
            self.register_indicator_for_bars(self.config.bar_type, self.slow_ema)
            self.register_indicator_for_bars(self.config.bar_type, self.rsi)
            self.register_indicator_for_bars(self.config.bar_type, self.macd)
            self.register_indicator_for_bars(self.config.bar_type, self.bb)
            self.register_indicator_for_bars(self.config.bar_type, self.atr)
            self.log.info("   ✓ All indicators registered", LogColor.GREEN)
        except Exception as e:
            self.log.error(f"❌ Error registering indicators: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

        # Subscribe to bars
        try:
            self.log.info(f"📡 Subscribing to bars: {self.config.bar_type}", LogColor.BLUE)
            self.subscribe_bars(self.config.bar_type)
            self.log.info("   ✓ Bar subscription active", LogColor.GREEN)
        except Exception as e:
            self.log.error(f"❌ Error subscribing to bars: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

        # Request historical bars for indicator warm-up
        if self.config.request_bars:
            try:
                lookback_days = 5
                start_time = self._clock.utc_now() - pd.Timedelta(days=lookback_days)
                self.log.info(f"📜 Requesting historical bars (last {lookback_days} days) for indicator warm-up...", LogColor.BLUE)
                self.request_bars(self.config.bar_type, start=start_time)
                self.log.info("   ✓ Historical bar request submitted", LogColor.GREEN)
            except Exception as e:
                self.log.error(f"❌ Error requesting historical bars: {e}", LogColor.RED)
                self.log.error(traceback.format_exc(), LogColor.RED)
                self.log.warning("   ⚠️ Continuing without historical data", LogColor.YELLOW)

        self.log.info("=" * 80, LogColor.GREEN)
        self.log.info("✅ STRATEGY STARTED - WAITING FOR ENTRY WINDOW", LogColor.GREEN)
        self.log.info("=" * 80, LogColor.GREEN)

    def on_bar(self, bar: Bar) -> None:
        """
        Main event handler for incoming bar data.
        
        This is the core logic loop that executes on every bar update.
        
        Execution Flow:
        ---------------
        1. Convert bar time to EST timezone
        2. Check for new trading day (reset daily flags)
        3. Update volume history
        4. Manage existing position (if any)
        5. Check if already traded today
        6. Check if within entry window
        7. Analyze market direction (CALL vs PUT)
        8. Enter position (MUST trade daily with $2000)
        
        Parameters
        ----------
        bar : Bar
            The bar data received from the data feed.
        
        Notes
        -----
        - This method is called by NautilusTrader for each bar update
        - All exceptions are caught to prevent strategy crash
        - Comprehensive logging at each decision point
        """
        try:
            # Convert bar time to EST for proper time handling
            bar_time_est = datetime.fromtimestamp(
                bar.ts_event / 1_000_000_000, 
                tz=self.est_tz
            )
            current_date = bar_time_est.date()
            
            # Log bar details (every 10th bar to avoid spam)
            if bar.ts_event % 10 == 0:
                self.log.info(
                    f"📊 Bar: {bar_time_est.strftime('%Y-%m-%d %H:%M:%S EST')} | "
                    f"O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume}",
                    LogColor.BLUE
                )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Check for new trading day (reset daily flags)
            # ═══════════════════════════════════════════════════════════
            if self.last_trade_date is not None and current_date > self.last_trade_date:
                self.log.info("=" * 80, LogColor.CYAN)
                self.log.info(f"🔄 NEW TRADING DAY: {current_date}", LogColor.CYAN)
                self.log.info("=" * 80, LogColor.CYAN)
                
                # Log previous day's stats
                self.log.info(f"📊 Previous Day Stats:", LogColor.CYAN)
                self.log.info(f"   Trades: {self.trades_today}", LogColor.CYAN)
                self.log.info(f"   P&L: ${self.daily_pnl:.2f}", LogColor.CYAN)
                
                # Reset daily flags
                self.traded_today = False
                self.trades_today = 0
                self.daily_pnl = 0.0
                
                # Cancel any orphaned orders from previous day
                self._cancel_all_tracked_orders("New trading day - cleaning up previous day's orders")
                
                self.log.info("✅ Daily reset complete - ready for new trading day", LogColor.GREEN)

            # ═══════════════════════════════════════════════════════════
            # STEP 2: Update volume history for analysis
            # ═══════════════════════════════════════════════════════════
            try:
                self.volume_history.append(float(bar.volume))
                if len(self.volume_history) > self.max_volume_samples:
                    self.volume_history.pop(0)
            except Exception as e:
                self.log.warning(f"⚠️ Error updating volume history: {e}", LogColor.YELLOW)

            # ═══════════════════════════════════════════════════════════
            # STEP 3: Manage existing position (if any)
            # ═══════════════════════════════════════════════════════════
            if self.current_position is not None:
                self._manage_position(bar, bar_time_est)
                return  # Skip entry logic if managing position

            # ═══════════════════════════════════════════════════════════
            # STEP 4: Check if already traded today
            # ═══════════════════════════════════════════════════════════
            if self.traded_today:
                # Only log once per day to avoid spam
                if bar_time_est.hour == 9 and bar_time_est.minute == 31:
                    self.log.info(
                        f"✓ Already traded today ({current_date}) - waiting for next day",
                        LogColor.CYAN
                    )
                return

            # ═══════════════════════════════════════════════════════════
            # STEP 5: Check if within entry window
            # ═══════════════════════════════════════════════════════════
            if not self._is_entry_time(bar_time_est):
                return

            # ═══════════════════════════════════════════════════════════
            # STEP 6: Analyze market direction (CALL vs PUT)
            # CRITICAL: MUST TRADE - indicators only determine direction
            # ═══════════════════════════════════════════════════════════
            self.log.info("=" * 80, LogColor.MAGENTA)
            self.log.info("🎯 ENTRY WINDOW ACTIVE - ANALYZING MARKET", LogColor.MAGENTA)
            self.log.info("=" * 80, LogColor.MAGENTA)
            
            if self._indicators_ready():
                self.log.info("✓ Indicators ready - analyzing market direction", LogColor.GREEN)
                signal = self._analyze_market_direction(bar)
            else:
                self.log.warning(
                    "⚠️ Indicators NOT ready - DEFAULTING TO CALL (MUST TRADE DAILY)",
                    LogColor.YELLOW
                )
                self.log.warning(
                    "   This is expected on the first few bars - indicators need warm-up data",
                    LogColor.YELLOW
                )
                signal = "BULLISH"  # Default to CALL

            # ═══════════════════════════════════════════════════════════
            # STEP 7: Enter position (MUST TRADE - no skipping)
            # ═══════════════════════════════════════════════════════════
            self._enter_position(bar, bar_time_est, signal)

        except Exception as e:
            self.log.error(f"❌ CRITICAL ERROR in on_bar(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            self.log.error("   Strategy will attempt to continue...", LogColor.RED)

    def _is_entry_time(self, bar_time_est: datetime) -> bool:
        """
        Check if current bar falls within the configured entry window.
        
        Entry window is defined by entry_time_start and entry_time_end.
        Default: 9:30 AM - 10:00 AM EST
        
        Parameters
        ----------
        bar_time_est : datetime
            Bar timestamp converted to EST timezone.
        
        Returns
        -------
        bool
            True if within entry window, False otherwise.
        
        Notes
        -----
        - Uses timezone-aware comparison (handles EST/EDT automatically)
        - Logs when entry window becomes active
        - Only trades once per entry window (controlled by traded_today flag)
        """
        try:
            bar_time_only = bar_time_est.time()
            
            # Check if within entry window
            in_window = (
                self.entry_time_start_parsed <= bar_time_only <= self.entry_time_end_parsed
            )
            
            if in_window and not self.traded_today:
                self.log.info("=" * 80, LogColor.MAGENTA)
                self.log.info(
                    f"⏰ ENTRY WINDOW ACTIVE: {bar_time_only} EST "
                    f"(Window: {self.entry_time_start_parsed} - {self.entry_time_end_parsed})",
                    LogColor.MAGENTA
                )
                self.log.info("=" * 80, LogColor.MAGENTA)
            
            return in_window
            
        except Exception as e:
            self.log.error(f"❌ Error checking entry time: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            return False

    def _indicators_ready(self) -> bool:
        """
        Check if all technical indicators have been initialized with sufficient data.
        
        Indicators need historical bars to "warm up" before they can produce valid values.
        For example, a 20-period EMA needs at least 20 bars of data.
        
        Returns
        -------
        bool
            True if all indicators are initialized, False otherwise.
        
        Notes
        -----
        - Called before entering positions
        - If False, strategy defaults to CALL direction
        - Does NOT prevent trading (MUST TRADE DAILY requirement)
        """
        try:
            indicators_status = {
                'Fast EMA': self.fast_ema.initialized,
                'Slow EMA': self.slow_ema.initialized,
                'RSI': self.rsi.initialized,
                'MACD': self.macd.initialized,
                'BB': self.bb.initialized,
                'ATR': self.atr.initialized,
            }
            
            all_ready = all(indicators_status.values())
            
            if not all_ready:
                self.log.info("📊 Indicator Status:", LogColor.YELLOW)
                for name, status in indicators_status.items():
                    status_icon = "✓" if status else "✗"
                    color = LogColor.GREEN if status else LogColor.RED
                    self.log.info(f"   {status_icon} {name}: {'Ready' if status else 'Warming up'}", color)
            
            return all_ready
            
        except Exception as e:
            self.log.error(f"❌ Error checking indicator readiness: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            return False

    def _analyze_market_direction(self, bar: Bar) -> str:
        """
        Analyze market conditions to determine trade direction (CALL vs PUT).
        
        CRITICAL: This function ALWAYS returns "BULLISH" or "BEARISH" (never skips trade).
        Uses multi-factor technical analysis with weighted scoring system.
        
        Scoring System:
        ---------------
        EMA Trend       : 3 points (primary signal)
        RSI Momentum    : 2 points
        MACD Trend      : 2 points
        Bollinger Bands : 1 point
        Volume Confirm  : 1 point (confirms leading direction)
        
        Decision Logic:
        ---------------
        - If bullish_score > bearish_score → "BULLISH" (buy CALL)
        - If bearish_score > bullish_score → "BEARISH" (buy PUT)
        - If tied → "BULLISH" (default to CALL)
        
        Parameters
        ----------
        bar : Bar
            Current bar data for price/volume analysis.
        
        Returns
        -------
        str
            "BULLISH" (buy CALL) or "BEARISH" (buy PUT).
        
        Notes
        -----
        - Comprehensive logging of each indicator's contribution
        - Never returns None or "NEUTRAL" (MUST TRADE requirement)
        - All exceptions caught with fallback to BULLISH
        """
        try:
            self.log.info("=" * 80, LogColor.CYAN)
            self.log.info("📊 MARKET DIRECTION ANALYSIS", LogColor.CYAN)
            self.log.info("=" * 80, LogColor.CYAN)
            
            bullish_score = 0
            bearish_score = 0
            
            # ═══════════════════════════════════════════════════════════
            # INDICATOR 1: EMA Trend Analysis (Primary Signal - 3 points)
            # ═══════════════════════════════════════════════════════════
            try:
                fast_ema_val = self.fast_ema.value
                slow_ema_val = self.slow_ema.value
                
                if fast_ema_val > slow_ema_val:
                    bullish_score += 3
                    self.log.info(
                        f"   ✓ EMA TREND: BULLISH (+3) | Fast={fast_ema_val:.2f} > Slow={slow_ema_val:.2f}",
                        LogColor.GREEN
                    )
                else:
                    bearish_score += 3
                    self.log.info(
                        f"   ✓ EMA TREND: BEARISH (+3) | Fast={fast_ema_val:.2f} < Slow={slow_ema_val:.2f}",
                        LogColor.RED
                    )
            except Exception as e:
                self.log.warning(f"   ⚠️ EMA analysis failed: {e}", LogColor.YELLOW)

            # ═══════════════════════════════════════════════════════════
            # INDICATOR 2: RSI Momentum (2 points)
            # ═══════════════════════════════════════════════════════════
            try:
                rsi_val = self.rsi.value
                
                if rsi_val > 50:
                    bullish_score += 2
                    self.log.info(
                        f"   ✓ RSI MOMENTUM: BULLISH (+2) | RSI={rsi_val:.1f} (above 50)",
                        LogColor.GREEN
                    )
                else:
                    bearish_score += 2
                    self.log.info(
                        f"   ✓ RSI MOMENTUM: BEARISH (+2) | RSI={rsi_val:.1f} (below 50)",
                        LogColor.RED
                    )
                
                # Log special conditions
                if rsi_val > 70:
                    self.log.info(f"      ⚠️ Overbought territory (RSI > 70)", LogColor.YELLOW)
                elif rsi_val < 30:
                    self.log.info(f"      ⚠️ Oversold territory (RSI < 30)", LogColor.YELLOW)
                    
            except Exception as e:
                self.log.warning(f"   ⚠️ RSI analysis failed: {e}", LogColor.YELLOW)

            # ═══════════════════════════════════════════════════════════
            # INDICATOR 3: MACD Trend Strength (2 points)
            # ═══════════════════════════════════════════════════════════
            try:
                macd_val = self.macd.value
                macd_signal = self.macd.signal
                
                if macd_val > macd_signal:
                    bullish_score += 2
                    self.log.info(
                        f"   ✓ MACD TREND: BULLISH (+2) | MACD={macd_val:.2f} > Signal={macd_signal:.2f}",
                        LogColor.GREEN
                    )
                else:
                    bearish_score += 2
                    self.log.info(
                        f"   ✓ MACD TREND: BEARISH (+2) | MACD={macd_val:.2f} < Signal={macd_signal:.2f}",
                        LogColor.RED
                    )
            except Exception as e:
                self.log.warning(f"   ⚠️ MACD analysis failed: {e}", LogColor.YELLOW)

            # ═══════════════════════════════════════════════════════════
            # INDICATOR 4: Bollinger Bands Position (1 point)
            # ═══════════════════════════════════════════════════════════
            try:
                price = float(bar.close)
                bb_upper = self.bb.upper
                bb_middle = self.bb.middle
                bb_lower = self.bb.lower
                
                if price > bb_middle:
                    bullish_score += 1
                    self.log.info(
                        f"   ✓ BOLLINGER BANDS: BULLISH (+1) | Price=${price:.2f} > Middle=${bb_middle:.2f}",
                        LogColor.GREEN
                    )
                else:
                    bearish_score += 1
                    self.log.info(
                        f"   ✓ BOLLINGER BANDS: BEARISH (+1) | Price=${price:.2f} < Middle=${bb_middle:.2f}",
                        LogColor.RED
                    )
                
                # Log band positions for context
                self.log.info(
                    f"      BB Bands: Upper=${bb_upper:.2f}, Middle=${bb_middle:.2f}, Lower=${bb_lower:.2f}",
                    LogColor.BLUE
                )
                
            except Exception as e:
                self.log.warning(f"   ⚠️ Bollinger Bands analysis failed: {e}", LogColor.YELLOW)

            # ═══════════════════════════════════════════════════════════
            # INDICATOR 5: Volume Confirmation (1 point)
            # ═══════════════════════════════════════════════════════════
            try:
                if len(self.volume_history) >= 5:
                    avg_volume = sum(self.volume_history[:-1]) / (len(self.volume_history) - 1)
                    current_volume = float(bar.volume)
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    
                    if volume_ratio >= self.config.min_volume_ratio:
                        # High volume - confirm the leading direction
                        if bullish_score > bearish_score:
                            bullish_score += 1
                            self.log.info(
                                f"   ✓ VOLUME: Confirms BULLISH (+1) | "
                                f"Current={current_volume:,.0f}, Avg={avg_volume:,.0f}, Ratio={volume_ratio:.2f}x",
                                LogColor.GREEN
                            )
                        elif bearish_score > bullish_score:
                            bearish_score += 1
                            self.log.info(
                                f"   ✓ VOLUME: Confirms BEARISH (+1) | "
                                f"Current={current_volume:,.0f}, Avg={avg_volume:,.0f}, Ratio={volume_ratio:.2f}x",
                                LogColor.RED
                            )
                        else:
                            # Tied - don't add to either
                            self.log.info(
                                f"   ○ VOLUME: Neutral (tied scores) | Ratio={volume_ratio:.2f}x",
                                LogColor.YELLOW
                            )
                    else:
                        self.log.info(
                            f"   ⚠️ VOLUME: Low ({volume_ratio:.2f}x avg) - no confirmation",
                            LogColor.YELLOW
                        )
                else:
                    self.log.info(
                        f"   ○ VOLUME: Insufficient history ({len(self.volume_history)} bars, need 5)",
                        LogColor.YELLOW
                    )
            except Exception as e:
                self.log.warning(f"   ⚠️ Volume analysis failed: {e}", LogColor.YELLOW)

            # ═══════════════════════════════════════════════════════════
            # FINAL DECISION
            # ═══════════════════════════════════════════════════════════
            self.log.info("=" * 80, LogColor.CYAN)
            self.log.info(
                f"📊 FINAL SCORE: BULLISH={bullish_score} | BEARISH={bearish_score}",
                LogColor.CYAN
            )
            
            if bullish_score > bearish_score:
                direction = "BULLISH"
                option_type = "CALL"
                color = LogColor.GREEN
                icon = "📈"
            elif bearish_score > bullish_score:
                direction = "BEARISH"
                option_type = "PUT"
                color = LogColor.RED
                icon = "📉"
            else:
                # Tied - default to CALL
                direction = "BULLISH"
                option_type = "CALL"
                color = LogColor.YELLOW
                icon = "⚖️"
                self.log.info("   ⚖️ Scores TIED - defaulting to CALL", LogColor.YELLOW)
            
            self.log.info(
                f"{icon} DECISION: {direction} → BUY {option_type}",
                color
            )
            self.log.info("=" * 80, LogColor.CYAN)
            
            return direction
            
        except Exception as e:
            self.log.error(f"❌ CRITICAL ERROR in market analysis: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            self.log.warning("⚠️ FALLBACK: Defaulting to BULLISH (CALL)", LogColor.YELLOW)
            return "BULLISH"

    def _enter_position(self, bar: Bar, bar_time_est: datetime, signal: str) -> None:
        """
        Enter a trading position based on market signal.
        
        CRITICAL: This function MUST successfully open a position (MUST TRADE requirement).
        If any validation fails, it attempts fallback mechanisms rather than skipping.
        
        Execution Steps:
        ----------------
        1. Determine option type (CALL or PUT) from signal
        2. Calculate ATM strike price (nearest $5)
        3. Find best available option contract
        4. Get option premium from market data
        5. Validate premium (with relaxed tolerances)
        6. Calculate position size ($2000 fixed)
        7. Cancel any existing orders
        8. Submit entry order
        9. Store state for order tracking
        
        Parameters
        ----------
        bar : Bar
            Current bar data for pricing.
        bar_time_est : datetime
            Bar timestamp in EST timezone.
        signal : str
            Market direction: "BULLISH" (CALL) or "BEARISH" (PUT).
        
        Notes
        -----
        - Uses MARKET orders for guaranteed fills
        - SL/TP orders created after entry fill (in on_order_filled)
        - Comprehensive logging of each step
        - Fallback mechanisms if primary method fails
        """
        try:
            self.log.info("=" * 80, LogColor.GREEN)
            self.log.info("🎯 ENTERING POSITION", LogColor.GREEN)
            self.log.info("=" * 80, LogColor.GREEN)
            
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Determine option type based on signal
            # ═══════════════════════════════════════════════════════════
            if signal == "BULLISH":
                order_side = OrderSide.BUY
                option_type = "CALL"
                self.log.info(f"📈 Signal: BULLISH → Buying CALL option", LogColor.GREEN)
            else:
                order_side = OrderSide.BUY
                option_type = "PUT"
                self.log.info(f"📉 Signal: BEARISH → Buying PUT option", LogColor.RED)

            underlying_price = float(bar.close)
            today_date = bar_time_est.date()
            
            self.log.info(f"📍 Underlying Price: ${underlying_price:.2f}", LogColor.CYAN)
            self.log.info(f"📅 Trade Date: {today_date}", LogColor.CYAN)

            # ═══════════════════════════════════════════════════════════
            # STEP 2: Calculate ATM strike (nearest $5 for COIN)
            # ═══════════════════════════════════════════════════════════
            atm_strike = round(underlying_price / 5) * 5
            self.log.info(f"🎯 ATM Strike: ${atm_strike:.2f} (rounded from ${underlying_price:.2f})", LogColor.CYAN)

            # ═══════════════════════════════════════════════════════════
            # STEP 3: Find best available option contract
            # ═══════════════════════════════════════════════════════════
            self.log.info("🔍 Searching for best option contract...", LogColor.BLUE)
            option_instrument = self._find_best_option(
                today_date, 
                atm_strike, 
                underlying_price,
                option_type
            )

            if option_instrument is None:
                self.log.error(f"❌ No {option_type} contract found!", LogColor.RED)
                
                # FALLBACK: Try opposite direction
                option_type = "PUT" if option_type == "CALL" else "CALL"
                self.log.warning(f"⚠️ FALLBACK: Trying {option_type} instead", LogColor.YELLOW)
                
                option_instrument = self._find_best_option(
                    today_date, 
                    atm_strike, 
                    underlying_price,
                    option_type
                )
                
                if option_instrument is None:
                    self.log.error("❌ CRITICAL: No options available - CANNOT TRADE TODAY", LogColor.RED)
                    return

            # ═══════════════════════════════════════════════════════════
            # STEP 4: Get option premium from market data
            # ═══════════════════════════════════════════════════════════
            self.log.info("💰 Fetching option premium from market...", LogColor.BLUE)
            estimated_premium = self._get_option_premium_relaxed(
                option_instrument, 
                underlying_price
            )
            
            if estimated_premium is None:
                self.log.error("❌ Cannot determine premium - CANNOT TRADE TODAY", LogColor.RED)
                return

            # ═══════════════════════════════════════════════════════════
            # STEP 5: Validate premium (with RELAXED tolerances)
            # ═══════════════════════════════════════════════════════════
            self.log.info("✓ Validating premium range...", LogColor.BLUE)
            
            if estimated_premium < self.config.min_option_premium:
                self.log.warning(
                    f"⚠️ Premium ${estimated_premium:.2f} below min ${self.config.min_option_premium:.2f} "
                    f"- USING ANYWAY (MUST TRADE)",
                    LogColor.YELLOW
                )
            elif estimated_premium > self.config.max_option_premium:
                self.log.warning(
                    f"⚠️ Premium ${estimated_premium:.2f} above max ${self.config.max_option_premium:.2f} "
                    f"- USING ANYWAY (MUST TRADE)",
                    LogColor.YELLOW
                )
            else:
                self.log.info(
                    f"✓ Premium ${estimated_premium:.2f} within acceptable range "
                    f"(${self.config.min_option_premium:.2f} - ${self.config.max_option_premium:.2f})",
                    LogColor.GREEN
                )

            # ═══════════════════════════════════════════════════════════
            # STEP 6: Calculate position size (FIXED $2000)
            # ═══════════════════════════════════════════════════════════
            self.log.info("📊 Calculating position size...", LogColor.BLUE)
            
            contract_cost = estimated_premium * 100  # Each contract = 100 shares
            num_contracts = max(1, int(self.config.fixed_position_value / contract_cost))
            actual_cost = num_contracts * contract_cost
            
            quantity = Quantity.from_int(num_contracts)

            self.log.info(f"   Contract Cost: ${contract_cost:.2f} (${estimated_premium:.2f} × 100)", LogColor.CYAN)
            self.log.info(f"   Contracts: {num_contracts}", LogColor.CYAN)
            self.log.info(f"   Total Position Value: ${actual_cost:.2f}", LogColor.CYAN)
            self.log.info(
                f"   {'✓' if actual_cost <= self.config.fixed_position_value * 1.1 else '⚠️'} "
                f"Target: ${self.config.fixed_position_value:.2f}",
                LogColor.GREEN if actual_cost <= self.config.fixed_position_value * 1.1 else LogColor.YELLOW
            )

            # ═══════════════════════════════════════════════════════════
            # STEP 7: Cancel any existing orders (cleanup)
            # ═══════════════════════════════════════════════════════════
            try:
                self.log.info("🗑️  Cancelling any existing orders for this instrument...", LogColor.YELLOW)
                self.cancel_all_orders(option_instrument.id)
                self.log.info("   ✓ Order cleanup complete", LogColor.GREEN)
            except Exception as e:
                self.log.warning(f"   ⚠️ Order cancellation failed: {e}", LogColor.YELLOW)

            # ═══════════════════════════════════════════════════════════
            # STEP 8: Submit entry order (MARKET order for guaranteed fill)
            # ═══════════════════════════════════════════════════════════
            self.log.info("📤 Submitting entry order...", LogColor.BLUE)
            
            entry_order = self.order_factory.market(
                instrument_id=option_instrument.id,
                order_side=order_side,
                quantity=quantity,
                time_in_force=TimeInForce.GTC,
            )

            self.submit_order(entry_order)

            # ═══════════════════════════════════════════════════════════
            # STEP 9: Store state for order tracking
            # ═══════════════════════════════════════════════════════════
            self.entry_order_id = entry_order.client_order_id
            self.pending_option_instrument = option_instrument
            self.current_option_instrument = option_instrument
            self.traded_today = True
            self.last_trade_date = today_date
            self.entry_timestamp = bar.ts_event
            self.trades_today += 1

            self.log.info("=" * 80, LogColor.GREEN)
            self.log.info(
                f"✅ ENTRY ORDER SUBMITTED",
                LogColor.GREEN
            )
            self.log.info(f"   Order ID: {entry_order.client_order_id}", LogColor.GREEN)
            self.log.info(f"   Direction: {order_side}", LogColor.GREEN)
            self.log.info(f"   Instrument: {option_instrument.id.symbol}", LogColor.GREEN)
            self.log.info(f"   Quantity: {num_contracts} contracts", LogColor.GREEN)
            self.log.info(f"   Estimated Cost: ${actual_cost:.2f}", LogColor.GREEN)
            self.log.info("   ⏳ Waiting for fill... (SL/TP will be set after fill)", LogColor.YELLOW)
            self.log.info("=" * 80, LogColor.GREEN)

        except Exception as e:
            self.log.error(f"❌ CRITICAL ERROR in _enter_position(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            self.log.error("   Failed to enter position - will try again on next entry window", LogColor.RED)

    def _find_best_option(
        self, 
        today_date: datetime.date, 
        target_strike: float,
        underlying_price: float,
        option_type: str
    ) -> Optional[OptionContract]:
        """
        Find the best ATM option contract matching criteria.
        
        Selection Criteria (Priority Order):
        -------------------------------------
        1. Correct option type (CALL or PUT)
        2. Not expired (expiration >= today)
        3. Nearest expiration date (prefer weekly 0DTE)
        4. Closest to target strike (ATM)
        
        Parameters
        ----------
        today_date : datetime.date
            Current trading date.
        target_strike : float
            Target ATM strike price.
        underlying_price : float
            Current underlying price (for logging).
        option_type : str
            "CALL" or "PUT".
        
        Returns
        -------
        Optional[OptionContract]
            Best matching option contract, or None if no suitable contract found.
        
        Notes
        -----
        - Searches all instruments in cache
        - Filters by COIN symbol
        - Comprehensive logging of search process
        """
        try:
            self.log.info(f"   🔍 Searching for {option_type} options...", LogColor.BLUE)
            self.log.info(f"      Target Strike: ${target_strike:.2f}", LogColor.BLUE)
            self.log.info(f"      Expiration: >= {today_date}", LogColor.BLUE)
            
            all_instruments = self.cache.instruments()
            self.log.info(f"      Total instruments in cache: {len(all_instruments)}", LogColor.BLUE)
            
            best_match = None
            best_expiry_days = float('inf')
            best_strike_diff = float('inf')
            
            coin_options_found = 0
            matching_type_found = 0
            valid_contracts = []

            for inst in all_instruments:
                # Check if it's an OptionContract
                if 'Option' not in str(type(inst)):
                    continue
                    
                # Check if it's a COIN option
                if 'COIN' not in str(inst.id):
                    continue
                
                coin_options_found += 1

                # Determine option type from instrument ID
                # Format: COIN241101C00185000.CBOE (C=CALL, P=PUT)
                inst_id_str = str(inst.id)
                try:
                    # Extract the part after COIN and before strike
                    option_part = inst_id_str.split('COIN')[1][:7]
                    inst_type = 'CALL' if 'C' in option_part else 'PUT'
                except Exception as e:
                    self.log.warning(f"      ⚠️ Could not parse option type from {inst_id_str}: {e}", LogColor.YELLOW)
                    continue

                if inst_type != option_type:
                    continue
                
                matching_type_found += 1

                # Check expiration
                try:
                    exp_datetime = datetime.fromtimestamp(inst.expiration_ns / 1_000_000_000)
                    exp_date = exp_datetime.date()
                except Exception as e:
                    self.log.warning(f"      ⚠️ Could not parse expiration for {inst.id}: {e}", LogColor.YELLOW)
                    continue

                if exp_date < today_date:
                    continue  # Skip expired contracts

                days_to_expiry = (exp_date - today_date).days
                strike_price = float(inst.strike_price)
                strike_diff = abs(strike_price - target_strike)
                
                valid_contracts.append({
                    'instrument': inst,
                    'strike': strike_price,
                    'strike_diff': strike_diff,
                    'expiry_days': days_to_expiry,
                    'exp_date': exp_date
                })

                # Update best match (nearest expiry, then closest strike)
                if days_to_expiry < best_expiry_days or \
                   (days_to_expiry == best_expiry_days and strike_diff < best_strike_diff):
                    best_expiry_days = days_to_expiry
                    best_strike_diff = strike_diff
                    best_match = inst

            # Log search results
            self.log.info(f"      COIN options found: {coin_options_found}", LogColor.BLUE)
            self.log.info(f"      Matching type ({option_type}): {matching_type_found}", LogColor.BLUE)
            self.log.info(f"      Valid (not expired): {len(valid_contracts)}", LogColor.BLUE)

            if best_match:
                selected_strike = float(best_match.strike_price)
                exp_datetime = datetime.fromtimestamp(best_match.expiration_ns / 1_000_000_000)
                exp_date = exp_datetime.date()
                strike_distance = abs(selected_strike - underlying_price)
                
                self.log.info("=" * 60, LogColor.GREEN)
                self.log.info(f"✅ SELECTED OPTION CONTRACT", LogColor.GREEN)
                self.log.info(f"   Symbol: {best_match.id.symbol}", LogColor.GREEN)
                self.log.info(f"   Type: {option_type}", LogColor.GREEN)
                self.log.info(f"   Strike: ${selected_strike:.2f}", LogColor.GREEN)
                self.log.info(f"   Distance from underlying: ${strike_distance:.2f} ({strike_distance/underlying_price*100:.1f}%)", LogColor.GREEN)
                self.log.info(f"   Expiration: {exp_date} ({best_expiry_days} days)", LogColor.GREEN)
                self.log.info(f"   Distance from ATM: ${best_strike_diff:.2f}", LogColor.GREEN)
                self.log.info("=" * 60, LogColor.GREEN)
                
                # Log top 3 alternatives for transparency
                if len(valid_contracts) > 1:
                    self.log.info("   📋 Top alternatives:", LogColor.BLUE)
                    sorted_contracts = sorted(valid_contracts, key=lambda x: (x['expiry_days'], x['strike_diff']))[:3]
                    for i, contract in enumerate(sorted_contracts, 1):
                        self.log.info(
                            f"      {i}. Strike=${contract['strike']:.2f}, "
                            f"Exp={contract['exp_date']} ({contract['expiry_days']}d)",
                            LogColor.BLUE
                        )
            else:
                self.log.error(f"❌ No suitable {option_type} contract found", LogColor.RED)
                self.log.error(f"   Search criteria: Strike≈${target_strike:.2f}, Expiry>={today_date}", LogColor.RED)

            return best_match
            
        except Exception as e:
            self.log.error(f"❌ Error finding option contract: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            return None

    def _get_option_premium_relaxed(
        self, 
        option_instrument: OptionContract,
        underlying_price: float
    ) -> Optional[float]:
        """
        Get option premium with RELAXED validation (MUST TRADE requirement).
        
        Premium Sources (Priority Order):
        ----------------------------------
        1. Market quote (mid price if both bid/ask available)
        2. Ask price only (if bid missing)
        3. Bid price only (if ask missing)
        4. Estimated (3% of underlying as fallback)
        
        Parameters
        ----------
        option_instrument : OptionContract
            The option contract to get premium for.
        underlying_price : float
            Current underlying price (for fallback estimation).
        
        Returns
        -------
        Optional[float]
            Option premium per share, or None if completely unavailable.
        
        Notes
        -----
        - Logs spread percentage if quote available
        - Warns if spread exceeds threshold (but doesn't block trade)
        - Always returns a value (never None) to ensure trade execution
        """
        try:
            self.log.info(f"   💰 Fetching premium for {option_instrument.id.symbol}...", LogColor.BLUE)
            
            try:
                quote = self.cache.quote_tick(option_instrument.id)
            except Exception as e:
                self.log.warning(f"      ⚠️ Error accessing quote cache: {e}", LogColor.YELLOW)
                quote = None
            
            if quote is not None:
                bid = float(quote.bid_price) if quote.bid_price else 0.0
                ask = float(quote.ask_price) if quote.ask_price else 0.0
                
                self.log.info(f"      Quote found: Bid=${bid:.2f}, Ask=${ask:.2f}", LogColor.BLUE)
                
                # Use mid price if both available
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    spread = ask - bid
                    spread_pct = (spread / mid * 100) if mid > 0 else 0

                    if spread_pct > self.config.max_bid_ask_spread_pct:
                        self.log.warning(
                            f"      ⚠️ Wide spread: {spread_pct:.1f}% "
                            f"(threshold: {self.config.max_bid_ask_spread_pct}%) "
                            f"- USING ANYWAY (MUST TRADE)",
                            LogColor.YELLOW
                        )
                    else:
                        self.log.info(
                            f"      ✓ Spread acceptable: {spread_pct:.1f}%",
                            LogColor.GREEN
                        )

                    self.log.info(f"   💰 Premium: ${mid:.2f} (mid price)", LogColor.CYAN)
                    return mid
                    
                elif ask > 0:
                    # Only ask available
                    self.log.warning(f"      ⚠️ Only ask price available (no bid)", LogColor.YELLOW)
                    self.log.info(f"   💰 Premium: ${ask:.2f} (ask price)", LogColor.CYAN)
                    return ask
                    
                elif bid > 0:
                    # Only bid available
                    self.log.warning(f"      ⚠️ Only bid price available (no ask)", LogColor.YELLOW)
                    self.log.info(f"   💰 Premium: ${bid:.2f} (bid price)", LogColor.CYAN)
                    return bid
            
            # No quote available - use fallback estimation
            estimated = underlying_price * 0.03  # 3% of underlying
            self.log.warning(
                f"      ⚠️ No quote available - using ESTIMATED premium",
                LogColor.YELLOW
            )
            self.log.info(
                f"   💰 Premium: ${estimated:.2f} (3% of underlying ${underlying_price:.2f})",
                LogColor.YELLOW
            )
            return estimated
            
        except Exception as e:
            self.log.error(f"❌ Error getting premium: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
            
            # CRITICAL: Return estimate anyway for MUST TRADE rule
            estimated = underlying_price * 0.03
            self.log.warning(
                f"   💰 FALLBACK Premium: ${estimated:.2f} (error occurred)",
                LogColor.RED
            )
            return estimated

    def _manage_position(self, bar: Bar, bar_time_est: datetime) -> None:
        """
        Manage open position with TIME-BASED exits only.
        
        SL/TP orders handle profit target and stop loss exits.
        This method only checks for TIME-based exits:
        1. Force exit at 3:55 PM EST (before 4 PM 0DTE expiration)
        2. Max hold time exceeded (default 30 minutes)
        
        Parameters
        ----------
        bar : Bar
            Current bar data.
        bar_time_est : datetime
            Bar timestamp in EST timezone.
        
        Notes
        -----
        - Uses timezone-aware time comparison (handles EST/EDT)
        - Logs position status periodically
        - Prevents duplicate closes with is_closing flag
        """
        if self.current_position is None or self.is_closing:
            return

        try:
            # Check if position still open
            if not self.portfolio.is_net_long(self.current_position.instrument_id):
                self.log.info("   ○ Position closed externally (SL/TP triggered)", LogColor.CYAN)
                self.current_position = None
                self.current_option_instrument = None
                return

            # Calculate hold time
            time_held_minutes = 0.0
            if self.entry_timestamp:
                time_held_ns = bar.ts_event - self.entry_timestamp
                time_held_minutes = time_held_ns / (60 * 1_000_000_000)

            # Log position status every 5 minutes
            if int(time_held_minutes) % 5 == 0 and bar_time_est.second < 5:
                current_qty = float(self.current_position.quantity)
                current_avg_px = float(self.current_position.avg_px_open)
                self.log.info(
                    f"📊 Position Status: {current_qty} @ ${current_avg_px:.2f} | "
                    f"Held: {time_held_minutes:.1f} min",
                    LogColor.CYAN
                )

            # ═══════════════════════════════════════════════════════════
            # CHECK 1: Force exit at 3:55 PM EST (before 4 PM expiration)
            # ═══════════════════════════════════════════════════════════
            if bar_time_est.hour == 15 and bar_time_est.minute >= 55:
                self.log.warning("=" * 80, LogColor.RED)
                self.log.warning(
                    f"🕐 FORCE EXIT: 3:55 PM EST REACHED (0DTE expires at 4 PM!)",
                    LogColor.RED
                )
                self.log.warning(f"   Current time: {bar_time_est.strftime('%H:%M:%S EST')}", LogColor.RED)
                self.log.warning(f"   Hold time: {time_held_minutes:.1f} minutes", LogColor.RED)
                self.log.warning("=" * 80, LogColor.RED)
                self._close_position("Force exit - 3:55 PM before 0DTE expiration")
                return

            # ═══════════════════════════════════════════════════════════
            # CHECK 2: Max hold time exceeded
            # ═══════════════════════════════════════════════════════════
            if time_held_minutes >= self.config.max_hold_minutes:
                self.log.warning("=" * 80, LogColor.YELLOW)
                self.log.warning(
                    f"⏱️  MAX HOLD TIME REACHED: {time_held_minutes:.1f} min "
                    f"(limit: {self.config.max_hold_minutes} min)",
                    LogColor.YELLOW
                )
                self.log.warning("=" * 80, LogColor.YELLOW)
                self._close_position("Max hold time exceeded")
                return

        except Exception as e:
            self.log.error(f"❌ Error in _manage_position(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

    def _cancel_all_tracked_orders(self, reason: str) -> None:
        """
        Cancel all tracked SL/TP orders by their client order IDs.
        
        Manual implementation since OCO OrderList doesn't work in backtest.
        
        Parameters
        ----------
        reason : str
            Reason for cancellation (for logging).
        
        Notes
        -----
        - Cancels both SL and TP orders
        - Resets tracked order IDs
        - Logs each cancellation
        - Continues even if individual cancellations fail
        """
        try:
            cancelled = 0
            
            for order_id in [self.active_sl_order_id, self.active_tp_order_id]:
                if order_id is not None:
                    try:
                        self.cancel_order(order_id)
                        order_type = "SL" if order_id == self.active_sl_order_id else "TP"
                        self.log.info(
                            f"   🗑️  Cancelled {order_type} order: {order_id}",
                            LogColor.YELLOW
                        )
                        cancelled += 1
                    except Exception as e:
                        self.log.warning(f"   ⚠️ Failed to cancel order {order_id}: {e}", LogColor.YELLOW)
            
            self.active_sl_order_id = None
            self.active_tp_order_id = None
            
            if cancelled > 0:
                self.log.info(f"✅ Cancelled {cancelled} tracked orders - {reason}", LogColor.GREEN)
            
        except Exception as e:
            self.log.error(f"❌ Error cancelling tracked orders: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

    def _close_position(self, reason: str) -> None:
        """
        Close the current position.
        
        Execution Steps:
        ----------------
        1. Check if position exists and not already closing
        2. Cancel SL/TP orders first (prevent fills during close)
        3. Close all positions for the option instrument
        4. Reset state variables
        5. Log closure details
        
        Parameters
        ----------
        reason : str
            Reason for closing the position (for logging).
        
        Notes
        -----
        - Uses is_closing flag to prevent duplicate close attempts
        - Comprehensive error handling
        - Detailed logging of closure process
        """
        if self.current_position is None or self.is_closing:
            return
            
        self.is_closing = True

        try:
            self.log.info("=" * 80, LogColor.YELLOW)
            self.log.info(f"🔒 CLOSING POSITION: {reason}", LogColor.YELLOW)
            self.log.info("=" * 80, LogColor.YELLOW)
            
            # Cancel SL/TP orders first
            self._cancel_all_tracked_orders(f"Before closing - {reason}")
            
            # Close position
            if self.current_option_instrument:
                self.log.info(f"   Closing positions for {self.current_option_instrument.id.symbol}...", LogColor.YELLOW)
                self.close_all_positions(self.current_option_instrument.id)
                self.log.info("   ✓ Close order submitted", LogColor.GREEN)
        
        except Exception as e:
            self.log.error(f"❌ Error closing position: {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)
        
        finally:
            # Reset state
            self.current_position = None
            self.current_option_instrument = None
            self.entry_price = None
            self.entry_timestamp = None
            self.is_closing = False
            
            self.log.info("=" * 80, LogColor.YELLOW)

    def on_position_opened(self, position: Position) -> None:
        """
        Handle position opened event from NautilusTrader.
        
        Only tracks positions for the current option instrument to avoid
        tracking unrelated positions.
        
        Parameters
        ----------
        position : Position
            The position that was opened.
        """
        try:
            # Only track positions for our current option instrument
            if self.current_option_instrument and position.instrument_id == self.current_option_instrument.id:
                self.current_position = position
                
                self.log.info("=" * 80, LogColor.GREEN)
                self.log.info("📍 POSITION OPENED", LogColor.GREEN)
                self.log.info(f"   Instrument: {position.instrument_id}", LogColor.GREEN)
                self.log.info(f"   Quantity: {position.quantity} contracts", LogColor.GREEN)
                self.log.info(f"   Entry Price: ${position.avg_px_open}", LogColor.GREEN)
                self.log.info(f"   Position Value: ${float(position.avg_px_open) * float(position.quantity) * 100:.2f}", LogColor.GREEN)
                self.log.info("=" * 80, LogColor.GREEN)
            else:
                self.log.warning(
                    f"⚠️ Ignoring position opened for {position.instrument_id} (not current instrument)",
                    LogColor.YELLOW
                )
        except Exception as e:
            self.log.error(f"❌ Error in on_position_opened(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

    def on_position_closed(self, position: Position) -> None:
        """
        Handle position closed event from NautilusTrader.
        
        Logs P&L and cleans up remaining orders.
        
        Parameters
        ----------
        position : Position
            The position that was closed.
        """
        try:
            pnl = position.realized_pnl.as_decimal()
            pnl_float = float(pnl)
            
            # Update daily tracking
            self.daily_pnl += pnl_float
            
            # Determine color based on P&L
            color = LogColor.CYAN if pnl_float >= 0 else LogColor.RED
            icon = "💰" if pnl_float >= 0 else "💸"
            
            self.log.info("=" * 80, color)
            self.log.info(f"{icon} POSITION CLOSED", color)
            self.log.info(f"   Instrument: {position.instrument_id}", color)
            self.log.info(f"   Entry Price: ${position.avg_px_open}", color)
            self.log.info(f"   Exit Price: ${position.avg_px_close if position.avg_px_close else 'N/A'}", color)
            self.log.info(f"   Quantity: {position.quantity} contracts", color)
            self.log.info(f"   P&L: ${pnl_float:.2f} ({'+' if pnl_float >= 0 else ''}{pnl})", color)
            self.log.info(f"   Daily P&L: ${self.daily_pnl:.2f}", color)
            self.log.info("=" * 80, color)
            
            # Cancel any remaining orders
            self._cancel_all_tracked_orders("Position closed - cleaning up remaining orders")
            
            # Reset state
            self.current_position = None
            self.current_option_instrument = None
            self.is_closing = False
            
        except Exception as e:
            self.log.error(f"❌ Error in on_position_closed(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

    def on_order_filled(self, event) -> None:
        """
        Handle order filled event from NautilusTrader.
        
        Handles three types of fills:
        1. Entry order fill → Create SL/TP orders based on actual fill price
        2. Stop Loss fill → Cancel Take Profit order
        3. Take Profit fill → Cancel Stop Loss order
        
        Parameters
        ----------
        event : OrderFilled
            The order filled event containing fill details.
        
        Notes
        -----
        - SL/TP prices calculated from ACTUAL fill price (not estimate)
        - Manual OCO implementation (cancel opposite order on fill)
        - Comprehensive logging of order flow
        """
        try:
            fill_price = float(event.last_px)
            fill_qty = int(event.last_qty)
            
            # ═══════════════════════════════════════════════════════════
            # CASE 1: Stop Loss filled
            # ═══════════════════════════════════════════════════════════
            if event.client_order_id == self.active_sl_order_id:
                self.log.info("=" * 80, LogColor.RED)
                self.log.info("🛑 STOP LOSS TRIGGERED", LogColor.RED)
                self.log.info(f"   Exit Price: ${fill_price:.2f}", LogColor.RED)
                self.log.info(f"   Quantity: {fill_qty} contracts", LogColor.RED)
                if self.entry_price:
                    loss_pct = ((fill_price - self.entry_price) / self.entry_price) * 100
                    self.log.info(f"   Loss: {loss_pct:.2f}%", LogColor.RED)
                self.log.info("=" * 80, LogColor.RED)
                
                self.active_sl_order_id = None
                
                # Cancel Take Profit order
                if self.active_tp_order_id is not None:
                    try:
                        self.cancel_order(self.active_tp_order_id)
                        self.log.info(f"   🗑️  Cancelled TP order (SL filled)", LogColor.YELLOW)
                        self.active_tp_order_id = None
                    except Exception as e:
                        self.log.warning(f"   ⚠️ Failed to cancel TP: {e}", LogColor.YELLOW)
                return

            # ═══════════════════════════════════════════════════════════
            # CASE 2: Take Profit filled
            # ═══════════════════════════════════════════════════════════
            if event.client_order_id == self.active_tp_order_id:
                self.log.info("=" * 80, LogColor.GREEN)
                self.log.info("🎯 TAKE PROFIT HIT!", LogColor.GREEN)
                self.log.info(f"   Exit Price: ${fill_price:.2f}", LogColor.GREEN)
                self.log.info(f"   Quantity: {fill_qty} contracts", LogColor.GREEN)
                if self.entry_price:
                    profit_pct = ((fill_price - self.entry_price) / self.entry_price) * 100
                    self.log.info(f"   Profit: +{profit_pct:.2f}%", LogColor.GREEN)
                self.log.info("=" * 80, LogColor.GREEN)
                
                self.active_tp_order_id = None
                
                # Cancel Stop Loss order
                if self.active_sl_order_id is not None:
                    try:
                        self.cancel_order(self.active_sl_order_id)
                        self.log.info(f"   🗑️  Cancelled SL order (TP filled)", LogColor.YELLOW)
                        self.active_sl_order_id = None
                    except Exception as e:
                        self.log.warning(f"   ⚠️ Failed to cancel SL: {e}", LogColor.YELLOW)
                return

            # ═══════════════════════════════════════════════════════════
            # CASE 3: Entry order filled
            # ═══════════════════════════════════════════════════════════
            if event.client_order_id != self.entry_order_id:
                return  # Ignore other orders

            if event.order_side != OrderSide.BUY or self.pending_option_instrument is None:
                return

            self.entry_price = fill_price

            self.log.info("=" * 80, LogColor.GREEN)
            self.log.info("📍 ENTRY FILLED", LogColor.GREEN)
            self.log.info(f"   Price: ${fill_price:.2f} per share", LogColor.GREEN)
            self.log.info(f"   Quantity: {fill_qty} contracts", LogColor.GREEN)
            self.log.info(f"   Total Cost: ${fill_price * fill_qty * 100:.2f}", LogColor.GREEN)
            self.log.info("=" * 80, LogColor.GREEN)

            # Calculate SL and TP prices based on ACTUAL fill price
            tp_price = fill_price * (1 + float(self.config.target_profit_pct) / 100)
            sl_price = fill_price * (1 - float(self.config.stop_loss_pct) / 100)

            self.log.info("🎯 Creating SL/TP orders based on actual fill price...", LogColor.BLUE)
            self.log.info(f"   Entry: ${fill_price:.2f}", LogColor.BLUE)
            self.log.info(f"   Take Profit: ${tp_price:.2f} (+{self.config.target_profit_pct}%)", LogColor.BLUE)
            self.log.info(f"   Stop Loss: ${sl_price:.2f} (-{self.config.stop_loss_pct}%)", LogColor.BLUE)

            # Cancel any existing orders first
            self._cancel_all_tracked_orders("Before submitting new SL/TP")

            # Create Stop Loss order
            sl_order = self.order_factory.stop_market(
                instrument_id=self.pending_option_instrument.id,
                order_side=OrderSide.SELL,
                quantity=event.last_qty,
                trigger_price=self.pending_option_instrument.make_price(sl_price),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )

            # Create Take Profit order
            tp_order = self.order_factory.limit(
                instrument_id=self.pending_option_instrument.id,
                order_side=OrderSide.SELL,
                quantity=event.last_qty,
                price=self.pending_option_instrument.make_price(tp_price),
                time_in_force=TimeInForce.GTC,
                post_only=False,
                reduce_only=True,
            )

            # Track order IDs for manual cancellation
            self.active_sl_order_id = sl_order.client_order_id
            self.active_tp_order_id = tp_order.client_order_id

            # Submit orders
            self.submit_order(sl_order)
            self.submit_order(tp_order)

            self.log.info("✅ SL/TP orders submitted", LogColor.GREEN)
            self.log.info(f"   SL Order ID: {sl_order.client_order_id}", LogColor.GREEN)
            self.log.info(f"   TP Order ID: {tp_order.client_order_id}", LogColor.GREEN)

            # Clear pending instrument
            self.pending_option_instrument = None
            self.entry_order_id = None

        except Exception as e:
            self.log.error(f"❌ Error in on_order_filled(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

    def on_stop(self) -> None:
        """
        Actions performed when strategy is stopped.
        
        Ensures clean shutdown by closing any open positions.
        """
        try:
            self.log.info("=" * 80, LogColor.YELLOW)
            self.log.info("🛑 STOPPING STRATEGY", LogColor.YELLOW)
            self.log.info("=" * 80, LogColor.YELLOW)
            
            # Close any open positions
            if self.current_position and not self.is_closing:
                self.log.info("   Closing open position...", LogColor.YELLOW)
                self._close_position("Strategy stopped")

            self.log.info("=" * 80, LogColor.YELLOW)
            self.log.info("🏁 STRATEGY STOPPED", LogColor.YELLOW)
            self.log.info("=" * 80, LogColor.YELLOW)
            
        except Exception as e:
            self.log.error(f"❌ Error in on_stop(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)

    def on_reset(self) -> None:
        """
        Reset the strategy state.
        
        Called when strategy needs to be reset (e.g., between backtest runs).
        """
        try:
            self.log.info("🔄 Resetting strategy state...", LogColor.CYAN)
            
            self.traded_today = False
            self.last_trade_date = None
            self.entry_price = None
            self.entry_timestamp = None
            self.current_position = None
            self.current_option_instrument = None
            self.pending_option_instrument = None
            self.volume_history = []
            self.is_closing = False
            self.entry_order_id = None
            self.active_sl_order_id = None
            self.active_tp_order_id = None
            self.trades_today = 0
            self.daily_pnl = 0.0
            
            self.log.info("✅ Strategy reset complete", LogColor.GREEN)
            
        except Exception as e:
            self.log.error(f"❌ Error in on_reset(): {e}", LogColor.RED)
            self.log.error(traceback.format_exc(), LogColor.RED)