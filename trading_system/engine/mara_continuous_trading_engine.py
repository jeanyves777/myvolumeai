#!/usr/bin/env python3
"""
THE VOLUME AI - MARA Continuous Trading Engine

Engine for continuous MARA options trading with:
- ATM contract selection at each entry
- Weekly expiry options
- Volume-based entry validation
- Automatic re-entry after position closes
- Full trade logging with Greeks
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import pytz

from trading_system.strategies.mara_continuous_momentum import (
    MARAContinuousMomentumConfig,
    MARAContinuousMomentumStrategy,
    TechnicalIndicators,
    PriceActionSignal,
    VolumeAnalysis,
    ATMContractInfo
)
from trading_system.engine.alpaca_client import AlpacaClient, Quote
from trading_system.analytics.options_trade_logger import OptionsTradeLogger

EST = pytz.timezone('US/Eastern')


@dataclass
class Position:
    """Active position information."""
    symbol: str = ""
    option_symbol: str = ""
    option_type: str = ""  # 'call' or 'put'
    strike: float = 0.0
    expiration: datetime = None
    entry_price: float = 0.0
    entry_time: datetime = None
    qty: int = 0
    entry_order_id: str = ""
    tp_order_id: str = ""
    entry_underlying_price: float = 0.0
    greeks: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradingSession:
    """Session state for continuous trading."""
    is_running: bool = False
    position: Optional[Position] = None
    trades_today: int = 0
    last_exit_time: Optional[datetime] = None
    daily_pnl: float = 0.0
    winners: int = 0
    losers: int = 0


class MARAContinuousTradingEngine:
    """
    Continuous trading engine for MARA options.

    Flow:
    1. Monitor MARA price and volume
    2. When volume spike detected + momentum confirmed -> select ATM contract
    3. Enter position with TP limit order
    4. Monitor for TP/SL/max hold
    5. After exit, wait cooldown then repeat
    """

    def __init__(
        self,
        config: MARAContinuousMomentumConfig,
        api_key: str,
        api_secret: str
    ):
        self.config = config
        self.strategy = MARAContinuousMomentumStrategy(config)
        self.client = AlpacaClient(api_key, api_secret, paper=True)
        self.trade_logger = OptionsTradeLogger()

        self.session = TradingSession()
        self.latest_underlying_quote: Optional[Quote] = None
        self.latest_option_quote: Optional[Quote] = None

        # Technical data storage
        self.bars_1min: List[Dict] = []
        self.bars_5min: List[Dict] = []

    def _log(self, message: str, level: str = "INFO"):
        """Print timestamped log message."""
        timestamp = datetime.now(EST).strftime("%H:%M:%S")
        prefix = {
            "INFO": "",
            "TRADE": "[TRADE]",
            "WARN": "[WARN]",
            "ERROR": "[ERROR]",
            "SIGNAL": "[SIGNAL]"
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EST)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def _fetch_bars(self) -> bool:
        """Fetch 1-min and 5-min bars for technical analysis."""
        try:
            # Get 1-minute bars (last 30 bars)
            self.bars_1min = self.client.get_stock_bars(
                self.config.underlying_symbol,
                timeframe="1Min",
                limit=30
            )

            # Get 5-minute bars (last 10 bars)
            self.bars_5min = self.client.get_stock_bars(
                self.config.underlying_symbol,
                timeframe="5Min",
                limit=10
            )

            return len(self.bars_1min) >= 20 and len(self.bars_5min) >= 5

        except Exception as e:
            self._log(f"Error fetching bars: {e}", "ERROR")
            return False

    def _calculate_indicators(self) -> TechnicalIndicators:
        """Calculate technical indicators from 1-min bars."""
        indicators = TechnicalIndicators()

        if len(self.bars_1min) < 20:
            return indicators

        closes = [b.close for b in self.bars_1min]
        volumes = [b.volume for b in self.bars_1min]
        highs = [b.high for b in self.bars_1min]
        lows = [b.low for b in self.bars_1min]

        # EMA calculations
        indicators.ema_9 = self._ema(closes, 9)
        indicators.ema_20 = self._ema(closes, 20)

        # VWAP (simplified - cumulative)
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        cum_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
        cum_vol = sum(volumes)
        indicators.vwap = cum_tp_vol / cum_vol if cum_vol > 0 else closes[-1]

        # RSI
        indicators.rsi = self._rsi(closes, 14)

        # MACD
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        indicators.macd = ema_12 - ema_26
        # Simplified signal line
        indicators.macd_signal = indicators.macd * 0.9

        # Bollinger Bands
        sma_20 = sum(closes[-20:]) / 20
        std_20 = (sum((c - sma_20) ** 2 for c in closes[-20:]) / 20) ** 0.5
        indicators.bb_mid = sma_20
        indicators.bb_upper = sma_20 + 2 * std_20
        indicators.bb_lower = sma_20 - 2 * std_20

        # Volume
        indicators.volume = volumes[-1]
        indicators.avg_volume = sum(volumes) / len(volumes)

        # ATR
        indicators.atr = self._atr(highs, lows, closes, 14)

        return indicators

    def _ema(self, data: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1] if data else 0

        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period

        for price in data[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        """Calculate ATR."""
        if len(highs) < period + 1:
            return highs[-1] - lows[-1] if highs else 0

        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        return sum(true_ranges[-period:]) / period

    def _find_atm_contract(self, direction: str) -> Optional[ATMContractInfo]:
        """
        Find ATM contract for the given direction.
        Uses weekly expiry and validates liquidity.
        """
        if not self.latest_underlying_quote:
            return None

        current_price = self.latest_underlying_quote.mid
        option_type = "call" if direction == "BULLISH" else "put"

        # Get weekly expiry
        now = datetime.now(EST)
        expiry_date = self.strategy.get_weekly_expiry(now)

        # Calculate days to expiry
        days_to_expiry = (expiry_date.date() - now.date()).days

        # If weekly is 0DTE and we want min 1 DTE, get next week
        if days_to_expiry < self.config.min_days_to_expiry:
            expiry_date = expiry_date + timedelta(days=7)
            days_to_expiry = (expiry_date.date() - now.date()).days

        self._log(f"Looking for {option_type.upper()} ATM, expiry {expiry_date.strftime('%Y-%m-%d')} ({days_to_expiry} DTE)")

        try:
            # Get option chain from Alpaca
            chain = self.client.get_option_chain(
                self.config.underlying_symbol,
                expiry_date.strftime("%Y-%m-%d")
            )

            if not chain:
                self._log("No option chain available", "WARN")
                return None

            # Filter for the option type we want
            options = [o for o in chain if o.get('option_type', '').lower() == option_type]

            if not options:
                self._log(f"No {option_type} options found", "WARN")
                return None

            # Find ATM (closest strike to current price)
            atm_option = min(options, key=lambda o: abs(o['strike'] - current_price))

            # Build contract info
            contract = ATMContractInfo(
                symbol=atm_option.get('symbol', ''),
                strike=atm_option['strike'],
                expiration=expiry_date,
                days_to_expiry=days_to_expiry,
                option_type=option_type,
                bid=atm_option.get('bid', 0),
                ask=atm_option.get('ask', 0),
                mid=(atm_option.get('bid', 0) + atm_option.get('ask', 0)) / 2,
                volume=atm_option.get('volume', 0),
                open_interest=atm_option.get('open_interest', 0),
                delta=atm_option.get('delta', 0.5 if option_type == 'call' else -0.5)
            )

            # Validate contract meets requirements
            contract = self.strategy.validate_contract(contract, current_price)

            if contract.is_valid:
                self._log(f"Found ATM: {contract.symbol} Strike=${contract.strike} "
                         f"Bid=${contract.bid:.2f} Ask=${contract.ask:.2f} "
                         f"Spread={contract.spread_pct:.1f}%")
            else:
                self._log(f"ATM contract rejected: {contract.rejection_reason}", "WARN")

            return contract

        except Exception as e:
            self._log(f"Error finding ATM contract: {e}", "ERROR")
            return None

    def _find_atm_strike_manual(self, direction: str) -> Optional[ATMContractInfo]:
        """
        Manually construct ATM option symbol when chain API not available.
        """
        if not self.latest_underlying_quote:
            return None

        current_price = self.latest_underlying_quote.mid
        option_type = "call" if direction == "BULLISH" else "put"

        # Round to nearest $0.50 or $1 strike (MARA has $0.50 strikes)
        # For MARA around $12, strikes are typically $0.50 apart
        atm_strike = round(current_price * 2) / 2

        # Get weekly expiry
        now = datetime.now(EST)
        expiry_date = self.strategy.get_weekly_expiry(now)
        days_to_expiry = (expiry_date.date() - now.date()).days

        # If 0DTE, get next week
        if days_to_expiry < self.config.min_days_to_expiry:
            expiry_date = expiry_date + timedelta(days=7)
            days_to_expiry = (expiry_date.date() - now.date()).days

        # Build OCC symbol: MARA251212C00012500 (for $12.50 call expiring 12/12/25)
        expiry_str = expiry_date.strftime("%y%m%d")
        opt_type_char = "C" if option_type == "call" else "P"
        strike_int = int(atm_strike * 1000)
        occ_symbol = f"MARA{expiry_str}{opt_type_char}{strike_int:08d}"

        self._log(f"Constructed ATM symbol: {occ_symbol} (Strike ${atm_strike})")

        # Get quote for this option
        try:
            option_quote = self.client.get_option_quote(occ_symbol)

            if option_quote:
                contract = ATMContractInfo(
                    symbol=occ_symbol,
                    strike=atm_strike,
                    expiration=expiry_date,
                    days_to_expiry=days_to_expiry,
                    option_type=option_type,
                    bid=option_quote.bid,
                    ask=option_quote.ask,
                    mid=option_quote.mid,
                    volume=0,  # Not available from quote
                    open_interest=0,
                    delta=0.5 if option_type == "call" else -0.5
                )

                # Calculate spread
                if contract.mid > 0:
                    contract.spread_pct = (contract.ask - contract.bid) / contract.mid * 100

                # Simple validation
                if contract.mid > 0 and contract.spread_pct < self.config.max_bid_ask_spread_pct:
                    contract.is_valid = True
                    self._log(f"ATM valid: ${contract.strike} Mid=${contract.mid:.2f} "
                             f"Spread={contract.spread_pct:.1f}%")
                else:
                    contract.is_valid = False
                    contract.rejection_reason = f"Spread too wide ({contract.spread_pct:.1f}%)"

                return contract
            else:
                self._log(f"Could not get quote for {occ_symbol}", "WARN")
                return None

        except Exception as e:
            self._log(f"Error getting ATM quote: {e}", "ERROR")
            return None

    def _enter_position(self, contract: ATMContractInfo, direction: str) -> bool:
        """Enter a position with the selected contract."""
        if not contract.is_valid:
            return False

        # Calculate position size
        qty = self.strategy.calculate_position_size(contract.mid)
        if qty <= 0:
            self._log("Position size too small", "WARN")
            return False

        self._log("=" * 70, "TRADE")
        self._log(f"ENTERING {direction} POSITION", "TRADE")
        self._log(f"  Contract: {contract.symbol}", "TRADE")
        self._log(f"  Strike: ${contract.strike}", "TRADE")
        self._log(f"  Expiry: {contract.expiration.strftime('%Y-%m-%d')} ({contract.days_to_expiry} DTE)", "TRADE")
        self._log(f"  Type: {contract.option_type.upper()}", "TRADE")
        self._log(f"  Mid Price: ${contract.mid:.2f}", "TRADE")
        self._log(f"  Quantity: {qty} contracts", "TRADE")
        self._log(f"  Est. Value: ${contract.mid * qty * 100:.2f}", "TRADE")

        # Submit market buy order
        try:
            order_result = self.client.submit_option_order(
                symbol=contract.symbol,
                qty=qty,
                side="buy",
                order_type="market"
            )

            if not order_result or not order_result.get('success'):
                self._log(f"Order failed: {order_result}", "ERROR")
                return False

            order_id = order_result.get('order_id', '')
            actual_fill_price = order_result.get('filled_avg_price', contract.mid)

            self._log(f"  Order ID: {order_id}", "TRADE")
            self._log(f"  Fill Price: ${actual_fill_price:.2f}", "TRADE")

            # Calculate and place TP limit order
            tp_price = round(actual_fill_price * (1 + self.config.target_profit_pct / 100), 2)
            self._log(f"  TP Target: ${tp_price:.2f} (+{self.config.target_profit_pct}%)", "TRADE")

            tp_result = self.client.submit_option_order(
                symbol=contract.symbol,
                qty=qty,
                side="sell",
                order_type="limit",
                limit_price=tp_price
            )

            tp_order_id = tp_result.get('order_id', '') if tp_result else ''

            # Create position object
            self.session.position = Position(
                symbol=self.config.underlying_symbol,
                option_symbol=contract.symbol,
                option_type=contract.option_type,
                strike=contract.strike,
                expiration=contract.expiration,
                entry_price=actual_fill_price,
                entry_time=datetime.now(EST),
                qty=qty,
                entry_order_id=order_id,
                tp_order_id=tp_order_id,
                entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0
            )

            self.session.trades_today += 1

            # Fetch and log Greeks
            try:
                greeks = self.client.get_option_greeks(contract.symbol)
                if greeks:
                    self.session.position.greeks = {
                        'delta': greeks.delta,
                        'gamma': greeks.gamma,
                        'theta': greeks.theta,
                        'vega': greeks.vega,
                        'iv': greeks.implied_volatility
                    }
                    self._log(f"  Greeks: Δ={greeks.delta:.3f} Γ={greeks.gamma:.4f} "
                             f"Θ={greeks.theta:.3f} V={greeks.vega:.3f} IV={greeks.implied_volatility:.1%}", "TRADE")
            except Exception as ge:
                self._log(f"  Could not fetch Greeks: {ge}", "WARN")

            # Log to trade logger
            try:
                trade_id = f"CONT_MARA_{contract.symbol}_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"
                self.trade_logger.log_entry(
                    trade_id=trade_id,
                    underlying_symbol=self.config.underlying_symbol,
                    option_symbol=contract.symbol,
                    option_type=contract.option_type,
                    strike_price=contract.strike,
                    expiration_date=contract.expiration.strftime("%Y-%m-%d"),
                    entry_time=datetime.now(EST),
                    entry_price=actual_fill_price,
                    entry_qty=qty,
                    entry_order_id=order_id,
                    entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0,
                    greeks=self.session.position.greeks,
                    target_profit_pct=self.config.target_profit_pct,
                    stop_loss_pct=self.config.stop_loss_pct,
                    notes=f"MARA CONTINUOUS | {direction} | {contract.days_to_expiry} DTE"
                )
                self.session.position.entry_order_id = trade_id
            except Exception as le:
                self._log(f"  Trade log error: {le}", "WARN")

            self._log("=" * 70, "TRADE")
            return True

        except Exception as e:
            self._log(f"Error entering position: {e}", "ERROR")
            return False

    def _check_position(self) -> Optional[str]:
        """
        Check position for exit conditions.
        Returns exit reason or None if still holding.
        """
        if not self.session.position:
            return None

        pos = self.session.position
        now = datetime.now(EST)

        # Check if TP limit order was filled
        if pos.tp_order_id:
            try:
                order_status = self.client.get_order_status(pos.tp_order_id)
                if order_status and order_status.get('status') == 'filled':
                    return "TAKE_PROFIT"
            except:
                pass

        # Get current option quote
        try:
            option_quote = self.client.get_option_quote(pos.option_symbol)
            if option_quote:
                self.latest_option_quote = option_quote
                current_price = option_quote.mid

                # Check stop loss
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                if pnl_pct <= -self.config.stop_loss_pct:
                    return "STOP_LOSS"

                # Log current status periodically
                self._log(f"Position: {pos.option_symbol} Entry=${pos.entry_price:.2f} "
                         f"Current=${current_price:.2f} P&L={pnl_pct:+.1f}%")

        except Exception as e:
            self._log(f"Error checking position: {e}", "WARN")

        # Check max hold time
        hold_minutes = (now - pos.entry_time).total_seconds() / 60
        if hold_minutes >= self.config.max_hold_minutes:
            return "MAX_HOLD_TIME"

        # Check force exit time
        force_exit = datetime.strptime(self.config.force_exit_time, "%H:%M:%S").time()
        if now.time() >= force_exit:
            return "FORCE_EXIT"

        return None

    def _exit_position(self, reason: str) -> bool:
        """Exit the current position."""
        if not self.session.position:
            return False

        pos = self.session.position
        now = datetime.now(EST)

        self._log("=" * 70, "TRADE")
        self._log(f"EXITING POSITION - {reason}", "TRADE")

        # Cancel TP order if exists and not filled
        if pos.tp_order_id and reason != "TAKE_PROFIT":
            try:
                self.client.cancel_order(pos.tp_order_id)
                self._log(f"  Cancelled TP order: {pos.tp_order_id}", "TRADE")
            except:
                pass

        # Get exit price
        exit_price = pos.entry_price  # Default
        exit_order_id = ""

        if reason == "TAKE_PROFIT":
            # TP was filled, get fill price
            try:
                order_status = self.client.get_order_status(pos.tp_order_id)
                if order_status:
                    exit_price = order_status.get('filled_avg_price', pos.entry_price * 1.075)
                    exit_order_id = pos.tp_order_id
            except:
                exit_price = pos.entry_price * (1 + self.config.target_profit_pct / 100)
        else:
            # Market sell
            try:
                sell_result = self.client.submit_option_order(
                    symbol=pos.option_symbol,
                    qty=pos.qty,
                    side="sell",
                    order_type="market"
                )
                if sell_result and sell_result.get('success'):
                    exit_price = sell_result.get('filled_avg_price', pos.entry_price)
                    exit_order_id = sell_result.get('order_id', '')
            except Exception as e:
                self._log(f"  Error selling: {e}", "ERROR")
                if self.latest_option_quote:
                    exit_price = self.latest_option_quote.mid

        # Calculate P&L
        gross_pnl = (exit_price - pos.entry_price) * pos.qty * 100
        fees = pos.qty * 1.30  # Estimated fees
        net_pnl = gross_pnl - fees
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        hold_minutes = (now - pos.entry_time).total_seconds() / 60

        self._log(f"  Exit Price: ${exit_price:.2f}", "TRADE")
        self._log(f"  Gross P&L: ${gross_pnl:+.2f}", "TRADE")
        self._log(f"  Net P&L: ${net_pnl:+.2f} ({pnl_pct:+.1f}%)", "TRADE")
        self._log(f"  Hold Time: {hold_minutes:.1f} minutes", "TRADE")

        # Update session stats
        self.session.daily_pnl += net_pnl
        if net_pnl > 0:
            self.session.winners += 1
        else:
            self.session.losers += 1

        # Log trade exit
        try:
            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=now,
                exit_price=exit_price,
                exit_qty=pos.qty,
                exit_order_id=exit_order_id,
                exit_reason=reason,
                exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0,
                fees_paid=fees,
                notes=f"MARA CONTINUOUS | EXIT: {reason} - Hold time: {hold_minutes:.1f}m"
            )
        except Exception as le:
            self._log(f"  Trade log error: {le}", "WARN")

        self._log("=" * 70, "TRADE")

        # Record exit for cooldown
        self.strategy.record_trade_exit(now)
        self.session.last_exit_time = now
        self.session.position = None

        return True

    def _print_status(self):
        """Print current status summary."""
        now = datetime.now(EST)
        in_cooldown, cooldown_mins = self.strategy.is_in_cooldown(now)

        status = "IN POSITION" if self.session.position else ("COOLDOWN" if in_cooldown else "SCANNING")

        print()
        self._log(f"Status: {status} | Trades: {self.session.trades_today} | "
                 f"P&L: ${self.session.daily_pnl:+.2f} | W/L: {self.session.winners}/{self.session.losers}")

        if in_cooldown:
            self._log(f"  Cooldown: {cooldown_mins}m remaining")

        if self.latest_underlying_quote:
            self._log(f"  MARA: ${self.latest_underlying_quote.mid:.2f}")
        print()

    def run(self):
        """Main trading loop."""
        self._log("Starting MARA Continuous Trading Engine")
        self._log(f"  Position Size: ${self.config.fixed_position_value}")
        self._log(f"  TP: {self.config.target_profit_pct}% | SL: {self.config.stop_loss_pct}%")
        self._log(f"  Cooldown: {self.config.cooldown_minutes}m between trades")
        self._log(f"  Max Trades/Day: {self.config.max_trades_per_day}")
        self._log(f"  Expiry: Weekly (min {self.config.min_days_to_expiry} DTE)")

        self.session.is_running = True
        poll_count = 0

        while self.session.is_running:
            try:
                now = datetime.now(EST)

                # Check market hours
                if not self._is_market_open():
                    self._log("Market closed. Waiting...")
                    time.sleep(60)
                    continue

                # Get underlying quote
                self.latest_underlying_quote = self.client.get_latest_stock_quote(
                    self.config.underlying_symbol
                )

                if not self.latest_underlying_quote:
                    self._log("Could not get MARA quote", "WARN")
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                # If we have a position, check it
                if self.session.position:
                    exit_reason = self._check_position()
                    if exit_reason:
                        self._exit_position(exit_reason)
                else:
                    # No position - look for entry
                    can_trade, reason = self.strategy.can_trade(now)

                    if can_trade:
                        # Fetch technical data
                        if self._fetch_bars():
                            indicators = self._calculate_indicators()

                            # Analyze price action (verbose=True to show PA breakdown)
                            price_action = self.strategy.analyze_price_action(self.bars_5min, verbose=True)

                            # Analyze volume
                            volume_analysis = self.strategy.analyze_volume(
                                indicators.volume,
                                indicators.avg_volume
                            )

                            # Get entry signal
                            should_enter, direction, opt_type, reasons = self.strategy.get_entry_signal(
                                self.latest_underlying_quote.mid,
                                indicators,
                                price_action,
                                volume_analysis,
                                now
                            )

                            # Always log signal analysis for debugging
                            tech_score, tech_dir, _ = self.strategy.calculate_technical_score(
                                self.latest_underlying_quote.mid, indicators
                            )
                            self._log(f"MARA ${self.latest_underlying_quote.mid:.2f} | "
                                     f"Tech: {tech_dir}({tech_score}) | "
                                     f"PA: {price_action.direction}({price_action.score}) | "
                                     f"Vol: {volume_analysis.trend}({volume_analysis.volume_ratio:.1f}x)")

                            if should_enter:
                                self._log(f"ENTRY SIGNAL: {direction}", "SIGNAL")
                                self._log(f"  Reasons: {', '.join(reasons)}", "SIGNAL")

                                # Find ATM contract
                                contract = self._find_atm_strike_manual(direction)

                                if contract and contract.is_valid:
                                    self._enter_position(contract, direction)
                                else:
                                    self._log("Could not find valid ATM contract", "WARN")
                            else:
                                if direction in ["CONFLICT", "NEUTRAL", "NONE"]:
                                    self._log(f"  No entry: {direction} - {', '.join(reasons)}")
                        else:
                            self._log("Could not fetch bars", "WARN")
                    else:
                        if poll_count % 6 == 0:  # Every minute
                            self._log(f"Cannot trade: {reason}")

                # Print status periodically
                poll_count += 1
                if poll_count % 12 == 0:  # Every 2 minutes
                    self._print_status()

                time.sleep(self.config.poll_interval_seconds)

            except KeyboardInterrupt:
                self._log("Shutdown requested")
                break
            except Exception as e:
                self._log(f"Error in main loop: {e}", "ERROR")
                time.sleep(self.config.poll_interval_seconds)

        # Cleanup
        if self.session.position:
            self._log("Exiting remaining position on shutdown")
            self._exit_position("SHUTDOWN")

        self._log("Engine stopped")
        self._print_status()

    def stop(self):
        """Stop the trading engine."""
        self.session.is_running = False
