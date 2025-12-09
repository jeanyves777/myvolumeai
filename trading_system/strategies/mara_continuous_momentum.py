#!/usr/bin/env python3
"""
THE VOLUME AI - MARA Continuous Momentum Strategy

A continuous trading strategy that:
- Picks ATM contracts at time of each entry
- Uses weekly expiry options (not 0DTE)
- Re-enters after positions close when conditions align
- Requires volume confirmation for entries
- Validates momentum with dual-signal approach

Key Features:
- Unlimited trades per day (with cooldown between trades)
- ATM strike selection based on current underlying price
- Weekly Friday expiry for better time value
- Volume spike detection for entry timing
- Bid/ask spread validation for liquidity
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional, Dict, List, Tuple, Any
import pytz

EST = pytz.timezone('US/Eastern')


@dataclass
class MARAContinuousMomentumConfig:
    """
    Configuration for MARA Continuous Momentum Strategy.
    """
    underlying_symbol: str = "MARA"

    # Position sizing
    fixed_position_value: float = 200.0  # $200 per trade

    # Profit targets
    target_profit_pct: float = 7.5   # Take profit at 7.5%
    stop_loss_pct: float = 25.0      # Stop loss at 25%

    # Trading window
    entry_time_start: str = "09:35:00"  # 5 min after open for stability
    entry_time_end: str = "15:30:00"    # Stop new entries 30 min before close
    force_exit_time: str = "15:50:00"   # Force exit before close

    # Continuous trading settings
    max_hold_minutes: int = 30          # Max hold time per trade
    cooldown_minutes: int = 5           # Wait time after closing a position
    max_trades_per_day: int = 10        # Safety limit

    # Volume requirements
    volume_spike_multiplier: float = 1.5  # Volume must be 1.5x average
    min_volume_threshold: int = 10000     # Minimum absolute volume

    # ATM contract selection
    use_weekly_expiry: bool = True       # Use weekly (Friday) expiry
    max_days_to_expiry: int = 7          # Max DTE for contract selection
    min_days_to_expiry: int = 1          # Min DTE (avoid 0DTE if desired)

    # Liquidity requirements
    max_bid_ask_spread_pct: float = 15.0  # Max spread as % of mid price
    min_option_volume: int = 10           # Minimum option volume
    min_open_interest: int = 50           # Minimum open interest

    # Signal thresholds
    min_signal_score: int = 3            # Minimum technical score (out of 6)
    require_dual_confirmation: bool = True  # Both methods must agree

    # Polling
    poll_interval_seconds: int = 10


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values."""
    ema_9: float = 0.0
    ema_20: float = 0.0
    vwap: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    volume: float = 0.0
    avg_volume: float = 0.0
    atr: float = 0.0


@dataclass
class PriceActionSignal:
    """5-minute price action analysis."""
    direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    score: int = 0
    candle_trend: str = ""
    higher_highs: bool = False
    lower_lows: bool = False
    momentum: float = 0.0
    last_bar_strength: float = 0.0


@dataclass
class VolumeAnalysis:
    """Volume analysis for entry validation."""
    current_volume: float = 0.0
    avg_volume: float = 0.0
    volume_ratio: float = 0.0
    is_spike: bool = False
    trend: str = "NORMAL"  # SPIKE, HIGH, NORMAL, LOW


@dataclass
class ATMContractInfo:
    """Information about selected ATM contract."""
    symbol: str = ""
    strike: float = 0.0
    expiration: datetime = None
    days_to_expiry: int = 0
    option_type: str = ""  # 'call' or 'put'
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread_pct: float = 0.0
    volume: int = 0
    open_interest: int = 0
    delta: float = 0.0
    is_valid: bool = False
    rejection_reason: str = ""


class MARAContinuousMomentumStrategy:
    """
    MARA Continuous Momentum Strategy

    This strategy continuously monitors MARA for trading opportunities:
    1. Detects volume spikes as entry triggers
    2. Validates momentum with dual-signal approach
    3. Selects ATM contract at time of entry
    4. Uses weekly expiry for better time value
    5. Re-enters after positions close (with cooldown)
    """

    def __init__(self, config: MARAContinuousMomentumConfig):
        self.config = config
        self.last_trade_exit_time: Optional[datetime] = None
        self.trades_today: int = 0

    def is_in_cooldown(self, current_time: datetime) -> Tuple[bool, int]:
        """
        Check if we're in cooldown period after last trade.
        Returns (is_in_cooldown, minutes_remaining)
        """
        if self.last_trade_exit_time is None:
            return False, 0

        elapsed = (current_time - self.last_trade_exit_time).total_seconds() / 60
        remaining = self.config.cooldown_minutes - elapsed

        if remaining > 0:
            return True, int(remaining)
        return False, 0

    def record_trade_exit(self, exit_time: datetime):
        """Record when a trade was exited for cooldown tracking."""
        self.last_trade_exit_time = exit_time
        self.trades_today += 1

    def reset_daily_state(self):
        """Reset daily tracking state."""
        self.last_trade_exit_time = None
        self.trades_today = 0

    def can_trade(self, current_time: datetime) -> Tuple[bool, str]:
        """
        Check if trading is allowed right now.
        Returns (can_trade, reason)
        """
        # Check max trades per day
        if self.trades_today >= self.config.max_trades_per_day:
            return False, f"Max trades reached ({self.config.max_trades_per_day})"

        # Check cooldown
        in_cooldown, minutes_left = self.is_in_cooldown(current_time)
        if in_cooldown:
            return False, f"In cooldown ({minutes_left}m remaining)"

        # Check trading window
        entry_start = datetime.strptime(self.config.entry_time_start, "%H:%M:%S").time()
        entry_end = datetime.strptime(self.config.entry_time_end, "%H:%M:%S").time()
        current = current_time.time()

        if current < entry_start:
            return False, f"Before entry window ({self.config.entry_time_start})"
        if current > entry_end:
            return False, f"After entry window ({self.config.entry_time_end})"

        return True, "OK"

    def analyze_volume(self, current_volume: float, avg_volume: float) -> VolumeAnalysis:
        """
        Analyze volume for entry validation.
        """
        analysis = VolumeAnalysis(
            current_volume=current_volume,
            avg_volume=avg_volume
        )

        if avg_volume > 0:
            analysis.volume_ratio = current_volume / avg_volume
        else:
            analysis.volume_ratio = 1.0

        # Classify volume
        if analysis.volume_ratio >= self.config.volume_spike_multiplier:
            analysis.is_spike = True
            analysis.trend = "SPIKE"
        elif analysis.volume_ratio >= 1.2:
            analysis.trend = "HIGH"
        elif analysis.volume_ratio >= 0.8:
            analysis.trend = "NORMAL"
        else:
            analysis.trend = "LOW"

        return analysis

    def calculate_technical_score(
        self,
        price: float,
        indicators: TechnicalIndicators
    ) -> Tuple[int, str, List[str]]:
        """
        Calculate technical score for entry signal.
        Returns (score, direction, reasons)

        Score components (6 total):
        1. EMA Stack alignment
        2. VWAP position
        3. RSI momentum
        4. MACD crossover
        5. Bollinger Band position
        6. Volume confirmation
        """
        bullish_points = 0
        bearish_points = 0
        reasons = []

        # 1. EMA Stack (Price vs EMA9 vs EMA20)
        if price > indicators.ema_9 > indicators.ema_20:
            bullish_points += 1
            reasons.append("EMA_BULL")
        elif price < indicators.ema_9 < indicators.ema_20:
            bearish_points += 1
            reasons.append("EMA_BEAR")

        # 2. VWAP Position
        if indicators.vwap > 0:
            if price > indicators.vwap * 1.002:  # Above VWAP by 0.2%
                bullish_points += 1
                reasons.append("VWAP_BULL")
            elif price < indicators.vwap * 0.998:  # Below VWAP by 0.2%
                bearish_points += 1
                reasons.append("VWAP_BEAR")

        # 3. RSI Momentum
        if indicators.rsi > 55:
            bullish_points += 1
            reasons.append(f"RSI_BULL({indicators.rsi:.0f})")
        elif indicators.rsi < 45:
            bearish_points += 1
            reasons.append(f"RSI_BEAR({indicators.rsi:.0f})")

        # 4. MACD Crossover
        macd_diff = indicators.macd - indicators.macd_signal
        if macd_diff > 0.01:
            bullish_points += 1
            reasons.append("MACD_BULL")
        elif macd_diff < -0.01:
            bearish_points += 1
            reasons.append("MACD_BEAR")

        # 5. Bollinger Band Position
        if indicators.bb_upper > 0 and indicators.bb_lower > 0:
            bb_range = indicators.bb_upper - indicators.bb_lower
            if bb_range > 0:
                bb_pct = (price - indicators.bb_lower) / bb_range
                if bb_pct > 0.7:
                    bullish_points += 1
                    reasons.append(f"BB_BULL({bb_pct:.0%})")
                elif bb_pct < 0.3:
                    bearish_points += 1
                    reasons.append(f"BB_BEAR({bb_pct:.0%})")

        # 6. Volume Confirmation
        if indicators.avg_volume > 0:
            vol_ratio = indicators.volume / indicators.avg_volume
            if vol_ratio >= self.config.volume_spike_multiplier:
                # Volume confirms whatever direction is stronger
                if bullish_points > bearish_points:
                    bullish_points += 1
                    reasons.append(f"VOL_CONFIRM({vol_ratio:.1f}x)")
                elif bearish_points > bullish_points:
                    bearish_points += 1
                    reasons.append(f"VOL_CONFIRM({vol_ratio:.1f}x)")

        # Determine direction
        if bullish_points >= self.config.min_signal_score and bullish_points > bearish_points:
            return bullish_points, "BULLISH", reasons
        elif bearish_points >= self.config.min_signal_score and bearish_points > bullish_points:
            return bearish_points, "BEARISH", reasons
        else:
            return max(bullish_points, bearish_points), "NEUTRAL", reasons

    def analyze_price_action(self, bars: List[Dict], verbose: bool = False) -> PriceActionSignal:
        """
        Analyze 5-minute price action for momentum.
        Expects list of OHLCV bars (most recent last).

        PA Score Components (5 total, need ≥3 for directional signal):
        1. Candle Trend: ≥3 green candles (bullish) or ≥3 red (bearish)
        2. Higher Highs (bullish) / Lower Lows (bearish)
        3. Momentum: >0.3% (bullish) or <-0.3% (bearish)
        4. Last Bar Strength: close near high >0.7 (bullish) or near low <0.3 (bearish)
        5. Above/Below 5-bar average by 0.2%
        """
        signal = PriceActionSignal()

        if len(bars) < 5:
            if verbose:
                print(f"    [PA] Not enough bars: {len(bars)}/5 required")
            return signal

        recent_bars = bars[-5:]

        # Count green vs red candles (support both dict and object access)
        def get_bar_val(bar, key):
            return bar[key] if isinstance(bar, dict) else getattr(bar, key)

        green_count = sum(1 for b in recent_bars if get_bar_val(b, 'close') > get_bar_val(b, 'open'))
        red_count = 5 - green_count

        # Higher highs / Lower lows
        highs = [get_bar_val(b, 'high') for b in recent_bars]
        lows = [get_bar_val(b, 'low') for b in recent_bars]

        higher_highs = all(highs[i] >= highs[i-1] for i in range(1, len(highs)))
        lower_lows = all(lows[i] <= lows[i-1] for i in range(1, len(lows)))

        signal.higher_highs = higher_highs
        signal.lower_lows = lower_lows

        # 5-bar momentum
        first_close = get_bar_val(recent_bars[0], 'close')
        last_close = get_bar_val(recent_bars[-1], 'close')
        if first_close > 0:
            signal.momentum = (last_close - first_close) / first_close * 100

        # Last bar strength
        last_bar = recent_bars[-1]
        bar_range = get_bar_val(last_bar, 'high') - get_bar_val(last_bar, 'low')
        if bar_range > 0:
            signal.last_bar_strength = (get_bar_val(last_bar, 'close') - get_bar_val(last_bar, 'low')) / bar_range

        # 5-bar average comparison
        avg_close = sum(get_bar_val(b, 'close') for b in recent_bars) / 5

        # Score calculation with verbose logging
        bullish_score = 0
        bearish_score = 0
        pa_details = []

        # 1. Candle Trend
        if green_count >= 3:
            bullish_score += 1
            signal.candle_trend = f"{green_count}/5 GREEN"
            pa_details.append(f"Candles:{green_count}/5 GREEN +1B")
        elif red_count >= 3:
            bearish_score += 1
            signal.candle_trend = f"{red_count}/5 RED"
            pa_details.append(f"Candles:{red_count}/5 RED +1S")
        else:
            pa_details.append(f"Candles:{green_count}G/{red_count}R (mixed)")

        # 2. Higher Highs / Lower Lows
        if higher_highs:
            bullish_score += 1
            pa_details.append("HH +1B")
        elif lower_lows:
            bearish_score += 1
            pa_details.append("LL +1S")
        else:
            pa_details.append("No HH/LL")

        # 3. Momentum
        if signal.momentum > 0.3:
            bullish_score += 1
            pa_details.append(f"Mom:{signal.momentum:+.2f}% +1B")
        elif signal.momentum < -0.3:
            bearish_score += 1
            pa_details.append(f"Mom:{signal.momentum:+.2f}% +1S")
        else:
            pa_details.append(f"Mom:{signal.momentum:+.2f}% (flat)")

        # 4. Last Bar Strength
        if signal.last_bar_strength > 0.7:
            bullish_score += 1
            pa_details.append(f"LastBar:{signal.last_bar_strength:.0%} +1B")
        elif signal.last_bar_strength < 0.3:
            bearish_score += 1
            pa_details.append(f"LastBar:{signal.last_bar_strength:.0%} +1S")
        else:
            pa_details.append(f"LastBar:{signal.last_bar_strength:.0%} (mid)")

        # 5. Above/Below 5-bar average
        pct_from_avg = (last_close - avg_close) / avg_close * 100
        if last_close > avg_close * 1.002:
            bullish_score += 1
            pa_details.append(f"vsAvg:{pct_from_avg:+.2f}% +1B")
        elif last_close < avg_close * 0.998:
            bearish_score += 1
            pa_details.append(f"vsAvg:{pct_from_avg:+.2f}% +1S")
        else:
            pa_details.append(f"vsAvg:{pct_from_avg:+.2f}% (near)")

        signal.score = max(bullish_score, bearish_score)

        if bullish_score >= 3 and bullish_score > bearish_score:
            signal.direction = "BULLISH"
        elif bearish_score >= 3 and bearish_score > bullish_score:
            signal.direction = "BEARISH"
        else:
            signal.direction = "NEUTRAL"

        # Store details for verbose output
        signal.candle_trend = f"{green_count}G/{red_count}R | " + " | ".join(pa_details)

        if verbose:
            print(f"    [PA] {signal.direction}({signal.score}) Bull={bullish_score} Bear={bearish_score}")
            for detail in pa_details:
                print(f"        {detail}")

        return signal

    def validate_contract(
        self,
        contract: ATMContractInfo,
        underlying_price: float
    ) -> ATMContractInfo:
        """
        Validate that a contract meets liquidity requirements.
        """
        contract.is_valid = True

        # Check bid/ask spread
        if contract.mid > 0:
            contract.spread_pct = (contract.ask - contract.bid) / contract.mid * 100
        else:
            contract.is_valid = False
            contract.rejection_reason = "No mid price"
            return contract

        if contract.spread_pct > self.config.max_bid_ask_spread_pct:
            contract.is_valid = False
            contract.rejection_reason = f"Spread too wide ({contract.spread_pct:.1f}%)"
            return contract

        # Check volume
        if contract.volume < self.config.min_option_volume:
            contract.is_valid = False
            contract.rejection_reason = f"Low volume ({contract.volume})"
            return contract

        # Check open interest
        if contract.open_interest < self.config.min_open_interest:
            contract.is_valid = False
            contract.rejection_reason = f"Low OI ({contract.open_interest})"
            return contract

        # Check days to expiry
        if contract.days_to_expiry < self.config.min_days_to_expiry:
            contract.is_valid = False
            contract.rejection_reason = f"DTE too low ({contract.days_to_expiry})"
            return contract

        if contract.days_to_expiry > self.config.max_days_to_expiry:
            contract.is_valid = False
            contract.rejection_reason = f"DTE too high ({contract.days_to_expiry})"
            return contract

        return contract

    def get_entry_signal(
        self,
        price: float,
        indicators: TechnicalIndicators,
        price_action: PriceActionSignal,
        volume_analysis: VolumeAnalysis,
        current_time: datetime
    ) -> Tuple[bool, str, str, List[str]]:
        """
        Determine if we should enter a trade.
        Returns (should_enter, direction, option_type, reasons)
        """
        reasons = []

        # Check if we can trade
        can_trade, trade_reason = self.can_trade(current_time)
        if not can_trade:
            return False, "NONE", "", [trade_reason]

        # Check volume requirement
        if not volume_analysis.is_spike:
            if volume_analysis.current_volume < self.config.min_volume_threshold:
                return False, "NONE", "", ["Volume below threshold"]
            reasons.append(f"Vol: {volume_analysis.trend}")
        else:
            reasons.append(f"Vol SPIKE: {volume_analysis.volume_ratio:.1f}x")

        # Get technical score
        tech_score, tech_direction, tech_reasons = self.calculate_technical_score(
            price, indicators
        )
        reasons.extend(tech_reasons)

        # Check dual confirmation
        if self.config.require_dual_confirmation:
            if tech_direction != price_action.direction:
                return False, "CONFLICT", "", [
                    f"Tech={tech_direction}, PA={price_action.direction}"
                ]
            if tech_direction == "NEUTRAL" or price_action.direction == "NEUTRAL":
                return False, "NEUTRAL", "", ["No clear direction"]

        # Determine option type
        if tech_direction == "BULLISH" and price_action.direction == "BULLISH":
            return True, "BULLISH", "call", reasons
        elif tech_direction == "BEARISH" and price_action.direction == "BEARISH":
            return True, "BEARISH", "put", reasons

        return False, "NONE", "", reasons

    def calculate_position_size(self, option_price: float) -> int:
        """Calculate number of contracts based on position value."""
        if option_price <= 0:
            return 0

        contract_value = option_price * 100  # Options are 100 shares
        qty = int(self.config.fixed_position_value / contract_value)

        return max(1, qty)  # At least 1 contract

    def get_weekly_expiry(self, from_date: datetime) -> datetime:
        """
        Get the next weekly (Friday) expiry date.
        If today is Friday and before market close, use today.
        Otherwise use next Friday.
        """
        # Find next Friday
        days_until_friday = (4 - from_date.weekday()) % 7

        # If it's Friday and we're in trading hours, use today
        if days_until_friday == 0:
            return from_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # Otherwise next Friday
        if days_until_friday == 0:
            days_until_friday = 7

        next_friday = from_date + timedelta(days=days_until_friday)
        return next_friday.replace(hour=0, minute=0, second=0, microsecond=0)
