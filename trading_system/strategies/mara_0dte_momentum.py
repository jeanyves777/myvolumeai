"""
MARA Daily 0DTE Momentum Strategy - Alpaca Implementation

MARA (Marathon Digital) 0DTE Options Strategy using the same
dual-confirmation signal system as COIN.

# Find THIS WEEK's Friday - options expire same week at 4PM Friday
# Monday (0) -> Friday in 4 days
# Tuesday (1) -> Friday in 3 days
# Wednesday (2) -> Friday in 2 days
# Thursday (3) -> Friday in 1 day
# Friday (4) -> Friday TODAY (0DTE)

CRITICAL REQUIREMENTS:
======================
1. MUST open exactly ONE trade per day (no exceptions)
2. FIXED position size: $2000 per day
3. Indicators determine DIRECTION only (CALL vs PUT)
4. If indicators neutral/weak -> NO TRADE (MARA is more volatile)
5. Entry window: 9:30-15:45 EST (extended for MARA)
6. Force exit: 15:50 PM EST (before 4 PM 0DTE expiration)
7. NEVER holds overnight

STRATEGY LOGIC:
===============
- Uses EMA, RSI, MACD, Bollinger Bands, Volume for direction analysis
- Buys ATM (at-the-money) weekly options based on market direction
- Target profit: 7.5% (configurable)
- Stop loss: 25% (configurable)
- MARA has smaller option premiums, so we can buy more contracts
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
import pytz


@dataclass
class MARADaily0DTEMomentumConfig:
    """
    Configuration for MARA Daily 0DTE Momentum Strategy.
    """
    underlying_symbol: str = "MARA"
    fixed_position_value: float = 200.0
    target_profit_pct: float = 7.5
    stop_loss_pct: float = 25.0
    entry_time_start: str = "09:30:00"
    entry_time_end: str = "15:45:00"
    force_exit_time: str = "15:50:00"
    max_hold_minutes: int = 30
    max_trades_per_day: int = 1
    poll_interval_seconds: int = 10

    # Indicator settings
    fast_ema_period: int = 9
    slow_ema_period: int = 20
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    bb_period: int = 20
    bb_std_dev: float = 2.0


class MARADaily0DTEMomentum:
    """
    MARA Daily 0DTE Momentum Strategy - Signal Calculator

    This class provides standalone methods for signal calculation
    that can be used by any trading engine.
    """

    @staticmethod
    def calculate_signal_from_bars(bars: list, config: MARADaily0DTEMomentumConfig = None) -> dict:
        """
        Calculate trading signal from raw bar data.

        Parameters
        ----------
        bars : list
            List of bar objects with: close, high, low, volume attributes
        config : MARADaily0DTEMomentumConfig, optional
            Strategy configuration

        Returns
        -------
        dict
            {
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'bullish_score': int,
                'bearish_score': int,
                'confidence': str,  # 'HIGH' | 'MEDIUM' | 'LOW'
                'signals': list,    # List of signal reasons
                'indicators': dict  # All indicator values
            }
        """
        if not bars or len(bars) < 30:
            return {
                'signal': 'NEUTRAL',
                'bullish_score': 0,
                'bearish_score': 0,
                'confidence': 'LOW',
                'signals': ['Not enough data'],
                'indicators': {}
            }

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [getattr(b, 'volume', 0) for b in bars]
        current_price = closes[-1]

        # Calculate indicators
        def calc_ema(prices, period):
            if len(prices) < period:
                return sum(prices) / len(prices)
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
            return ema

        def calc_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            gains, losses = [], []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(0, change))
                losses.append(max(0, -change))
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        def calc_bb(prices, period=20, std_dev=2.0):
            if len(prices) < period:
                return None, None, None
            recent = prices[-period:]
            middle = sum(recent) / period
            variance = sum((p - middle) ** 2 for p in recent) / period
            std = variance ** 0.5
            return middle - std_dev * std, middle, middle + std_dev * std

        def calc_vwap(bars_list):
            total_vol = sum(getattr(b, 'volume', 0) for b in bars_list)
            if total_vol == 0:
                return bars_list[-1].close
            total_vwap = sum(((b.high + b.low + b.close) / 3) * getattr(b, 'volume', 0) for b in bars_list)
            return total_vwap / total_vol

        def calc_atr(bars_list, period=14):
            if len(bars_list) < 2:
                return 0
            trs = []
            for i in range(1, len(bars_list)):
                tr = max(
                    bars_list[i].high - bars_list[i].low,
                    abs(bars_list[i].high - bars_list[i-1].close),
                    abs(bars_list[i].low - bars_list[i-1].close)
                )
                trs.append(tr)
            return sum(trs[-period:]) / min(period, len(trs))

        # Calculate all indicators
        ema_9 = calc_ema(closes, 9)
        ema_20 = calc_ema(closes, 20)
        ema_50 = calc_ema(closes, min(50, len(closes)))
        rsi = calc_rsi(closes, 14)
        bb_lower, bb_middle, bb_upper = calc_bb(closes, 20, 2.0)
        vwap = calc_vwap(bars)
        atr = calc_atr(bars, 14)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        # MACD
        ema_12 = calc_ema(closes, 12)
        ema_26 = calc_ema(closes, 26)
        macd_line = ema_12 - ema_26
        macd_signal = macd_line * 0.9  # Approximation
        macd_hist = macd_line - macd_signal

        # Volume analysis
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / max(1, len(volumes))
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        # ========== SCORING SYSTEM ==========
        bullish_score = 0
        bearish_score = 0
        bullish_signals = []
        bearish_signals = []

        # 1. EMA STACK (up to 4 points)
        if current_price > ema_9 > ema_20:
            bullish_score += 4
            bullish_signals.append("EMA Stack Bullish")
        elif current_price < ema_9 < ema_20:
            bearish_score += 4
            bearish_signals.append("EMA Stack Bearish")
        elif current_price > ema_20:
            bullish_score += 2
            bullish_signals.append("Above EMA20")
        elif current_price < ema_20:
            bearish_score += 2
            bearish_signals.append("Below EMA20")

        # 2. VWAP (up to 3 points)
        vwap_dist = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
        if current_price > vwap:
            bullish_score += 2
            bullish_signals.append(f"Above VWAP (+{vwap_dist:.2f}%)")
            if abs(vwap_dist) < 0.3:
                bullish_score += 1
                bullish_signals.append("Near VWAP support")
        else:
            bearish_score += 2
            bearish_signals.append(f"Below VWAP ({vwap_dist:.2f}%)")
            if abs(vwap_dist) < 0.3:
                bearish_score += 1
                bearish_signals.append("Near VWAP resistance")

        # 3. RSI (up to 3 points)
        if 30 <= rsi <= 50:
            bullish_score += 3
            bullish_signals.append(f"RSI bullish zone ({rsi:.1f})")
        elif 50 < rsi <= 65:
            bullish_score += 2
            bullish_signals.append(f"RSI momentum ({rsi:.1f})")
        elif rsi > 70:
            bearish_score += 3
            bearish_signals.append(f"RSI OVERBOUGHT ({rsi:.1f})")
        elif rsi < 30:
            bullish_score += 3
            bullish_signals.append(f"RSI OVERSOLD ({rsi:.1f})")
        elif 50 <= rsi <= 70:
            bearish_score += 2
            bearish_signals.append(f"RSI neutral-high ({rsi:.1f})")

        # 4. MACD (up to 3 points)
        if macd_hist > 0 and macd_line > macd_signal:
            bullish_score += 3
            bullish_signals.append("MACD Bullish")
        elif macd_hist < 0 and macd_line < macd_signal:
            bearish_score += 3
            bearish_signals.append("MACD Bearish")

        # 5. Bollinger Bands (up to 2 points)
        if bb_middle:
            if current_price > bb_middle:
                bullish_score += 2
                bullish_signals.append("Above BB Middle")
            else:
                bearish_score += 2
                bearish_signals.append("Below BB Middle")

        # 6. Volume (up to 2 points)
        if volume_ratio > 1.5:
            price_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
            if price_change > 0:
                bullish_score += 2
                bullish_signals.append(f"High volume bullish ({volume_ratio:.1f}x)")
            else:
                bearish_score += 2
                bearish_signals.append(f"High volume bearish ({volume_ratio:.1f}x)")

        # Volatility adjustment (MARA is more volatile, so less adjustment)
        if atr_pct > 3.0:  # Higher threshold for MARA
            bullish_score = int(bullish_score * 0.8)
            bearish_score = int(bearish_score * 0.8)

        # ========== DECISION ==========
        MIN_SCORE = 8
        MIN_LEAD = 4

        if bullish_score >= MIN_SCORE and bullish_score >= bearish_score + MIN_LEAD:
            signal = 'BULLISH'
            confidence = 'HIGH'
        elif bearish_score >= MIN_SCORE and bearish_score >= bullish_score + MIN_LEAD:
            signal = 'BEARISH'
            confidence = 'HIGH'
        elif bullish_score >= 6 or bearish_score >= 6:
            signal = 'BULLISH' if bullish_score >= bearish_score else 'BEARISH'
            confidence = 'MEDIUM'
        else:
            signal = 'NEUTRAL'
            confidence = 'LOW'

        return {
            'signal': signal,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'confidence': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'indicators': {
                'price': current_price,
                'ema_9': ema_9,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'rsi': rsi,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_upper': bb_upper,
                'vwap': vwap,
                'atr': atr,
                'atr_pct': atr_pct,
                'volume_ratio': volume_ratio,
            }
        }

    @staticmethod
    def calculate_price_action_signal(bars: list) -> dict:
        """
        Calculate trading signal based on PRICE ACTION patterns.

        This is METHOD 2 for dual-confirmation signal validation.

        Parameters
        ----------
        bars : list
            List of bar objects with: open, close, high, low attributes

        Returns
        -------
        dict
            {
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'strength': str,  # 'STRONG' | 'MODERATE' | 'WEAK'
                'bullish_points': int,
                'bearish_points': int,
                'reasons': list
            }
        """
        if not bars or len(bars) < 10:
            return {
                'signal': 'NEUTRAL',
                'strength': 'WEAK',
                'bullish_points': 0,
                'bearish_points': 0,
                'reasons': ['Not enough bars for price action analysis']
            }

        # Use last 10 bars for price action analysis
        recent_bars = bars[-10:]
        closes = [b.close for b in recent_bars]
        opens = [b.open for b in recent_bars]
        highs = [b.high for b in recent_bars]
        lows = [b.low for b in recent_bars]

        bullish_points = 0
        bearish_points = 0
        reasons = []

        # ========== 1. CANDLE COLOR COUNT (last 5 bars) ==========
        last_5 = recent_bars[-5:]
        green_candles = sum(1 for b in last_5 if b.close > b.open)
        red_candles = 5 - green_candles

        if green_candles >= 4:
            bullish_points += 3
            reasons.append(f"{green_candles}/5 bars bullish (green)")
        elif green_candles >= 3:
            bullish_points += 2
            reasons.append(f"{green_candles}/5 bars bullish")
        elif red_candles >= 4:
            bearish_points += 3
            reasons.append(f"{red_candles}/5 bars bearish (red)")
        elif red_candles >= 3:
            bearish_points += 2
            reasons.append(f"{red_candles}/5 bars bearish")

        # ========== 2. HIGHER HIGHS / LOWER LOWS (last 5 bars) ==========
        higher_highs = sum(1 for i in range(1, 5) if highs[-i] > highs[-i-1])
        lower_lows = sum(1 for i in range(1, 5) if lows[-i] < lows[-i-1])
        higher_lows = sum(1 for i in range(1, 5) if lows[-i] > lows[-i-1])
        lower_highs = sum(1 for i in range(1, 5) if highs[-i] < highs[-i-1])

        # Bullish: Higher highs AND higher lows (uptrend)
        if higher_highs >= 3 and higher_lows >= 2:
            bullish_points += 3
            reasons.append(f"Uptrend: {higher_highs} higher highs, {higher_lows} higher lows")
        elif higher_highs >= 2:
            bullish_points += 2
            reasons.append(f"Higher highs pattern ({higher_highs}/4)")

        # Bearish: Lower lows AND lower highs (downtrend)
        if lower_lows >= 3 and lower_highs >= 2:
            bearish_points += 3
            reasons.append(f"Downtrend: {lower_lows} lower lows, {lower_highs} lower highs")
        elif lower_lows >= 2:
            bearish_points += 2
            reasons.append(f"Lower lows pattern ({lower_lows}/4)")

        # ========== 3. PRICE vs 5-BAR AVERAGE ==========
        avg_5 = sum(closes[-5:]) / 5
        current_price = closes[-1]
        price_vs_avg = ((current_price - avg_5) / avg_5) * 100

        if price_vs_avg > 0.1:
            bullish_points += 2
            reasons.append(f"Price above 5-bar avg (+{price_vs_avg:.2f}%)")
        elif price_vs_avg < -0.1:
            bearish_points += 2
            reasons.append(f"Price below 5-bar avg ({price_vs_avg:.2f}%)")

        # ========== 4. MOMENTUM (5-bar change) ==========
        momentum_5 = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0

        if momentum_5 > 0.15:
            bullish_points += 2
            reasons.append(f"Strong 5-bar momentum (+{momentum_5:.2f}%)")
        elif momentum_5 > 0.05:
            bullish_points += 1
            reasons.append(f"Positive momentum (+{momentum_5:.2f}%)")
        elif momentum_5 < -0.15:
            bearish_points += 2
            reasons.append(f"Strong bearish momentum ({momentum_5:.2f}%)")
        elif momentum_5 < -0.05:
            bearish_points += 1
            reasons.append(f"Negative momentum ({momentum_5:.2f}%)")

        # ========== 5. LAST BAR ANALYSIS ==========
        last_bar = recent_bars[-1]
        bar_range = last_bar.high - last_bar.low
        bar_body = abs(last_bar.close - last_bar.open)
        body_ratio = bar_body / bar_range if bar_range > 0 else 0

        # Strong bullish bar: closes near high with good body
        if last_bar.close > last_bar.open and body_ratio > 0.6:
            upper_wick = last_bar.high - last_bar.close
            if upper_wick < bar_body * 0.3:  # Small upper wick
                bullish_points += 2
                reasons.append("Strong bullish last bar (closes near high)")

        # Strong bearish bar: closes near low with good body
        if last_bar.close < last_bar.open and body_ratio > 0.6:
            lower_wick = last_bar.close - last_bar.low
            if lower_wick < bar_body * 0.3:  # Small lower wick
                bearish_points += 2
                reasons.append("Strong bearish last bar (closes near low)")

        # ========== DECISION ==========
        if bullish_points >= 5 and bullish_points >= bearish_points + 3:
            signal = 'BULLISH'
            strength = 'STRONG'
        elif bearish_points >= 5 and bearish_points >= bullish_points + 3:
            signal = 'BEARISH'
            strength = 'STRONG'
        elif bullish_points >= 4 and bullish_points > bearish_points:
            signal = 'BULLISH'
            strength = 'MODERATE'
        elif bearish_points >= 4 and bearish_points > bullish_points:
            signal = 'BEARISH'
            strength = 'MODERATE'
        else:
            signal = 'NEUTRAL'
            strength = 'WEAK'

        return {
            'signal': signal,
            'strength': strength,
            'bullish_points': bullish_points,
            'bearish_points': bearish_points,
            'reasons': reasons
        }
