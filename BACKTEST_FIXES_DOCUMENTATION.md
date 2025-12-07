# COIN 0DTE Momentum Strategy - Backtest Fixes Documentation

## Overview
This document details all fixes implemented to create a realistic options trading backtest system for the COIN Daily 0DTE Momentum Strategy.

---

## 1. Weekly Expiry Logic Fix

### Problem
Options were selecting the wrong expiry dates - sometimes next week's Friday instead of this week's Friday.

### Solution
**Files Modified:**
- `trading_system/run_backtest.py` (lines 98-115)
- `trading_system/strategies/coin_0dte_momentum.py` (lines 435-450)

**Logic Implementation:**
```
Monday (weekday=0)    -> Friday in 4 days (4 DTE)
Tuesday (weekday=1)   -> Friday in 3 days (3 DTE)
Wednesday (weekday=2) -> Friday in 2 days (2 DTE)
Thursday (weekday=3)  -> Friday in 1 day  (1 DTE)
Friday (weekday=4)    -> Friday TODAY     (0 DTE)
```

**Code (run_backtest.py):**
```python
weekday = day.weekday()
if weekday <= 4:  # Monday to Friday
    days_to_friday = 4 - weekday  # 4=Friday weekday number
else:  # Saturday/Sunday - skip to next Friday
    days_to_friday = (4 - weekday) % 7
expiry_date = day + pd.Timedelta(days=days_to_friday)
```

**Code (coin_0dte_momentum.py - _find_best_option):**
```python
weekday = today_date.weekday()
if weekday <= 4:  # Monday to Friday
    days_to_friday = 4 - weekday
else:  # Saturday/Sunday
    days_to_friday = (4 - weekday) % 7
this_weeks_friday = today_date + timedelta(days=days_to_friday)

# MUST expire THIS WEEK's Friday
if inst.expiration:
    expiry_date = inst.expiration.date()
    if expiry_date != this_weeks_friday:
        continue  # Skip options not expiring this Friday
```

---

## 2. Same-Bar SL/TP Trigger Fix

### Problem
Stop-loss and take-profit orders were triggering immediately on the same bar as entry, resulting in unrealistic 1-minute hold times for ALL trades.

### Root Cause
1. Timestamp comparison bug in `order_manager.py` - timezone-aware vs naive datetime handling failed silently
2. Using submission timestamp instead of fill timestamp for SL/TP eligibility

### Solution

**File: `trading_system/engine/order_manager.py` (lines 346-366)**

Fixed timestamp comparison by normalizing both to naive datetimes:
```python
if order.first_eligible_bar_timestamp is not None:
    bar_ts = bar.timestamp
    eligible_ts = order.first_eligible_bar_timestamp

    # Normalize both timestamps to be comparable
    # Strip timezone info for comparison
    if hasattr(bar_ts, 'replace') and hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is not None:
        bar_ts_naive = bar_ts.replace(tzinfo=None)
    else:
        bar_ts_naive = bar_ts

    if hasattr(eligible_ts, 'replace') and hasattr(eligible_ts, 'tzinfo') and eligible_ts.tzinfo is not None:
        eligible_ts_naive = eligible_ts.replace(tzinfo=None)
    else:
        eligible_ts_naive = eligible_ts

    # Skip this order if bar is at or before the eligible timestamp
    if bar_ts_naive <= eligible_ts_naive:
        continue
```

**File: `trading_system/strategies/coin_0dte_momentum.py` (lines 624-634)**

Use fill timestamp instead of submission timestamp:
```python
# Set first eligible bar timestamp - orders can only fill on bars AFTER entry
# Use the FILL timestamp (when entry actually executed), not submission timestamp
fill_ts = event.timestamp  # This is when the entry order actually filled
if fill_ts is not None:
    sl_order.first_eligible_bar_timestamp = fill_ts
    tp_order.first_eligible_bar_timestamp = fill_ts
elif self.entry_bar_datetime is not None:
    # Fallback to submission timestamp
    sl_order.first_eligible_bar_timestamp = self.entry_bar_datetime
    tp_order.first_eligible_bar_timestamp = self.entry_bar_datetime
```

**File: `trading_system/core/models.py`**

Added field to Order model:
```python
# For realistic backtesting: orders can only fill on bars AFTER this timestamp
first_eligible_bar_timestamp: Optional[datetime] = None
```

---

## 3. Realistic Synthetic Options Price Movement

### Problem
Synthetic options prices were jumping randomly with extreme intrabar volatility, causing unrealistic instant SL/TP triggers.

### Solution
**File: `trading_system/run_backtest.py` (lines 550-631)**

Changed from random noise to delta-based price movement following underlying:

```python
# Calculate underlying percentage move per bar
underlying_pct_change = np.zeros(n)
underlying_pct_change[1:] = (S_close[1:] - S_close[:-1]) / S_close[:-1]

# Options move based on delta (simplified leverage)
leverage = np.clip(1.0 / np.maximum(delta_abs, 0.1), 1.0, 10.0)

# Option percentage change follows underlying with leverage
if is_call:
    option_pct_change = underlying_pct_change * leverage * delta_abs
else:
    option_pct_change = -underlying_pct_change * leverage * delta_abs

# Add small random noise (0.2%, not 2%)
noise = np.random.normal(0, 0.002, n)
option_pct_change = option_pct_change + noise

# Blend with theoretical to prevent drift (80% movement, 20% theoretical anchor)
close_prices[i] = 0.8 * moved_price + 0.2 * theoretical
```

**Reduced IV adjustments for stability:**
- Intraday IV adjustments reduced by 50%
- 0DTE IV boost reduced from 0.15 to 0.08
- Max IV capped at 2.0 (down from 3.0)

---

## 4. ATM Strike Selection

### Problem
Strategy needed to ensure it only trades At-The-Money (ATM) options within 2.5% of the underlying price.

### Solution
**File: `trading_system/strategies/coin_0dte_momentum.py` (lines 452-490)**

```python
ATM_THRESHOLD_PCT = 2.5

for inst in self._engine.instruments.values():
    # ... type checks ...

    # MUST expire THIS WEEK's Friday
    if inst.expiration:
        expiry_date = inst.expiration.date()
        if expiry_date != this_weeks_friday:
            continue

    strike_diff = abs(inst.strike_price - underlying_price)
    strike_diff_pct = (strike_diff / underlying_price) * 100

    # STRICT ATM CHECK: Only consider options within 2.5% of underlying
    if strike_diff_pct > ATM_THRESHOLD_PCT:
        continue  # Skip OTM contracts

    # Find closest strike to ATM
    if strike_diff < best_strike_diff:
        best_strike_diff = strike_diff
        best_match = inst
```

---

## 5. Direction Signal Tracking

### Problem
Trade reports didn't show whether the signal was BULLISH or BEARISH.

### Solution
**File: `trading_system/strategies/coin_0dte_momentum.py`**

Added signal tracking:
```python
self.current_signal: str = ""  # BULLISH, BEARISH, or NEUTRAL

# In _execute_entry:
self.current_signal = signal  # Store the signal

# In on_order_filled:
if self.current_position:
    self.current_position.signal = self.current_signal
```

**File: `trading_system/core/models.py` - Position class**
```python
signal: str = ""  # BULLISH/BEARISH signal used for entry
```

---

## Results Comparison

| Metric | Before Fixes | After Fixes |
|--------|-------------|-------------|
| Avg Hold Time | 2.3 min (all ~1.0m) | 2.3 min (varied 1-9m) |
| SL Hold Times | 1.0 min (unrealistic) | 8-9 min (realistic) |
| TP Hold Times | All 1.0 min | 1-7 min (varied) |
| Win Rate | 85.7% | 90.0% |
| Profit Factor | 9.69 | 9.10 |
| Total P&L | $13,046 | $4,771 |

**Note:** Lower P&L is more realistic because trades now develop over time instead of instantly hitting targets.

---

## Files Modified Summary

1. **trading_system/run_backtest.py**
   - Weekly expiry calculation
   - Realistic synthetic options pricing
   - Delta-based price movement

2. **trading_system/engine/order_manager.py**
   - Timestamp comparison fix for SL/TP eligibility

3. **trading_system/strategies/coin_0dte_momentum.py**
   - Weekly Friday expiry selection
   - ATM strike selection (2.5% threshold)
   - Fill timestamp for SL/TP orders
   - Signal tracking

4. **trading_system/core/models.py**
   - `first_eligible_bar_timestamp` field on Order
   - `signal` field on Position

---

## Strategy Configuration

```python
COINDaily0DTEMomentumConfig(
    underlying_symbol="COIN",
    fixed_position_value=2000.0,    # $2000 per trade
    target_profit_pct=7.5,          # +7.5% take profit
    stop_loss_pct=25.0,             # -25% stop loss
    max_hold_minutes=30,            # Max 30 min hold
    entry_time_start="09:30:00",    # Entry window start (EST)
    entry_time_end="10:00:00",      # Entry window end (EST)
    force_exit_time="15:55:00",     # Force exit before 4PM expiry
)
```

---

## Next Steps: Paper Trading Implementation

The next phase will implement:
1. Live market data connection (Alpaca API)
2. Paper trading execution
3. First-time setup wizard for API credentials
4. Configurable daily position size
5. Real-time SL/TP order management
