# Crypto Scalping Strategy - Implementation Plan

## Overview

A **24/7 continuous scalping strategy** for crypto SPOT trading on **Alpaca only**.

**Core Philosophy**: Buy low, sell high with tight spreads. Continuous trading when market conditions are favorable.

> **Note**: Futures/Perpetuals trading (Kraken, Binance) will be built as separate strategies later.

---

## 1. Strategy Logic: "Buy Low, Sell High" Scalping

### Entry Conditions (BUY Signal)
Must meet **ALL** conditions:
1. **RSI Oversold**: RSI < 30 (or configurable threshold)
2. **Price at Lower Bollinger Band**: Price touches or breaks below lower BB
3. **EMA Alignment**: Price below fast EMA (pullback in trend)
4. **Volume Spike**: Current volume > 1.5x average volume (confirmation)
5. **VWAP**: Price below VWAP (buying below fair value)

### Exit Conditions (SELL Signal)
Exit when **ANY** condition is met:
1. **Take Profit**: +0.5% to +2% profit (configurable)
2. **Stop Loss**: -0.3% to -1% loss (tight for scalping)
3. **RSI Overbought**: RSI > 70 (reversal signal)
4. **Price at Upper Bollinger Band**: Price touches upper BB
5. **Trailing Stop**: Lock in profits as price moves up

### SHORT Entry Conditions (for Futures/Perpetuals)
Must meet **ALL** conditions:
1. **RSI Overbought**: RSI > 70
2. **Price at Upper Bollinger Band**: Price touches or breaks above upper BB
3. **EMA Alignment**: Price above fast EMA (pullback in downtrend)
4. **Volume Spike**: Confirmation of momentum
5. **VWAP**: Price above VWAP (selling above fair value)

---

## 2. Technical Indicators Required

### Existing Indicators (Reuse)
| Indicator | Purpose | Parameters |
|-----------|---------|------------|
| `ExponentialMovingAverage` | Trend direction | Fast: 9, Slow: 21 |
| `RelativeStrengthIndex` | Overbought/Oversold | Period: 14 |
| `BollingerBands` | Support/Resistance bands | Period: 20, StdDev: 2.0 |
| `AverageTrueRange` | Volatility for SL sizing | Period: 14 |
| `MACD` | Momentum confirmation | 12/26/9 |

### New Indicators to Implement
| Indicator | Purpose | Parameters |
|-----------|---------|------------|
| `VWAP` | Volume-weighted fair value | Reset: Daily/Session |
| `VolumeMA` | Volume average for spike detection | Period: 20 |
| `Stochastic` | Additional oversold/overbought | %K: 14, %D: 3 |
| `ADX` | Trend strength filter | Period: 14 |

### Indicator Logic Summary
```
BUY SIGNAL:
  RSI < 30
  AND Price <= BB_Lower
  AND Price < EMA_Fast
  AND Volume > VolumeMA * 1.5
  AND Price < VWAP
  AND ADX > 20 (trending market)

SELL SIGNAL (Exit Long):
  Price >= Entry * (1 + TP%)
  OR Price <= Entry * (1 - SL%)
  OR RSI > 70
  OR Price >= BB_Upper
  OR Trailing Stop Hit

SHORT SIGNAL (Futures Only):
  RSI > 70
  AND Price >= BB_Upper
  AND Price > EMA_Fast
  AND Volume > VolumeMA * 1.5
  AND Price > VWAP
  AND ADX > 20
```

---

## 3. Alpaca Crypto Spot Integration

### Supported Cryptocurrencies (Top 10 for Trading)

Based on [Alpaca's supported cryptocurrencies](https://alpaca.markets/support/what-cryptocurrencies-does-alpaca-currently-support):

| # | Symbol | Name | Pair | Notes |
|---|--------|------|------|-------|
| 1 | BTC | Bitcoin | BTC/USD | Largest market cap |
| 2 | ETH | Ethereum | ETH/USD | #2 by market cap |
| 3 | SOL | Solana | SOL/USD | High volatility (if available) |
| 4 | DOGE | Dogecoin | DOGE/USD | High volume meme coin |
| 5 | LINK | Chainlink | LINK/USD | Oracle leader |
| 6 | AVAX | Avalanche | AVAX/USD | Layer 1 |
| 7 | DOT | Polkadot | DOT/USD | Interoperability |
| 8 | LTC | Litecoin | LTC/USD | BTC alternative |
| 9 | UNI | Uniswap | UNI/USD | DeFi leader |
| 10 | SHIB | Shiba Inu | SHIB/USD | High volume meme |

### Alpaca API Endpoints
- **Trading**: `https://api.alpaca.markets` (live) / `https://paper-api.alpaca.markets` (paper)
- **Crypto Data**: `https://data.alpaca.markets/v1beta3/crypto/us`
- **Order Types**: Market, Limit, Stop, Stop-Limit
- **Trading Hours**: 24/7/365

### API Methods Needed
```python
# Extend existing AlpacaClient:
get_crypto_quote(symbol: str) -> Quote      # e.g., "BTC/USD"
get_crypto_bars(symbol: str, timeframe: str, start: datetime, end: datetime) -> List[Bar]
submit_crypto_order(symbol: str, qty: float, side: str, type: str) -> dict
get_crypto_positions() -> List[Position]
```

---

## 4. Configuration Structure

### 4.1 Strategy Config
```python
@dataclass
class CryptoScalpingConfig(StrategyConfig):
    """Configuration for Crypto Scalping Strategy (Alpaca Spot Only)."""

    # Symbols to trade (top 10 Alpaca crypto)
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USD", "ETH/USD", "DOGE/USD", "LINK/USD", "AVAX/USD",
        "DOT/USD", "LTC/USD", "UNI/USD", "SHIB/USD", "BCH/USD"
    ])

    # Position sizing
    fixed_position_value: float = 500.0  # USD per trade
    max_position_value: float = 2000.0   # Max exposure per symbol

    # Risk parameters
    target_profit_pct: float = 1.0       # Take profit %
    stop_loss_pct: float = 0.5           # Stop loss %
    trailing_stop_pct: float = 0.3       # Trailing stop %

    # Indicator parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std_dev: float = 2.0
    ema_fast: int = 9
    ema_slow: int = 21
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 1.5
    adx_period: int = 14
    adx_threshold: float = 20.0          # Min ADX for trending market

    # Trading controls
    max_trades_per_day: int = 50         # Daily trade limit
    max_concurrent_positions: int = 3    # Max open positions at once
    min_time_between_trades: int = 60    # Seconds between trades

    # Market hours (24/7 = no restriction)
    trading_enabled: bool = True
```

### 4.2 Trading Config (Alpaca)
```python
@dataclass
class CryptoTradingConfig:
    """Configuration for Alpaca crypto trading."""

    # API credentials (uses same as paper/live trading)
    api_key: str = ""
    api_secret: str = ""
    use_paper: bool = True  # Paper or live trading

    # Risk management (same pattern as live trading)
    max_daily_loss: float = 500.0        # Stop trading if daily loss exceeds
    max_daily_trades: int = 50
    max_position_size: float = 2000.0    # Per position
    max_total_exposure: float = 10000.0  # Total across all positions

    # Drawdown protection
    max_drawdown_pct: float = 5.0        # Pause trading if drawdown exceeds

    # Cooldown periods
    cooldown_after_loss: int = 300       # Seconds to wait after losing trade

    # Tracking
    last_trade_date: str = ""
    trades_today: int = 0
    pnl_today: float = 0.0
```

---

## 5. File Structure

```
trading_system/
├── strategies/
│   ├── coin_0dte_momentum.py      # Existing (options)
│   ├── crypto_scalping.py         # NEW: Crypto scalping strategy
│   └── __init__.py                # Update exports
│
├── indicators/
│   ├── moving_averages.py         # Existing (EMA, SMA)
│   ├── oscillators.py             # Existing (RSI, MACD)
│   ├── volatility.py              # Existing (BB, ATR)
│   ├── volume.py                  # NEW: VWAP, VolumeMA
│   ├── trend.py                   # NEW: ADX, Stochastic
│   └── __init__.py                # Update exports
│
├── engine/
│   ├── alpaca_client.py           # EXTEND: Add crypto methods
│   ├── crypto_trading_engine.py   # NEW: 24/7 crypto engine
│   └── __init__.py                # Update exports
│
├── config/
│   ├── crypto_trading_config.py   # NEW: Crypto config
│   ├── crypto_setup_wizard.py     # NEW: Interactive setup
│   └── __init__.py                # Update exports
│
├── run_crypto_trading.py          # NEW: Paper/Live trading runner
├── run_crypto_backtest.py         # NEW: Crypto backtest runner
└── __init__.py
```

---

## 6. Implementation Phases

### Phase 1: Core Indicators
1. Implement `VWAP` indicator
2. Implement `VolumeMA` indicator
3. Implement `ADX` indicator
4. Implement `Stochastic` oscillator
5. Update `__init__.py` exports

### Phase 2: Strategy Logic
1. Create `CryptoScalpingConfig` dataclass
2. Create `CryptoScalping` strategy class
3. Implement entry logic (BUY signals)
4. Implement exit logic (TP/SL/Time)
5. Add multi-symbol position management

### Phase 3: Alpaca Crypto Integration
1. Extend `AlpacaClient` for crypto endpoints:
   - `get_crypto_quote()`
   - `get_crypto_bars()`
   - `submit_crypto_order()`
2. Create `CryptoTradingEngine` for 24/7 operation
3. Create `CryptoTradingConfig` and setup wizard
4. Create `run_crypto_trading.py` runner

### Phase 4: Backtest Engine
1. Create `run_crypto_backtest.py`
2. Add crypto historical data fetching from Alpaca
3. Test with all 10 symbols
4. Generate performance reports

### Phase 5: Testing & Validation
1. **Quick Test**: November 2024 (1 month)
2. **Full Test**: September - November 2024 (3 months)
3. Optimize parameters based on results
4. Paper trading validation

---

## 7. Indicator Specifications

### 7.1 VWAP (Volume Weighted Average Price)
```python
@dataclass
class VWAP(Indicator):
    """
    Volume Weighted Average Price.

    Formula: VWAP = Cumulative(Price * Volume) / Cumulative(Volume)

    Resets: Daily at midnight UTC (configurable)
    """
    reset_period: str = "daily"  # "daily", "session", "never"

    def update(self, price: float, volume: float) -> None:
        self._cumulative_pv += price * volume
        self._cumulative_volume += volume
        self._value = self._cumulative_pv / self._cumulative_volume
```

### 7.2 Volume Moving Average
```python
@dataclass
class VolumeMA(Indicator):
    """
    Simple Moving Average of Volume.

    Used to detect volume spikes: current_volume > VolumeMA * multiplier
    """
    period: int = 20

    def is_spike(self, current_volume: float, multiplier: float = 1.5) -> bool:
        return current_volume > self.value * multiplier
```

### 7.3 ADX (Average Directional Index)
```python
@dataclass
class ADX(Indicator):
    """
    Average Directional Index - measures trend strength.

    Components:
    - +DI: Positive Directional Indicator
    - -DI: Negative Directional Indicator
    - ADX: Smoothed average of DX

    Interpretation:
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend / ranging
    """
    period: int = 14

    @property
    def plus_di(self) -> float: ...

    @property
    def minus_di(self) -> float: ...

    @property
    def is_trending(self) -> bool:
        return self.value > 25
```

### 7.4 Stochastic Oscillator
```python
@dataclass
class Stochastic(Indicator):
    """
    Stochastic Oscillator - momentum indicator.

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, d_period)

    Signals:
    - %K < 20: Oversold
    - %K > 80: Overbought
    - %K crosses above %D: Bullish
    - %K crosses below %D: Bearish
    """
    k_period: int = 14
    d_period: int = 3

    @property
    def k(self) -> float: ...

    @property
    def d(self) -> float: ...

    @property
    def is_oversold(self) -> bool:
        return self.k < 20

    @property
    def is_overbought(self) -> bool:
        return self.k > 80
```

---

## 8. Alpaca Crypto API Details

### REST Endpoints
```python
# Base URLs
TRADING_URL = "https://api.alpaca.markets"           # Live
PAPER_URL = "https://paper-api.alpaca.markets"       # Paper
CRYPTO_DATA_URL = "https://data.alpaca.markets/v1beta3/crypto/us"

# Get latest crypto quote
GET /v1beta3/crypto/us/latest/quotes?symbols=BTC/USD,ETH/USD

# Get crypto bars (OHLCV)
GET /v1beta3/crypto/us/bars?symbols=BTC/USD&timeframe=1Min&start=2024-11-01&end=2024-11-30

# Place crypto order
POST /v2/orders
{
    "symbol": "BTC/USD",
    "qty": "0.01",           # Fractional supported
    "side": "buy",           # "buy" or "sell"
    "type": "market",        # "market", "limit", "stop", "stop_limit"
    "time_in_force": "gtc"   # "gtc", "day", "ioc", "fok"
}

# Get crypto positions
GET /v2/positions

# Get account info
GET /v2/account
```

### WebSocket Streaming
```python
# Crypto data stream
wss://stream.data.alpaca.markets/v1beta3/crypto/us

# Subscribe message
{
    "action": "subscribe",
    "bars": ["BTC/USD", "ETH/USD"],
    "quotes": ["BTC/USD", "ETH/USD"],
    "trades": ["BTC/USD", "ETH/USD"]
}
```

### Timeframes Supported
| Timeframe | Code | Use Case |
|-----------|------|----------|
| 1 Minute | `1Min` | Scalping |
| 5 Minutes | `5Min` | Short-term |
| 15 Minutes | `15Min` | Medium-term |
| 1 Hour | `1Hour` | Trend analysis |
| 1 Day | `1Day` | Backtest |

---

## 9. Risk Controls Summary

| Control | Value | Description |
|---------|-------|-------------|
| Max Daily Loss | $500 | Stop trading for the day |
| Max Trades/Day | 50 | Prevent overtrading |
| Max Position Size | $2,000 | Per trade limit |
| Max Total Exposure | $10,000 | All positions combined |
| Max Concurrent Positions | 3 | Multi-symbol limit |
| Max Drawdown | 5% | Pause if account drops |
| Take Profit | 0.5-2% | Exit target |
| Stop Loss | 0.3-1% | Risk per trade |
| Trailing Stop | 0.3% | Lock in profits |
| Min Time Between | 60 sec | Prevent rapid fire |
| Cooldown After Loss | 5 min | Wait after losing trade |

---

## 10. Approval Checklist

Please review and approve:

- [ ] **Strategy Logic**: Buy low (RSI oversold + BB lower) / Sell high (TP or RSI overbought + BB upper)
- [ ] **Indicators**: RSI, BB, EMA, VWAP, VolumeMA, ADX, Stochastic (4 new + existing)
- [ ] **Exchange**: Alpaca Spot only (futures = separate strategy later)
- [ ] **Symbols**: Top 10 Alpaca crypto (BTC, ETH, DOGE, LINK, AVAX, DOT, LTC, UNI, SHIB, BCH)
- [ ] **Risk Controls**: Daily limits, position limits, drawdown protection, cooldowns
- [ ] **Configuration**: Flexible parameters for all thresholds
- [ ] **Backtest Plan**: Quick = November 2024 / Full = Sept-Nov 2024 (3 months)

---

## 11. Backtest Commands (After Implementation)

```bash
# Quick backtest - November 2024 only (BTC)
python -m trading_system.run_crypto_backtest --symbols BTC/USD --start 2024-11-01 --end 2024-11-30 --capital 10000

# Quick backtest - All symbols, November 2024
python -m trading_system.run_crypto_backtest --symbols ALL --start 2024-11-01 --end 2024-11-30 --capital 10000

# Full backtest - 3 months (Sept-Nov 2024)
python -m trading_system.run_crypto_backtest --symbols ALL --start 2024-09-01 --end 2024-11-30 --capital 10000

# Output results to file
python -m trading_system.run_crypto_backtest --symbols BTC/USD,ETH/USD --start 2024-11-01 --end 2024-11-30 --capital 10000 --output crypto_backtest_nov2024.json
```

---

*Plan Version: 2.0 (Alpaca Spot Only)*
*Created: December 2024*
*Status: AWAITING APPROVAL*
