# THE VOLUME AI - Trading System Instructions

## Overview
This trading system implements a COIN 0DTE (Zero Days to Expiration) Momentum Strategy for options trading. It includes backtesting, paper trading, and LIVE trading with real money via Alpaca.

---

## Table of Contents
1. [Installation](#installation)
2. [Backtesting Commands](#backtesting-commands)
3. [Paper Trading Commands](#paper-trading-commands)
4. [Live Trading Commands](#live-trading-commands)
5. [Configuration](#configuration)
6. [Strategy Parameters](#strategy-parameters)
7. [File Structure](#file-structure)

---

## Installation

### Prerequisites
```bash
# Required packages
pip install pandas numpy pytz yfinance requests scipy python-dotenv

# For paper trading (Alpaca)
pip install alpaca-py
```

### Verify Installation
```bash
python -c "from trading_system.config import PaperTradingConfig; print('OK')"
```

---

## Backtesting Commands

### Run Basic Backtest
```bash
python -m trading_system.run_backtest --symbol COIN --start 2024-11-01 --end 2024-11-30 --capital 100000
```

### Run Backtest with Output File
```bash
# JSON output
python -m trading_system.run_backtest --symbol COIN --start 2024-11-01 --end 2024-11-30 --capital 100000 --output backtest_results.json

# Text report is also generated automatically
```

### Backtest Parameters
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--symbol` | Underlying stock symbol | `COIN` |
| `--start` | Start date (YYYY-MM-DD) | `2024-11-01` |
| `--end` | End date (YYYY-MM-DD) | `2024-11-30` |
| `--capital` | Initial capital in dollars | `100000` |
| `--output` | Output file path (optional) | `results.json` |

### Example Backtest Commands
```bash
# November 2024 backtest
python -m trading_system.run_backtest --symbol COIN --start 2024-11-01 --end 2024-11-30 --capital 100000 --output backtest_nov2024.json

# October 2024 backtest
python -m trading_system.run_backtest --symbol COIN --start 2024-10-01 --end 2024-10-31 --capital 100000 --output backtest_oct2024.json

# Full Q4 2024 backtest
python -m trading_system.run_backtest --symbol COIN --start 2024-10-01 --end 2024-12-31 --capital 100000 --output backtest_q4_2024.json
```

---

## Paper Trading Commands

### First-Time Setup
When running for the first time, the system will prompt you to enter:
1. Alpaca API Key
2. Alpaca Secret Key
3. Fixed position size per trade
4. Take profit / Stop loss percentages
5. Strategy selection
6. Trading hours

```bash
python -m trading_system.run_paper_trading
```

### Start Paper Trading
```bash
python -m trading_system.run_paper_trading
```

### Test API Connection Only
```bash
python -m trading_system.run_paper_trading --test
```

### Force Setup Wizard (Reconfigure Everything)
```bash
python -m trading_system.run_paper_trading --setup
```

### Quick Reconfigure (Change Specific Settings)
```bash
python -m trading_system.run_paper_trading --reconfigure
```

### Show Help
```bash
python -m trading_system.run_paper_trading --help
```

---

## Live Trading Commands

### !!! WARNING: LIVE TRADING USES REAL MONEY !!!

Before using live trading:
1. Test thoroughly with paper trading first
2. Understand options trading risks
3. Only use funds you can afford to lose
4. Review and set appropriate risk limits

### First-Time Setup (LIVE)
```bash
python -m trading_system.run_live_trading --setup
```

The setup wizard will prompt for:
1. Alpaca LIVE API Key (different from paper!)
2. Alpaca LIVE Secret Key
3. Position size per trade
4. Risk limits (daily loss limit, max trades/day)
5. Take profit / Stop loss percentages
6. Strategy selection
7. Trade confirmation preference

### Start Live Trading
```bash
python -m trading_system.run_live_trading
```

### Test LIVE API Connection Only
```bash
python -m trading_system.run_live_trading --test
```

### View Trade Log History
```bash
python -m trading_system.run_live_trading --log
```

### Force Setup Wizard (Reconfigure)
```bash
python -m trading_system.run_live_trading --setup
```

### Quick Reconfigure (Change Specific Settings)
```bash
python -m trading_system.run_live_trading --reconfigure
```

### Show Help
```bash
python -m trading_system.run_live_trading --help
```

### Live Trading Safety Features

| Feature | Description |
|---------|-------------|
| Daily Loss Limit | Stops trading if daily loss exceeds limit (default: $500) |
| Max Trades/Day | Uses strategy's `max_trades_per_day` setting (COIN 0DTE = 1) |
| Max Position Value | Caps single position size (default: $2,000) |
| Trade Confirmation | Optional confirmation before each trade |
| Trade Logging | All trades logged to persistent file |
| Multiple Confirmations | Must type 'I UNDERSTAND' and 'START' to begin |

### Live Trading Configuration

Configuration file: `C:\Users\Jean-Yves\.thevolumeai\live_trading_config.json`
Trade log file: `C:\Users\Jean-Yves\.thevolumeai\live_trade_log.json`

```json
{
  "api_key": "YOUR_LIVE_API_KEY",
  "api_secret": "***",
  "use_paper": false,
  "fixed_position_value": 500.0,
  "max_daily_loss": 500.0,
  "max_trades_per_day": 0,
  "max_position_value": 2000.0,
  "require_confirmation": true,
  "target_profit_pct": 7.5,
  "stop_loss_pct": 25.0,
  "max_hold_minutes": 30
}
```

**Note:** `max_trades_per_day: 0` means use the strategy's setting (recommended). The COIN 0DTE strategy has `max_trades_per_day: 1` since it strictly trades once per day.

---

## Configuration

### Configuration File Location
```
C:\Users\Jean-Yves\.thevolumeai\paper_trading_config.json
```

### Current Configuration
```json
{
  "api_key": "PK5KZ6L5XCSCA7BBV291",
  "api_secret": "***",
  "use_paper": true,
  "fixed_position_value": 2000.0,
  "strategy_file": "coin_0dte_momentum.py",
  "underlying_symbol": "COIN",
  "entry_time_start": "09:30:00",
  "entry_time_end": "10:00:00",
  "force_exit_time": "15:55:00",
  "target_profit_pct": 7.5,
  "stop_loss_pct": 25.0,
  "max_hold_minutes": 30
}
```

### Environment Variables (.env)
```bash
# Paper trading credentials
ALPACA_PAPER_KEY=PK5KZ6L5XCSCA7BBV291
ALPACA_PAPER_SECRET=BZ2HRkmbCqGeweLlpFTE3IfZuolr08B93Yuz09kL
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
```

---

## Strategy Parameters

### COIN 0DTE Momentum Strategy

| Parameter | Value | Description |
|-----------|-------|-------------|
| `underlying_symbol` | COIN | Stock to trade options on |
| `fixed_position_value` | $2,000 | Dollar amount per trade |
| `target_profit_pct` | 7.5% | Take profit target |
| `stop_loss_pct` | 25.0% | Stop loss threshold |
| `max_hold_minutes` | 30 | Maximum hold time before exit |
| `entry_time_start` | 09:30:00 EST | Entry window opens |
| `entry_time_end` | 10:00:00 EST | Entry window closes |
| `force_exit_time` | 15:55:00 EST | Force exit before 4PM expiry |

### Weekly Expiry Logic
- **Monday**: Options expire in 4 days (4 DTE)
- **Tuesday**: Options expire in 3 days (3 DTE)
- **Wednesday**: Options expire in 2 days (2 DTE)
- **Thursday**: Options expire in 1 day (1 DTE)
- **Friday**: Options expire TODAY at 4PM (0 DTE)

### ATM Strike Selection
- Only trades At-The-Money (ATM) options
- Strike must be within 2.5% of underlying price

---

## File Structure

```
thevolumeainative/
├── trading_system/
│   ├── __init__.py
│   ├── run_backtest.py           # Backtest runner
│   ├── run_paper_trading.py      # Paper trading runner
│   ├── run_live_trading.py       # LIVE trading runner (REAL MONEY)
│   ├── config/
│   │   ├── __init__.py
│   │   ├── paper_trading_config.py
│   │   ├── live_trading_config.py
│   │   ├── setup_wizard.py
│   │   └── live_setup_wizard.py
│   ├── core/
│   │   ├── events.py
│   │   └── models.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── alpaca_client.py      # Alpaca API client
│   │   ├── backtest_engine.py
│   │   ├── order_manager.py
│   │   ├── paper_trading_engine.py
│   │   └── live_trading_engine.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── coin_0dte_momentum.py # Main strategy
│   ├── indicators/
│   │   ├── moving_averages.py
│   │   ├── oscillators.py
│   │   └── volatility.py
│   └── analytics/
│       └── performance.py
├── backtest_results_v6.txt       # Latest backtest report
├── BACKTEST_FIXES_DOCUMENTATION.md
├── TRADING_SYSTEM_INSTRUCTIONS.md
└── .env                          # API credentials

~/.thevolumeai/                   # User config directory
├── paper_trading_config.json     # Paper trading settings
├── live_trading_config.json      # LIVE trading settings
└── live_trade_log.json           # LIVE trade history log
```

---

## Quick Reference

### Daily Workflow (Market Days)

1. **Before Market Open (before 9:30 AM EST)**
   ```bash
   # Test connection
   python -m trading_system.run_paper_trading --test
   ```

2. **Start Trading (at 9:30 AM EST)**
   ```bash
   python -m trading_system.run_paper_trading
   ```

3. **Monitor**
   - System will show real-time status
   - Press Ctrl+C to stop gracefully

### Weekend/After Hours

1. **Run Backtests**
   ```bash
   python -m trading_system.run_backtest --symbol COIN --start 2024-11-01 --end 2024-11-30 --capital 100000
   ```

2. **Review Results**
   - Check `backtest_results_*.txt` for detailed reports
   - Check `backtest_results_*.json` for raw data

---

## Troubleshooting

### Common Issues

**1. Import Error: alpaca-py not found**
```bash
pip install alpaca-py
```

**2. API Connection Failed**
- Check API key and secret are correct
- Verify you're using Paper Trading keys (not Live)
- Run: `python -m trading_system.run_paper_trading --test`

**3. No Trades Executing**
- Check if within entry window (9:30-10:00 AM EST)
- Verify market is open (weekdays only)
- Check for existing positions

**4. Reset Configuration**
```bash
# Delete config file and re-run setup
del C:\Users\Jean-Yves\.thevolumeai\paper_trading_config.json
python -m trading_system.run_paper_trading --setup
```

---

## Backtest Results Summary (November 2024)

| Metric | Value |
|--------|-------|
| Total Trades | 20 |
| Win Rate | 90.0% |
| Profit Factor | 9.10 |
| Total P&L | $4,771.15 |
| Total Return | +4.77% |
| Max Drawdown | 2.26% |
| Avg Hold Time | 2.3 minutes |
| CALL Trades | 13 (92.3% win rate) |
| PUT Trades | 7 (85.7% win rate) |

---

## Support

For issues or questions, refer to:
- `BACKTEST_FIXES_DOCUMENTATION.md` - Technical fixes documentation
- Strategy code: `trading_system/strategies/coin_0dte_momentum.py`

---

*Last Updated: December 2024*
