"""
Live Trading Configuration Module

Handles configuration for LIVE trading with REAL money.
Includes additional safety checks and confirmations.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
from datetime import datetime


# Configuration file path - SEPARATE from paper trading
CONFIG_DIR = Path.home() / ".thevolumeai"
CONFIG_FILE = CONFIG_DIR / "live_trading_config.json"
TRADE_LOG_FILE = CONFIG_DIR / "live_trade_log.json"


@dataclass
class LiveTradingConfig:
    """Configuration for LIVE trading with real money."""

    # Alpaca API credentials (LIVE - not paper!)
    api_key: str = ""
    api_secret: str = ""

    # MUST be False for live trading
    use_paper: bool = False

    # Trading parameters
    fixed_position_value: float = 500.0  # Lower default for live trading

    # Selected strategy file path
    strategy_file: str = ""

    # Underlying symbol
    underlying_symbol: str = "COIN"

    # Trading times (EST)
    entry_time_start: str = "09:30:00"
    entry_time_end: str = "10:00:00"
    force_exit_time: str = "15:55:00"

    # Risk parameters
    target_profit_pct: float = 7.5
    stop_loss_pct: float = 25.0
    max_hold_minutes: int = 30

    # SAFETY LIMITS for live trading
    max_daily_loss: float = 500.0  # Stop trading if daily loss exceeds this
    max_trades_per_day: int = 0  # 0 = use strategy's max_trades_per_day (recommended)
    max_position_value: float = 2000.0  # Maximum single position value
    require_confirmation: bool = True  # Require confirmation before each trade

    # Tracking
    last_trade_date: str = ""
    trades_today: int = 0
    pnl_today: float = 0.0

    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.api_key and self.api_secret)

    def can_trade(self, effective_max_trades: int = 0) -> tuple[bool, str]:
        """
        Check if trading is allowed based on safety limits.

        Args:
            effective_max_trades: The actual max trades per day (from strategy).
                                  If 0, uses config value.

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Reset daily counters if new day
        if self.last_trade_date != today:
            self.trades_today = 0
            self.pnl_today = 0.0
            self.last_trade_date = today

        # Check daily loss limit
        if self.pnl_today <= -self.max_daily_loss:
            return False, f"Daily loss limit reached (${self.max_daily_loss:.2f})"

        # Use effective_max_trades if provided, otherwise fall back to config
        max_trades = effective_max_trades if effective_max_trades > 0 else self.max_trades_per_day
        if max_trades > 0 and self.trades_today >= max_trades:
            return False, f"Max trades per day reached ({max_trades})"

        # Check position size limit
        if self.fixed_position_value > self.max_position_value:
            return False, f"Position value exceeds max (${self.max_position_value:.2f})"

        return True, "OK"

    def record_trade(self, pnl: float):
        """Record a completed trade."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_trade_date != today:
            self.trades_today = 0
            self.pnl_today = 0.0
            self.last_trade_date = today

        self.trades_today += 1
        self.pnl_today += pnl
        self.save()

    def save(self) -> None:
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nConfiguration saved to: {CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'LiveTradingConfig':
        """Load configuration from file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load config file: {e}")
                return cls()
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Check if configuration file exists."""
        return CONFIG_FILE.exists()


def log_trade(trade_data: dict) -> None:
    """Log trade to persistent trade log file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing log
    trades = []
    if TRADE_LOG_FILE.exists():
        try:
            with open(TRADE_LOG_FILE, 'r') as f:
                trades = json.load(f)
        except (json.JSONDecodeError, TypeError):
            trades = []

    # Add timestamp
    trade_data['logged_at'] = datetime.now().isoformat()
    trades.append(trade_data)

    # Save log
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def get_trade_log() -> list:
    """Get all logged trades."""
    if TRADE_LOG_FILE.exists():
        try:
            with open(TRADE_LOG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def mask_api_key(key: str) -> str:
    """Mask API key for display (show first 4 and last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]
