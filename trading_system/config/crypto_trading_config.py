"""
Crypto Trading Configuration Module

Handles configuration for crypto spot trading including:
- API credentials (Alpaca)
- Position sizing
- Risk controls
- Symbol selection

Supports both paper trading and LIVE trading modes.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from pathlib import Path


# Configuration file paths
CONFIG_DIR = Path.home() / ".thevolumeai"
CRYPTO_CONFIG_FILE = CONFIG_DIR / "crypto_trading_config.json"
CRYPTO_LIVE_CONFIG_FILE = CONFIG_DIR / "crypto_live_trading_config.json"

# Supported crypto symbols on Alpaca (V6: 5 best performers)
ALPACA_CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"
]


@dataclass
class CryptoTradingConfig:
    """Configuration for crypto trading."""

    # Alpaca API credentials
    api_key: str = ""
    api_secret: str = ""

    # Use paper trading endpoint
    use_paper: bool = True

    # Trading parameters
    fixed_position_value: float = 500.0   # Fixed $ amount per trade
    max_position_value: float = 2000.0    # Max position per symbol

    # Selected symbols to trade
    symbols: List[str] = field(default_factory=lambda: ALPACA_CRYPTO_SYMBOLS.copy())

    # Risk parameters - V6: Optimized for crypto volatility
    target_profit_pct: float = 1.5        # V6: Take profit % (net ~1.0% after fees)
    stop_loss_pct: float = 1.0            # V6: Stop loss % (wider for crypto volatility)
    trailing_stop_pct: float = 0.5        # V6: Trailing stop %
    use_trailing_stop: bool = True

    # Risk controls
    max_daily_loss: float = 500.0         # Daily loss limit
    max_trades_per_day: int = 50          # Max trades per day
    max_concurrent_positions: int = 3     # Max concurrent positions
    min_time_between_trades: int = 60     # Seconds between trades per symbol

    # Indicator parameters - V6: Widened RSI for trend filter
    rsi_period: int = 14
    rsi_oversold: float = 35.0            # V6: Widened from 30 to 35
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std_dev: float = 2.0
    vwap_period: int = 50
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 1.5
    adx_period: int = 14
    adx_trend_threshold: float = 20.0

    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.api_key and self.api_secret)

    def save(self) -> None:
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CRYPTO_CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nConfiguration saved to: {CRYPTO_CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'CryptoTradingConfig':
        """Load configuration from file."""
        if CRYPTO_CONFIG_FILE.exists():
            try:
                with open(CRYPTO_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load crypto config file: {e}")
                return cls()
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Check if configuration file exists."""
        return CRYPTO_CONFIG_FILE.exists()


def mask_api_key(key: str) -> str:
    """Mask API key for display (show first 4 and last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def run_crypto_setup_wizard() -> CryptoTradingConfig:
    """
    Interactive setup wizard for crypto trading configuration.

    Returns:
        Configured CryptoTradingConfig object
    """
    print("\n" + "=" * 60)
    print("     CRYPTO SCALPING STRATEGY - SETUP WIZARD")
    print("=" * 60)

    config = CryptoTradingConfig()

    # Load existing config if available
    if CryptoTradingConfig.exists():
        existing = CryptoTradingConfig.load()
        if existing.is_configured():
            print(f"\nExisting configuration found:")
            print(f"  API Key: {mask_api_key(existing.api_key)}")
            print(f"  Paper Trading: {'Yes' if existing.use_paper else 'No'}")
            print(f"  Symbols: {len(existing.symbols)}")

            use_existing = input("\nUse existing config? [Y/n]: ").strip().lower()
            if use_existing != 'n':
                return existing

    # API Credentials
    print("\n--- ALPACA API CREDENTIALS ---")
    print("Get your API keys from: https://app.alpaca.markets/")

    config.api_key = input("Enter Alpaca API Key: ").strip()
    config.api_secret = input("Enter Alpaca Secret Key: ").strip()

    # Paper or Live
    print("\n--- TRADING MODE ---")
    paper_mode = input("Use Paper Trading? [Y/n]: ").strip().lower()
    config.use_paper = paper_mode != 'n'

    # Symbol Selection
    print("\n--- SYMBOL SELECTION ---")
    print("Available crypto symbols on Alpaca:")
    for i, symbol in enumerate(ALPACA_CRYPTO_SYMBOLS, 1):
        print(f"  {i}. {symbol}")

    print(f"\nDefault: All {len(ALPACA_CRYPTO_SYMBOLS)} symbols")
    custom_symbols = input("Enter symbol numbers (e.g., 1,2,3) or press Enter for all: ").strip()

    if custom_symbols:
        try:
            indices = [int(x.strip()) - 1 for x in custom_symbols.split(',')]
            config.symbols = [ALPACA_CRYPTO_SYMBOLS[i] for i in indices if 0 <= i < len(ALPACA_CRYPTO_SYMBOLS)]
        except (ValueError, IndexError):
            print("Invalid input, using all symbols")
            config.symbols = ALPACA_CRYPTO_SYMBOLS.copy()
    else:
        config.symbols = ALPACA_CRYPTO_SYMBOLS.copy()

    print(f"Selected symbols: {', '.join(config.symbols)}")

    # Position Sizing
    print("\n--- POSITION SIZING ---")
    try:
        pos_size = input(f"Fixed position size per trade (default ${config.fixed_position_value}): ").strip()
        if pos_size:
            config.fixed_position_value = float(pos_size)

        max_pos = input(f"Max position per symbol (default ${config.max_position_value}): ").strip()
        if max_pos:
            config.max_position_value = float(max_pos)
    except ValueError:
        print("Invalid input, using defaults")

    # Risk Parameters
    print("\n--- RISK PARAMETERS ---")
    try:
        tp = input(f"Take profit % (default {config.target_profit_pct}%): ").strip()
        if tp:
            config.target_profit_pct = float(tp)

        sl = input(f"Stop loss % (default {config.stop_loss_pct}%): ").strip()
        if sl:
            config.stop_loss_pct = float(sl)

        ts = input(f"Trailing stop % (default {config.trailing_stop_pct}%): ").strip()
        if ts:
            config.trailing_stop_pct = float(ts)

        use_ts = input("Enable trailing stop? [Y/n]: ").strip().lower()
        config.use_trailing_stop = use_ts != 'n'
    except ValueError:
        print("Invalid input, using defaults")

    # Risk Controls
    print("\n--- RISK CONTROLS ---")
    try:
        daily_loss = input(f"Max daily loss limit (default ${config.max_daily_loss}): ").strip()
        if daily_loss:
            config.max_daily_loss = float(daily_loss)

        max_trades = input(f"Max trades per day (default {config.max_trades_per_day}): ").strip()
        if max_trades:
            config.max_trades_per_day = int(max_trades)

        max_pos_count = input(f"Max concurrent positions (default {config.max_concurrent_positions}): ").strip()
        if max_pos_count:
            config.max_concurrent_positions = int(max_pos_count)
    except ValueError:
        print("Invalid input, using defaults")

    # Summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"API Key: {mask_api_key(config.api_key)}")
    print(f"Paper Trading: {'Yes' if config.use_paper else 'No (LIVE TRADING!)'}")
    print(f"Symbols: {', '.join(config.symbols)}")
    print(f"Position Size: ${config.fixed_position_value}")
    print(f"Max Position: ${config.max_position_value}")
    print(f"Take Profit: {config.target_profit_pct}%")
    print(f"Stop Loss: {config.stop_loss_pct}%")
    print(f"Trailing Stop: {config.trailing_stop_pct}% ({'enabled' if config.use_trailing_stop else 'disabled'})")
    print(f"Max Daily Loss: ${config.max_daily_loss}")
    print(f"Max Trades/Day: {config.max_trades_per_day}")
    print(f"Max Concurrent: {config.max_concurrent_positions}")

    # Confirm
    confirm = input("\nSave this configuration? [Y/n]: ").strip().lower()
    if confirm != 'n':
        config.save()
        print("\nConfiguration saved successfully!")
    else:
        print("\nConfiguration not saved.")

    return config


@dataclass
class CryptoLiveTradingConfig:
    """
    Configuration for LIVE crypto trading.

    WARNING: This configuration is for REAL MONEY trading.
    All trades will be executed with real funds.
    """

    # Alpaca API credentials (LIVE account)
    api_key: str = ""
    api_secret: str = ""

    # ALWAYS False for live trading (do not change)
    use_paper: bool = False

    # Trading parameters - CONSERVATIVE defaults for live trading
    fixed_position_value: float = 0.0   # User MUST set this explicitly
    max_position_value: float = 500.0   # Conservative max per symbol

    # Selected symbols to trade
    symbols: List[str] = field(default_factory=lambda: ALPACA_CRYPTO_SYMBOLS.copy())

    # Risk parameters - V6: Optimized for crypto volatility
    target_profit_pct: float = 1.5        # V6: Take profit % (net ~1.0% after fees)
    stop_loss_pct: float = 1.0            # V6: Stop loss % (wider for crypto volatility)
    trailing_stop_pct: float = 0.5        # V6: Trailing stop %
    use_trailing_stop: bool = True

    # Strict risk controls for live trading
    max_daily_loss: float = 100.0         # Daily loss limit (conservative)
    max_trades_per_day: int = 20          # Max trades per day (conservative)
    max_concurrent_positions: int = 3     # Max concurrent positions
    min_time_between_trades: int = 120    # Longer cooldown for live

    # Trading hours (UTC) - Peak hours only for live trading
    use_time_filter: bool = True
    allowed_trading_hours: List[int] = field(
        default_factory=lambda: list(range(13, 21))  # US trading hours only
    )

    # Hold time limits
    max_hold_minutes: int = 60
    min_hold_minutes: int = 5

    # Entry requirements - stricter for live
    min_entry_score: int = 7  # Higher threshold for live

    # Alpaca crypto fee (0.25% taker)
    taker_fee_pct: float = 0.25

    def is_configured(self) -> bool:
        """Check if API credentials and position size are configured."""
        return bool(self.api_key and self.api_secret and self.fixed_position_value > 0)

    def save(self) -> None:
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CRYPTO_LIVE_CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nLIVE configuration saved to: {CRYPTO_LIVE_CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'CryptoLiveTradingConfig':
        """Load configuration from file."""
        if CRYPTO_LIVE_CONFIG_FILE.exists():
            try:
                with open(CRYPTO_LIVE_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                # Ensure use_paper is always False for live config
                data['use_paper'] = False
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load live crypto config file: {e}")
                return cls()
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Check if configuration file exists."""
        return CRYPTO_LIVE_CONFIG_FILE.exists()

    def get_active_symbols(self) -> List[str]:
        """Get list of symbols to actively trade."""
        return self.symbols if self.symbols else ALPACA_CRYPTO_SYMBOLS


def run_crypto_reconfigure() -> CryptoTradingConfig:
    """
    Quick reconfigure specific settings.

    Returns:
        Updated CryptoTradingConfig object
    """
    if not CryptoTradingConfig.exists():
        print("No existing configuration found. Running full setup...")
        return run_crypto_setup_wizard()

    config = CryptoTradingConfig.load()

    print("\n" + "=" * 60)
    print("     CRYPTO TRADING - QUICK RECONFIGURE")
    print("=" * 60)
    print("\nWhat would you like to change?")
    print("1. Position sizing")
    print("2. Risk parameters (TP/SL)")
    print("3. Risk controls (daily limits)")
    print("4. Symbol selection")
    print("5. API credentials")
    print("6. Cancel")

    choice = input("\nEnter choice (1-6): ").strip()

    try:
        if choice == '1':
            pos_size = input(f"Fixed position size (current: ${config.fixed_position_value}): ").strip()
            if pos_size:
                config.fixed_position_value = float(pos_size)
            max_pos = input(f"Max position per symbol (current: ${config.max_position_value}): ").strip()
            if max_pos:
                config.max_position_value = float(max_pos)

        elif choice == '2':
            tp = input(f"Take profit % (current: {config.target_profit_pct}%): ").strip()
            if tp:
                config.target_profit_pct = float(tp)
            sl = input(f"Stop loss % (current: {config.stop_loss_pct}%): ").strip()
            if sl:
                config.stop_loss_pct = float(sl)
            ts = input(f"Trailing stop % (current: {config.trailing_stop_pct}%): ").strip()
            if ts:
                config.trailing_stop_pct = float(ts)

        elif choice == '3':
            daily_loss = input(f"Max daily loss (current: ${config.max_daily_loss}): ").strip()
            if daily_loss:
                config.max_daily_loss = float(daily_loss)
            max_trades = input(f"Max trades/day (current: {config.max_trades_per_day}): ").strip()
            if max_trades:
                config.max_trades_per_day = int(max_trades)

        elif choice == '4':
            print("\nAvailable symbols:")
            for i, symbol in enumerate(ALPACA_CRYPTO_SYMBOLS, 1):
                status = "[x]" if symbol in config.symbols else "[ ]"
                print(f"  {i}. {status} {symbol}")
            custom = input("\nEnter symbol numbers (e.g., 1,2,3): ").strip()
            if custom:
                indices = [int(x.strip()) - 1 for x in custom.split(',')]
                config.symbols = [ALPACA_CRYPTO_SYMBOLS[i] for i in indices if 0 <= i < len(ALPACA_CRYPTO_SYMBOLS)]

        elif choice == '5':
            config.api_key = input("Enter new API Key: ").strip()
            config.api_secret = input("Enter new Secret Key: ").strip()

        elif choice == '6':
            print("Cancelled.")
            return config

        config.save()
        print("\nConfiguration updated!")

    except ValueError:
        print("Invalid input, configuration not changed.")

    return config
