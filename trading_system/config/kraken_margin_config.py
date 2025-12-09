"""
Kraken Futures Margin Trading Configuration Module

Handles configuration for ETH margin trading on Kraken Futures:
- API credentials (Kraken Futures)
- Position sizing and leverage
- Risk parameters
- Demo vs Production environment
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from pathlib import Path


# Configuration file paths
CONFIG_DIR = Path.home() / ".thevolumeai"
KRAKEN_DEMO_CONFIG_FILE = CONFIG_DIR / "kraken_demo_config.json"
KRAKEN_LIVE_CONFIG_FILE = CONFIG_DIR / "kraken_live_config.json"

# Default symbol (ETH perpetual)
DEFAULT_SYMBOL = "PI_ETHUSD"


@dataclass
class KrakenDemoConfig:
    """
    Configuration for Kraken Futures DEMO (paper) trading.

    Uses demo-futures.kraken.com for paper trading.
    """

    # Kraken Futures API credentials (from demo-futures.kraken.com)
    api_key: str = ""
    api_secret: str = ""

    # Always True for demo environment
    use_demo: bool = True

    # Symbol to trade
    symbol: str = DEFAULT_SYMBOL

    # Position sizing
    fixed_position_size: float = 0.0  # Fixed contracts per trade (user must set)
    position_value_usd: float = 500.0  # Alternative: USD value per trade
    use_notional: bool = True  # If True, use position_value_usd instead of fixed size

    # Leverage settings
    leverage: float = 5.0  # 5x leverage (conservative)
    max_leverage: float = 10.0  # Maximum allowed leverage

    # Risk parameters - V10 aligned with crypto spot strategy
    target_profit_pct: float = 1.5  # Take profit at 1.5%
    stop_loss_pct: float = 1.0  # Stop loss at 1.0%
    trailing_stop_pct: float = 0.5  # Trailing stop at 0.5%
    use_trailing_stop: bool = True

    # Trading hours (UTC) - Peak hours for ETH
    use_time_filter: bool = True
    allowed_trading_hours: List[int] = field(
        default_factory=lambda: list(range(0, 9)) + list(range(13, 22))
    )

    # Position management
    max_concurrent_positions: int = 1  # Only 1 ETH position at a time
    max_trades_per_hour: int = 5

    # Hold time limits
    max_hold_minutes: int = 60
    min_hold_minutes: int = 5

    # Entry requirements - V10 signal hierarchy
    min_entry_score: int = 7  # V10: Same as crypto spot

    # Fee structure (Kraken Futures)
    # Taker: 0.05%, Maker: 0.02%
    taker_fee_pct: float = 0.05
    maker_fee_pct: float = 0.02

    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        has_size = self.fixed_position_size > 0 or self.position_value_usd > 0
        return bool(self.api_key and self.api_secret and has_size)

    def save(self) -> None:
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(KRAKEN_DEMO_CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nDemo configuration saved to: {KRAKEN_DEMO_CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'KrakenDemoConfig':
        """Load configuration from file."""
        if KRAKEN_DEMO_CONFIG_FILE.exists():
            try:
                with open(KRAKEN_DEMO_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                # Ensure use_demo is always True for demo config
                data['use_demo'] = True
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load demo config file: {e}")
                return cls()
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Check if configuration file exists."""
        return KRAKEN_DEMO_CONFIG_FILE.exists()

    def get_position_size(self, current_price: float) -> float:
        """
        Calculate position size in contracts.

        Args:
            current_price: Current ETH price

        Returns:
            Position size in contracts (ETH)
        """
        if self.use_notional and self.position_value_usd > 0:
            # Calculate contracts from USD value
            return self.position_value_usd / current_price
        return self.fixed_position_size


@dataclass
class KrakenLiveConfig:
    """
    Configuration for Kraken Futures LIVE trading.

    WARNING: This is for REAL MONEY trading with MARGIN/LEVERAGE.
    Risk of liquidation exists.
    """

    # Kraken Futures API credentials (from futures.kraken.com)
    api_key: str = ""
    api_secret: str = ""

    # Always False for live environment
    use_demo: bool = False

    # Symbol to trade
    symbol: str = DEFAULT_SYMBOL

    # Position sizing - CONSERVATIVE defaults for live
    fixed_position_size: float = 0.0  # User MUST set this explicitly
    position_value_usd: float = 0.0  # User MUST set this explicitly
    use_notional: bool = True

    # Leverage settings - CONSERVATIVE for live
    leverage: float = 3.0  # 3x leverage (more conservative than demo)
    max_leverage: float = 5.0  # Maximum allowed leverage

    # Risk parameters - STRICTER for live
    target_profit_pct: float = 1.5
    stop_loss_pct: float = 0.75  # Tighter stop for live
    trailing_stop_pct: float = 0.4
    use_trailing_stop: bool = True

    # Trading hours (UTC) - Peak hours only for live
    use_time_filter: bool = True
    allowed_trading_hours: List[int] = field(
        default_factory=lambda: list(range(13, 21))  # US trading hours only
    )

    # Position management - STRICT for live
    max_concurrent_positions: int = 1
    max_trades_per_hour: int = 3  # Fewer trades for live

    # Hold time limits
    max_hold_minutes: int = 45  # Shorter holds for live
    min_hold_minutes: int = 5

    # Entry requirements - STRICTER for live
    min_entry_score: int = 8  # Higher threshold for live

    # Risk controls
    max_daily_loss_pct: float = 2.0  # Max 2% daily loss on account
    max_position_pct: float = 20.0  # Max 20% of account per position

    # Fee structure
    taker_fee_pct: float = 0.05
    maker_fee_pct: float = 0.02

    def is_configured(self) -> bool:
        """Check if API credentials and position size are configured."""
        has_size = self.fixed_position_size > 0 or self.position_value_usd > 0
        return bool(self.api_key and self.api_secret and has_size)

    def save(self) -> None:
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(KRAKEN_LIVE_CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nLIVE configuration saved to: {KRAKEN_LIVE_CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'KrakenLiveConfig':
        """Load configuration from file."""
        if KRAKEN_LIVE_CONFIG_FILE.exists():
            try:
                with open(KRAKEN_LIVE_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                # Ensure use_demo is always False for live config
                data['use_demo'] = False
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load live config file: {e}")
                return cls()
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Check if configuration file exists."""
        return KRAKEN_LIVE_CONFIG_FILE.exists()

    def get_position_size(self, current_price: float) -> float:
        """
        Calculate position size in contracts.

        Args:
            current_price: Current ETH price

        Returns:
            Position size in contracts (ETH)
        """
        if self.use_notional and self.position_value_usd > 0:
            return self.position_value_usd / current_price
        return self.fixed_position_size


def mask_api_key(key: str) -> str:
    """Mask API key for display (show first 4 and last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def run_kraken_demo_setup_wizard() -> KrakenDemoConfig:
    """
    Interactive setup wizard for Kraken demo trading.

    Returns:
        Configured KrakenDemoConfig object
    """
    print("\n" + "=" * 60)
    print("     KRAKEN FUTURES ETH MARGIN - DEMO SETUP")
    print("=" * 60)
    print("\nThis wizard will configure paper trading on Kraken Futures Demo.")
    print("Demo URL: https://demo-futures.kraken.com/")
    print("\nSign up at demo-futures.kraken.com (no email verification needed)")

    config = KrakenDemoConfig()

    # Load existing config if available
    if KrakenDemoConfig.exists():
        existing = KrakenDemoConfig.load()
        if existing.is_configured():
            print(f"\nExisting configuration found:")
            print(f"  API Key: {mask_api_key(existing.api_key)}")
            print(f"  Position Value: ${existing.position_value_usd}")
            print(f"  Leverage: {existing.leverage}x")

            use_existing = input("\nUse existing config? [Y/n]: ").strip().lower()
            if use_existing != 'n':
                return existing

    # API Credentials
    print("\n--- KRAKEN FUTURES API CREDENTIALS ---")
    print("Get your API keys from: https://demo-futures.kraken.com/")
    print("Go to: Settings > API > Create API Key")

    config.api_key = input("Enter Kraken Futures API Key: ").strip()
    config.api_secret = input("Enter Kraken Futures API Secret: ").strip()

    # Position Sizing
    print("\n--- POSITION SIZING ---")
    try:
        pos_value = input(f"Position value in USD (default ${config.position_value_usd}): ").strip()
        if pos_value:
            config.position_value_usd = float(pos_value)
    except ValueError:
        print("Invalid input, using default")

    # Leverage
    print("\n--- LEVERAGE ---")
    print("Leverage multiplies gains AND losses. Higher = more risk.")
    try:
        lev = input(f"Leverage multiplier (default {config.leverage}x, max {config.max_leverage}x): ").strip()
        if lev:
            config.leverage = min(float(lev), config.max_leverage)
    except ValueError:
        print("Invalid input, using default")

    # Risk Parameters
    print("\n--- RISK PARAMETERS ---")
    try:
        tp = input(f"Take profit % (default {config.target_profit_pct}%): ").strip()
        if tp:
            config.target_profit_pct = float(tp)

        sl = input(f"Stop loss % (default {config.stop_loss_pct}%): ").strip()
        if sl:
            config.stop_loss_pct = float(sl)
    except ValueError:
        print("Invalid input, using defaults")

    # Summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Environment: DEMO (Paper Trading)")
    print(f"API Key: {mask_api_key(config.api_key)}")
    print(f"Symbol: {config.symbol}")
    print(f"Position Value: ${config.position_value_usd}")
    print(f"Leverage: {config.leverage}x")
    print(f"Take Profit: {config.target_profit_pct}%")
    print(f"Stop Loss: {config.stop_loss_pct}%")
    print(f"Trailing Stop: {config.trailing_stop_pct}% ({'enabled' if config.use_trailing_stop else 'disabled'})")

    # Confirm
    confirm = input("\nSave this configuration? [Y/n]: ").strip().lower()
    if confirm != 'n':
        config.save()
        print("\nConfiguration saved successfully!")
    else:
        print("\nConfiguration not saved.")

    return config


def run_kraken_live_setup_wizard() -> KrakenLiveConfig:
    """
    Interactive setup wizard for Kraken LIVE trading.

    WARNING: This is for REAL MONEY trading.

    Returns:
        Configured KrakenLiveConfig object
    """
    print("\n" + "=" * 60)
    print("     KRAKEN FUTURES ETH MARGIN - LIVE SETUP")
    print("=" * 60)
    print("\n" + "!" * 60)
    print("WARNING: THIS IS FOR REAL MONEY TRADING WITH LEVERAGE")
    print("YOU CAN LOSE MORE THAN YOUR INITIAL INVESTMENT")
    print("!" * 60)

    confirm_warning = input("\nDo you understand the risks? Type 'YES' to continue: ").strip()
    if confirm_warning != 'YES':
        print("\nSetup cancelled. Use demo mode first to test the strategy.")
        return KrakenLiveConfig()

    config = KrakenLiveConfig()

    # Load existing config if available
    if KrakenLiveConfig.exists():
        existing = KrakenLiveConfig.load()
        if existing.is_configured():
            print(f"\nExisting configuration found:")
            print(f"  API Key: {mask_api_key(existing.api_key)}")
            print(f"  Position Value: ${existing.position_value_usd}")
            print(f"  Leverage: {existing.leverage}x")

            use_existing = input("\nUse existing config? [Y/n]: ").strip().lower()
            if use_existing != 'n':
                return existing

    # API Credentials
    print("\n--- KRAKEN FUTURES API CREDENTIALS ---")
    print("Get your API keys from: https://futures.kraken.com/")
    print("Go to: Settings > API > Create API Key")
    print("\nEnsure your API key has 'Full Access' for trading.")

    config.api_key = input("Enter Kraken Futures API Key: ").strip()
    config.api_secret = input("Enter Kraken Futures API Secret: ").strip()

    # Position Sizing
    print("\n--- POSITION SIZING ---")
    print("Start with SMALL positions until you're comfortable.")
    try:
        pos_value = input("Position value in USD (NO DEFAULT - you must set this): ").strip()
        if pos_value:
            config.position_value_usd = float(pos_value)
        else:
            print("ERROR: Position value is required for live trading.")
            return config
    except ValueError:
        print("Invalid input. Configuration not complete.")
        return config

    # Leverage
    print("\n--- LEVERAGE ---")
    print(f"Recommended: 2-3x for live trading. Max allowed: {config.max_leverage}x")
    try:
        lev = input(f"Leverage multiplier (default {config.leverage}x): ").strip()
        if lev:
            config.leverage = min(float(lev), config.max_leverage)
    except ValueError:
        print("Invalid input, using default")

    # Risk Parameters
    print("\n--- RISK PARAMETERS ---")
    try:
        tp = input(f"Take profit % (default {config.target_profit_pct}%): ").strip()
        if tp:
            config.target_profit_pct = float(tp)

        sl = input(f"Stop loss % (default {config.stop_loss_pct}%): ").strip()
        if sl:
            config.stop_loss_pct = float(sl)
    except ValueError:
        print("Invalid input, using defaults")

    # Summary
    print("\n" + "=" * 60)
    print("LIVE CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Environment: LIVE (REAL MONEY)")
    print(f"API Key: {mask_api_key(config.api_key)}")
    print(f"Symbol: {config.symbol}")
    print(f"Position Value: ${config.position_value_usd}")
    print(f"Leverage: {config.leverage}x")
    print(f"Effective Exposure: ${config.position_value_usd * config.leverage}")
    print(f"Take Profit: {config.target_profit_pct}%")
    print(f"Stop Loss: {config.stop_loss_pct}%")

    # Final confirmation
    print("\n" + "!" * 60)
    final_confirm = input("Type 'CONFIRM' to save LIVE configuration: ").strip()
    if final_confirm == 'CONFIRM':
        config.save()
        print("\nLIVE configuration saved!")
    else:
        print("\nConfiguration not saved.")

    return config
