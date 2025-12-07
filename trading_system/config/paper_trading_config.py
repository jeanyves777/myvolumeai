"""
Paper Trading Configuration Module

Handles configuration for paper trading including:
- API credentials (Alpaca)
- Position sizing
- Strategy selection
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


# Configuration file path
CONFIG_DIR = Path.home() / ".thevolumeai"
CONFIG_FILE = CONFIG_DIR / "paper_trading_config.json"


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""

    # Alpaca API credentials
    api_key: str = ""
    api_secret: str = ""

    # Use paper trading endpoint (always True for paper trading)
    use_paper: bool = True

    # Trading parameters
    fixed_position_value: float = 2000.0  # Fixed $ amount per trade

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

    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.api_key and self.api_secret)

    def save(self) -> None:
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nConfiguration saved to: {CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'PaperTradingConfig':
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


def get_available_strategies() -> list[dict]:
    """
    Discover available strategy files in the strategies directory.

    Returns:
        List of dicts with 'name' and 'path' keys
    """
    strategies_dir = Path(__file__).parent.parent / "strategies"
    strategies = []

    for py_file in strategies_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        # Try to extract strategy class name from file
        strategy_name = py_file.stem.replace("_", " ").title()
        strategies.append({
            "name": strategy_name,
            "file": py_file.name,
            "path": str(py_file)
        })

    return strategies


def mask_api_key(key: str) -> str:
    """Mask API key for display (show first 4 and last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]
