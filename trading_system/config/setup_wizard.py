"""
First-Time Setup Wizard for Paper Trading

Guides users through:
1. Alpaca API credentials setup
2. Fixed position size per trade
3. Strategy file selection
"""

import sys
from pathlib import Path
from typing import Optional

from .paper_trading_config import (
    PaperTradingConfig,
    get_available_strategies,
    mask_api_key,
)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def get_input(prompt: str, default: Optional[str] = None, required: bool = True) -> str:
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    while True:
        value = input(prompt).strip()

        if not value and default:
            return default

        if not value and required:
            print("  This field is required. Please enter a value.")
            continue

        return value


def get_float_input(prompt: str, default: float) -> float:
    """Get float input from user."""
    while True:
        value = get_input(prompt, str(default))
        try:
            return float(value)
        except ValueError:
            print("  Please enter a valid number.")


def get_int_input(prompt: str, default: int) -> int:
    """Get integer input from user."""
    while True:
        value = get_input(prompt, str(default))
        try:
            return int(value)
        except ValueError:
            print("  Please enter a valid whole number.")


def select_from_list(items: list, prompt: str = "Select an option") -> int:
    """Display numbered list and get user selection."""
    print()
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            print(f"  {i}. {item.get('name', item.get('file', str(item)))}")
        else:
            print(f"  {i}. {item}")

    while True:
        try:
            choice = int(get_input(f"\n{prompt} (1-{len(items)})"))
            if 1 <= choice <= len(items):
                return choice - 1
            print(f"  Please enter a number between 1 and {len(items)}.")
        except ValueError:
            print("  Please enter a valid number.")


def run_setup_wizard(force_reconfigure: bool = False) -> PaperTradingConfig:
    """
    Run the first-time setup wizard.

    Args:
        force_reconfigure: If True, run wizard even if config exists

    Returns:
        Configured PaperTradingConfig instance
    """
    print_header("THE VOLUME AI - Paper Trading Setup Wizard")

    # Check for existing configuration
    config = PaperTradingConfig.load()

    if config.is_configured() and not force_reconfigure:
        print("\nExisting configuration found!")
        print(f"  API Key: {mask_api_key(config.api_key)}")
        print(f"  Position Size: ${config.fixed_position_value:,.2f}")
        print(f"  Strategy: {config.strategy_file or 'Not set'}")

        reconfigure = get_input(
            "\nWould you like to reconfigure? (y/n)",
            default="n"
        ).lower()

        if reconfigure != 'y':
            return config

    # Step 1: Alpaca API Credentials
    print_section("Step 1: Alpaca API Credentials")
    print("""
To trade with Alpaca, you need API credentials.
Get your keys from: https://app.alpaca.markets/

For PAPER trading, use your Paper Trading API keys.
""")

    config.api_key = get_input("Enter your Alpaca API Key")
    config.api_secret = get_input("Enter your Alpaca Secret Key")

    # Always use paper trading endpoint for paper trading
    config.use_paper = True
    print("\n  Using Alpaca PAPER trading endpoint.")

    # Step 2: Position Sizing
    print_section("Step 2: Position Sizing")
    print("""
Set the fixed dollar amount you want to risk per trade.
This is the total value of options contracts per position.
""")

    config.fixed_position_value = get_float_input(
        "Fixed position size per trade ($)",
        default=2000.0
    )

    # Step 3: Risk Parameters
    print_section("Step 3: Risk Parameters")
    print("""
Configure your take-profit and stop-loss percentages.
""")

    config.target_profit_pct = get_float_input(
        "Take profit percentage (%)",
        default=7.5
    )

    config.stop_loss_pct = get_float_input(
        "Stop loss percentage (%)",
        default=25.0
    )

    config.max_hold_minutes = get_int_input(
        "Maximum hold time (minutes)",
        default=30
    )

    # Step 4: Strategy Selection
    print_section("Step 4: Strategy Selection")
    print("""
Select the trading strategy to use.
""")

    strategies = get_available_strategies()

    if not strategies:
        print("  Warning: No strategy files found in strategies directory.")
        config.strategy_file = ""
    else:
        idx = select_from_list(strategies, "Select strategy")
        selected = strategies[idx]
        config.strategy_file = selected['file']
        print(f"\n  Selected: {selected['name']} ({selected['file']})")

    # Step 5: Underlying Symbol
    print_section("Step 5: Underlying Symbol")

    config.underlying_symbol = get_input(
        "Enter underlying symbol to trade",
        default="COIN"
    ).upper()

    # Step 6: Trading Hours
    print_section("Step 6: Trading Hours (EST)")
    print("""
Set the time window for entering trades and force exit time.
Format: HH:MM:SS (24-hour EST)
""")

    config.entry_time_start = get_input(
        "Entry window start",
        default="09:30:00"
    )

    config.entry_time_end = get_input(
        "Entry window end",
        default="10:00:00"
    )

    config.force_exit_time = get_input(
        "Force exit time (before 4PM expiry)",
        default="15:55:00"
    )

    # Summary and Confirmation
    print_header("Configuration Summary")
    print(f"""
  API Key:          {mask_api_key(config.api_key)}
  Trading Mode:     PAPER (Simulated)

  Position Size:    ${config.fixed_position_value:,.2f}
  Take Profit:      {config.target_profit_pct}%
  Stop Loss:        {config.stop_loss_pct}%
  Max Hold Time:    {config.max_hold_minutes} minutes

  Strategy:         {config.strategy_file}
  Symbol:           {config.underlying_symbol}

  Entry Window:     {config.entry_time_start} - {config.entry_time_end} EST
  Force Exit:       {config.force_exit_time} EST
""")

    confirm = get_input("Save this configuration? (y/n)", default="y").lower()

    if confirm == 'y':
        config.save()
        print("\nSetup complete! You can now run paper trading.")
    else:
        print("\nConfiguration not saved. Run setup again when ready.")

    return config


def quick_reconfigure() -> PaperTradingConfig:
    """Quick reconfiguration menu for specific settings."""
    print_header("Quick Reconfigure")

    config = PaperTradingConfig.load()

    options = [
        "API Credentials",
        "Position Size",
        "Risk Parameters (TP/SL)",
        "Strategy Selection",
        "Trading Hours",
        "Full Setup Wizard",
        "Cancel"
    ]

    idx = select_from_list(options, "What would you like to change?")

    if idx == 0:  # API Credentials
        config.api_key = get_input("Enter your Alpaca API Key")
        config.api_secret = get_input("Enter your Alpaca Secret Key")

    elif idx == 1:  # Position Size
        config.fixed_position_value = get_float_input(
            "Fixed position size per trade ($)",
            default=config.fixed_position_value
        )

    elif idx == 2:  # Risk Parameters
        config.target_profit_pct = get_float_input(
            "Take profit percentage (%)",
            default=config.target_profit_pct
        )
        config.stop_loss_pct = get_float_input(
            "Stop loss percentage (%)",
            default=config.stop_loss_pct
        )
        config.max_hold_minutes = get_int_input(
            "Maximum hold time (minutes)",
            default=config.max_hold_minutes
        )

    elif idx == 3:  # Strategy Selection
        strategies = get_available_strategies()
        if strategies:
            s_idx = select_from_list(strategies, "Select strategy")
            config.strategy_file = strategies[s_idx]['file']

    elif idx == 4:  # Trading Hours
        config.entry_time_start = get_input(
            "Entry window start",
            default=config.entry_time_start
        )
        config.entry_time_end = get_input(
            "Entry window end",
            default=config.entry_time_end
        )
        config.force_exit_time = get_input(
            "Force exit time",
            default=config.force_exit_time
        )

    elif idx == 5:  # Full Setup
        return run_setup_wizard(force_reconfigure=True)

    else:  # Cancel
        print("\nNo changes made.")
        return config

    # Save changes
    if idx < 5:
        config.save()
        print("\nConfiguration updated!")

    return config


if __name__ == "__main__":
    # Run setup wizard if executed directly
    run_setup_wizard()
