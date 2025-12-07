"""
Live Trading Setup Wizard

IMPORTANT: This is for REAL MONEY trading.
Includes multiple safety confirmations and warnings.
"""

import sys
from pathlib import Path
from typing import Optional

from .live_trading_config import (
    LiveTradingConfig,
    mask_api_key,
)
from .paper_trading_config import get_available_strategies


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_warning(msg: str) -> None:
    """Print a warning message."""
    print(f"\n  *** WARNING: {msg} ***\n")


def print_danger(msg: str) -> None:
    """Print a danger message."""
    print("\n" + "!" * 60)
    print(f"  !!! {msg} !!!")
    print("!" * 60 + "\n")


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


def get_confirmation(prompt: str) -> bool:
    """Get yes/no confirmation from user."""
    while True:
        response = input(f"{prompt} (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        print("  Please enter 'yes' or 'no'.")


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


def run_live_setup_wizard(force_reconfigure: bool = False) -> LiveTradingConfig:
    """
    Run the live trading setup wizard with safety confirmations.

    Args:
        force_reconfigure: If True, run wizard even if config exists

    Returns:
        Configured LiveTradingConfig instance
    """
    print_header("THE VOLUME AI - LIVE TRADING SETUP")

    print_danger("THIS IS FOR REAL MONEY TRADING")

    print("""
  You are about to configure LIVE trading with REAL MONEY.
  All trades will be executed with actual funds in your account.

  Please ensure you:
  1. Understand the risks of options trading
  2. Have tested the strategy thoroughly with paper trading
  3. Are using funds you can afford to lose
  4. Have set appropriate risk limits
""")

    # First confirmation
    if not get_confirmation("Do you understand this is REAL MONEY trading?"):
        print("\nSetup cancelled. Use paper trading to test first.")
        sys.exit(0)

    # Check for existing configuration
    config = LiveTradingConfig.load()

    if config.is_configured() and not force_reconfigure:
        print("\nExisting LIVE configuration found!")
        print(f"  API Key: {mask_api_key(config.api_key)}")
        print(f"  Position Size: ${config.fixed_position_value:,.2f}")
        print(f"  Max Daily Loss: ${config.max_daily_loss:,.2f}")
        print(f"  Strategy: {config.strategy_file or 'Not set'}")

        reconfigure = get_input(
            "\nWould you like to reconfigure? (y/n)",
            default="n"
        ).lower()

        if reconfigure != 'y':
            return config

    # Step 1: Alpaca LIVE API Credentials
    print_header("Step 1: Alpaca LIVE API Credentials")
    print_warning("Make sure you are using LIVE API keys, NOT paper trading keys!")
    print("""
  Get your LIVE API keys from: https://app.alpaca.markets/

  IMPORTANT: Live API keys are different from Paper Trading keys.
  Live keys will execute REAL trades with REAL money.
""")

    config.api_key = get_input("Enter your Alpaca LIVE API Key")
    config.api_secret = get_input("Enter your Alpaca LIVE Secret Key")
    config.use_paper = False  # MUST be False for live

    # Verify they understand this is live
    print_warning("You entered LIVE trading credentials")
    if not get_confirmation("Confirm these are your LIVE (not paper) API keys?"):
        print("\nSetup cancelled. Please verify your API keys.")
        sys.exit(0)

    # Step 2: Position Sizing (with lower defaults for safety)
    print_header("Step 2: Position Sizing")
    print("""
  Set the fixed dollar amount per trade.
  For live trading, we recommend starting with smaller amounts.
""")

    config.fixed_position_value = get_float_input(
        "Fixed position size per trade ($)",
        default=500.0  # Lower default for live
    )

    # Step 3: Risk Limits (IMPORTANT for live)
    print_header("Step 3: Risk Limits (VERY IMPORTANT)")
    print("""
  These limits protect you from excessive losses.
  The system will STOP trading when limits are hit.
""")

    config.max_daily_loss = get_float_input(
        "Maximum daily loss before stopping ($)",
        default=500.0
    )

    config.max_trades_per_day = get_int_input(
        "Maximum trades per day",
        default=3
    )

    config.max_position_value = get_float_input(
        "Maximum single position value ($)",
        default=2000.0
    )

    # Ensure position size doesn't exceed max
    if config.fixed_position_value > config.max_position_value:
        print_warning(f"Position size reduced to max allowed: ${config.max_position_value}")
        config.fixed_position_value = config.max_position_value

    # Step 4: Take Profit / Stop Loss
    print_header("Step 4: Exit Parameters")

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

    # Step 5: Strategy Selection
    print_header("Step 5: Strategy Selection")

    strategies = get_available_strategies()

    if not strategies:
        print("  Warning: No strategy files found in strategies directory.")
        config.strategy_file = ""
    else:
        idx = select_from_list(strategies, "Select strategy")
        selected = strategies[idx]
        config.strategy_file = selected['file']
        print(f"\n  Selected: {selected['name']} ({selected['file']})")

    # Step 6: Underlying Symbol
    print_header("Step 6: Underlying Symbol")

    config.underlying_symbol = get_input(
        "Enter underlying symbol to trade",
        default="COIN"
    ).upper()

    # Step 7: Trading Hours
    print_header("Step 7: Trading Hours (EST)")

    config.entry_time_start = get_input(
        "Entry window start",
        default="09:30:00"
    )

    config.entry_time_end = get_input(
        "Entry window end",
        default="10:00:00"
    )

    config.force_exit_time = get_input(
        "Force exit time",
        default="15:55:00"
    )

    # Step 8: Trade Confirmation Setting
    print_header("Step 8: Trade Confirmation")
    print("""
  If enabled, you will be asked to confirm each trade before execution.
  Recommended for live trading until you are comfortable.
""")

    config.require_confirmation = get_confirmation(
        "Require confirmation before each trade?"
    )

    # Final Summary
    print_header("LIVE TRADING CONFIGURATION SUMMARY")
    print_danger("REVIEW CAREFULLY - THIS IS REAL MONEY")

    print(f"""
  API Key:            {mask_api_key(config.api_key)}
  Trading Mode:       *** LIVE (REAL MONEY) ***

  Position Size:      ${config.fixed_position_value:,.2f}
  Max Position:       ${config.max_position_value:,.2f}
  Max Daily Loss:     ${config.max_daily_loss:,.2f}
  Max Trades/Day:     {config.max_trades_per_day}

  Take Profit:        {config.target_profit_pct}%
  Stop Loss:          {config.stop_loss_pct}%
  Max Hold Time:      {config.max_hold_minutes} minutes

  Strategy:           {config.strategy_file}
  Symbol:             {config.underlying_symbol}

  Entry Window:       {config.entry_time_start} - {config.entry_time_end} EST
  Force Exit:         {config.force_exit_time} EST

  Confirm Each Trade: {'Yes' if config.require_confirmation else 'No'}
""")

    # Final confirmation (must type 'CONFIRM' for live trading)
    print_warning("To save this LIVE trading configuration, type 'CONFIRM' below")

    confirm = get_input("Type CONFIRM to save (or anything else to cancel)")

    if confirm.upper() == 'CONFIRM':
        config.save()
        print("\n" + "=" * 60)
        print("  LIVE TRADING CONFIGURATION SAVED")
        print("=" * 60)
        print("\nYou can now run live trading with:")
        print("  python -m trading_system.run_live_trading")
    else:
        print("\nConfiguration NOT saved. Run setup again when ready.")

    return config


def quick_reconfigure_live() -> LiveTradingConfig:
    """Quick reconfiguration menu for live trading settings."""
    print_header("Quick Reconfigure - LIVE Trading")
    print_warning("You are modifying LIVE trading settings")

    config = LiveTradingConfig.load()

    options = [
        "API Credentials",
        "Position Size",
        "Risk Limits",
        "Exit Parameters (TP/SL)",
        "Strategy Selection",
        "Trading Hours",
        "Full Setup Wizard",
        "Cancel"
    ]

    idx = select_from_list(options, "What would you like to change?")

    if idx == 0:  # API Credentials
        config.api_key = get_input("Enter your Alpaca LIVE API Key")
        config.api_secret = get_input("Enter your Alpaca LIVE Secret Key")

    elif idx == 1:  # Position Size
        config.fixed_position_value = get_float_input(
            "Fixed position size per trade ($)",
            default=config.fixed_position_value
        )

    elif idx == 2:  # Risk Limits
        config.max_daily_loss = get_float_input(
            "Maximum daily loss ($)",
            default=config.max_daily_loss
        )
        config.max_trades_per_day = get_int_input(
            "Maximum trades per day",
            default=config.max_trades_per_day
        )
        config.max_position_value = get_float_input(
            "Maximum single position value ($)",
            default=config.max_position_value
        )

    elif idx == 3:  # Exit Parameters
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

    elif idx == 4:  # Strategy Selection
        strategies = get_available_strategies()
        if strategies:
            s_idx = select_from_list(strategies, "Select strategy")
            config.strategy_file = strategies[s_idx]['file']

    elif idx == 5:  # Trading Hours
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

    elif idx == 6:  # Full Setup
        return run_live_setup_wizard(force_reconfigure=True)

    else:  # Cancel
        print("\nNo changes made.")
        return config

    # Save changes with confirmation
    if idx < 6:
        if get_confirmation("Save these changes to LIVE trading config?"):
            config.save()
            print("\nConfiguration updated!")
        else:
            print("\nChanges NOT saved.")

    return config


if __name__ == "__main__":
    run_live_setup_wizard()
