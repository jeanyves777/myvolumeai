"""
Strategy logger with colored output.
Compatible with NautilusTrader LogColor enum.
"""

from enum import Enum
from datetime import datetime


class LogColor(Enum):
    """Log colors for console output"""
    NORMAL = "NORMAL"
    GREEN = "GREEN"
    BLUE = "BLUE"
    CYAN = "CYAN"
    YELLOW = "YELLOW"
    RED = "RED"
    MAGENTA = "MAGENTA"


class StrategyLogger:
    """
    Logger for trading strategies with colored output.

    Compatible with NautilusTrader's logging interface.
    """

    # ANSI color codes
    COLORS = {
        LogColor.NORMAL: '\033[0m',
        LogColor.GREEN: '\033[92m',
        LogColor.BLUE: '\033[94m',
        LogColor.CYAN: '\033[96m',
        LogColor.YELLOW: '\033[93m',
        LogColor.RED: '\033[91m',
        LogColor.MAGENTA: '\033[95m',
    }
    RESET = '\033[0m'

    def __init__(self, name: str = "Strategy", enabled: bool = True):
        """
        Initialize logger.

        Parameters
        ----------
        name : str
            Logger name (usually strategy class name)
        enabled : bool
            Whether logging is enabled
        """
        self.name = name
        self.enabled = enabled

    def _format_message(self, msg: str, color: LogColor = LogColor.NORMAL) -> str:
        """Format log message with timestamp and color"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_code = self.COLORS.get(color, '')
        return f"{color_code}[{timestamp}] [{self.name}] {msg}{self.RESET}"

    def info(self, msg: str, color: LogColor = LogColor.NORMAL) -> None:
        """Log info message"""
        if self.enabled:
            print(self._format_message(msg, color))

    def warning(self, msg: str, color: LogColor = LogColor.YELLOW) -> None:
        """Log warning message"""
        if self.enabled:
            print(self._format_message(f"⚠️  {msg}", color))

    def error(self, msg: str, color: LogColor = LogColor.RED) -> None:
        """Log error message"""
        if self.enabled:
            print(self._format_message(f"❌ {msg}", color))

    def debug(self, msg: str, color: LogColor = LogColor.BLUE) -> None:
        """Log debug message"""
        if self.enabled:
            print(self._format_message(f"🔍 {msg}", color))

    def success(self, msg: str, color: LogColor = LogColor.GREEN) -> None:
        """Log success message"""
        if self.enabled:
            print(self._format_message(f"✓ {msg}", color))
