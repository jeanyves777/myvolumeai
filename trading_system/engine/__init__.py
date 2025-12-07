"""
Trading engine components.
"""

from .order_manager import OrderManager
from .backtest_engine import BacktestEngine

# Paper and Live trading imports (optional - requires alpaca-py)
try:
    from .alpaca_client import AlpacaClient, Quote, Bar
    from .paper_trading_engine import PaperTradingEngine
    from .live_trading_engine import LiveTradingEngine
    TRADING_AVAILABLE = True
except ImportError:
    TRADING_AVAILABLE = False

__all__ = [
    'OrderManager',
    'BacktestEngine',
    'AlpacaClient',
    'PaperTradingEngine',
    'LiveTradingEngine',
    'Quote',
    'Bar',
    'TRADING_AVAILABLE',
]
