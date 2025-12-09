"""
Configuration module for trading system.
"""

from .paper_trading_config import (
    PaperTradingConfig,
    get_available_strategies,
    mask_api_key,
    CONFIG_FILE,
    CONFIG_DIR,
)

from .live_trading_config import (
    LiveTradingConfig,
    log_trade,
    get_trade_log,
)

from .crypto_trading_config import (
    CryptoTradingConfig,
    ALPACA_CRYPTO_SYMBOLS,
    run_crypto_setup_wizard,
    run_crypto_reconfigure,
)

from .crypto_paper_trading_config import (
    CryptoPaperTradingConfig,
    DEFAULT_CRYPTO_SYMBOLS,
)

__all__ = [
    # Paper trading
    'PaperTradingConfig',
    'get_available_strategies',
    'mask_api_key',
    'CONFIG_FILE',
    'CONFIG_DIR',
    # Live trading
    'LiveTradingConfig',
    'log_trade',
    'get_trade_log',
    # Crypto trading
    'CryptoTradingConfig',
    'ALPACA_CRYPTO_SYMBOLS',
    'run_crypto_setup_wizard',
    'run_crypto_reconfigure',
    # Crypto paper trading
    'CryptoPaperTradingConfig',
    'DEFAULT_CRYPTO_SYMBOLS',
]
