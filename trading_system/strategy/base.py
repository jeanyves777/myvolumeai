"""
Base Strategy Class.

All trading strategies inherit from this class.
Provides the interface for backtesting and live trading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from ..core.models import (
    Bar, Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Fill, Position, Trade, Account, Instrument, OptionContract,
    InstrumentType, OptionType
)
from ..core.events import FillEvent, BarEvent
from .logger import StrategyLogger, LogColor

if TYPE_CHECKING:
    from ..engine.backtest_engine import BacktestEngine


@dataclass
class StrategyConfig:
    """
    Base configuration for strategies.

    Subclass this to add strategy-specific parameters.
    """
    strategy_id: str = ""

    def __post_init__(self):
        if not self.strategy_id:
            import uuid
            self.strategy_id = str(uuid.uuid4())[:8]


class OrderFactory:
    """
    Factory for creating orders.

    Provides methods compatible with NautilusTrader's order factory.
    """

    def __init__(self, strategy: 'Strategy'):
        self._strategy = strategy

    def market(
        self,
        instrument_id,
        order_side: OrderSide,
        quantity,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        **kwargs
    ) -> Order:
        """Create a market order"""
        instrument = self._strategy._get_instrument(instrument_id)
        qty = int(quantity) if hasattr(quantity, '__int__') else int(str(quantity))

        return Order(
            instrument=instrument,
            side=order_side,
            quantity=qty,
            order_type=OrderType.MARKET,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )

    def limit(
        self,
        instrument_id,
        order_side: OrderSide,
        quantity,
        price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        post_only: bool = False,
        **kwargs
    ) -> Order:
        """Create a limit order"""
        instrument = self._strategy._get_instrument(instrument_id)
        qty = int(quantity) if hasattr(quantity, '__int__') else int(str(quantity))
        px = float(price) if hasattr(price, '__float__') else float(str(price))

        return Order(
            instrument=instrument,
            side=order_side,
            quantity=qty,
            order_type=OrderType.LIMIT,
            price=px,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )

    def stop_market(
        self,
        instrument_id,
        order_side: OrderSide,
        quantity,
        trigger_price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        **kwargs
    ) -> Order:
        """Create a stop-market order"""
        instrument = self._strategy._get_instrument(instrument_id)
        qty = int(quantity) if hasattr(quantity, '__int__') else int(str(quantity))
        px = float(trigger_price) if hasattr(trigger_price, '__float__') else float(str(trigger_price))

        return Order(
            instrument=instrument,
            side=order_side,
            quantity=qty,
            order_type=OrderType.STOP_MARKET,
            stop_price=px,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )

    def stop_limit(
        self,
        instrument_id,
        order_side: OrderSide,
        quantity,
        trigger_price,
        price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        **kwargs
    ) -> Order:
        """Create a stop-limit order"""
        instrument = self._strategy._get_instrument(instrument_id)
        qty = int(quantity) if hasattr(quantity, '__int__') else int(str(quantity))
        trigger_px = float(trigger_price) if hasattr(trigger_price, '__float__') else float(str(trigger_price))
        limit_px = float(price) if hasattr(price, '__float__') else float(str(price))

        return Order(
            instrument=instrument,
            side=order_side,
            quantity=qty,
            order_type=OrderType.STOP_LIMIT,
            price=limit_px,
            stop_price=trigger_px,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )


class Clock:
    """
    Clock for strategy time management.

    Provides current time in backtesting or live trading.
    """

    def __init__(self):
        self._current_time: Optional[datetime] = None

    def set_time(self, time: datetime) -> None:
        """Set current time (used by backtest engine)"""
        self._current_time = time

    def utc_now(self) -> datetime:
        """Get current UTC time"""
        if self._current_time:
            return self._current_time
        return datetime.utcnow()


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Strategies must implement:
    - on_start(): Called when strategy starts
    - on_bar(bar): Called for each price bar
    - on_stop(): Called when strategy stops

    Optional overrides:
    - on_order_filled(event): Called when an order is filled
    - on_position_opened(position): Called when a position opens
    - on_position_closed(position): Called when a position closes
    - on_reset(): Called to reset strategy state
    """

    def __init__(self, config: StrategyConfig = None):
        """
        Initialize strategy.

        Parameters
        ----------
        config : StrategyConfig, optional
            Strategy configuration
        """
        self.config = config or StrategyConfig()

        # Engine reference (set by backtest engine)
        self._engine: Optional['BacktestEngine'] = None

        # Logger
        self.log = StrategyLogger(name=self.__class__.__name__)

        # Order factory
        self.order_factory = OrderFactory(self)

        # Clock
        self._clock = Clock()

    def _set_engine(self, engine: 'BacktestEngine') -> None:
        """Set engine reference (called by BacktestEngine)"""
        self._engine = engine

    # ═══════════════════════════════════════════════════════════════════════
    # PROPERTIES FOR COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def cache(self):
        """Access engine cache (for instrument lookup)"""
        return self._engine

    @property
    def portfolio(self):
        """Access portfolio (for position checking)"""
        return self._engine

    # ═══════════════════════════════════════════════════════════════════════
    # INSTRUMENT METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def _get_instrument(self, instrument_id) -> Instrument:
        """Get instrument by ID"""
        if self._engine:
            # Handle various ID formats
            symbol = str(instrument_id)
            if '.' in symbol:
                symbol = symbol.split('.')[0]

            # Try direct lookup
            inst = self._engine.instruments.get(symbol)
            if inst:
                return inst

            # Try searching all instruments
            for sym, inst in self._engine.instruments.items():
                if symbol in sym or sym in symbol:
                    return inst

        # Fallback: create a basic instrument
        return Instrument(
            symbol=str(instrument_id),
            instrument_type=InstrumentType.OPTION,
            multiplier=100,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # INDICATOR REGISTRATION
    # ═══════════════════════════════════════════════════════════════════════

    def register_indicator_for_bars(self, bar_type, indicator) -> None:
        """
        Register an indicator for automatic updates.

        Parameters
        ----------
        bar_type : any
            Bar type identifier (for compatibility)
        indicator : Indicator
            Indicator to register
        """
        # In our system, indicators are updated manually in on_bar
        # This is a no-op for compatibility
        pass

    # ═══════════════════════════════════════════════════════════════════════
    # BAR SUBSCRIPTION
    # ═══════════════════════════════════════════════════════════════════════

    def subscribe_bars(self, bar_type) -> None:
        """
        Subscribe to bar updates.

        Parameters
        ----------
        bar_type : any
            Bar type identifier
        """
        # In backtesting, all bars are processed automatically
        pass

    def request_bars(self, bar_type, start=None, end=None) -> None:
        """
        Request historical bars.

        Parameters
        ----------
        bar_type : any
            Bar type identifier
        start : datetime, optional
            Start time
        end : datetime, optional
            End time
        """
        # In backtesting, data is pre-loaded
        pass

    # ═══════════════════════════════════════════════════════════════════════
    # ORDER SUBMISSION
    # ═══════════════════════════════════════════════════════════════════════

    def submit_order(self, order: Order) -> None:
        """
        Submit an order for execution.

        Parameters
        ----------
        order : Order
            Order to submit
        """
        if self._engine:
            self._engine.order_manager.submit_order(order)
        # else: No internal engine - order handled externally (e.g., crypto_paper_trading_engine)

    def cancel_order(self, order_id) -> bool:
        """
        Cancel a pending order.

        Parameters
        ----------
        order_id : str or ClientOrderId
            Order ID to cancel

        Returns
        -------
        bool
            True if cancelled
        """
        if self._engine:
            # Handle both string and ClientOrderId
            oid = str(order_id)
            return self._engine.order_manager.cancel_order(oid)
        return False

    def cancel_all_orders(self, instrument_id=None) -> int:
        """
        Cancel all pending orders.

        Parameters
        ----------
        instrument_id : any, optional
            Only cancel orders for this instrument

        Returns
        -------
        int
            Number of orders cancelled
        """
        if self._engine:
            symbol = None
            if instrument_id:
                symbol = str(instrument_id).split('.')[0]
            return self._engine.order_manager.cancel_all_orders(symbol)
        return 0

    # ═══════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def close_all_positions(self, instrument_id) -> None:
        """
        Close all positions for an instrument.

        Parameters
        ----------
        instrument_id : any
            Instrument to close positions for
        """
        if not self._engine:
            return

        symbol = str(instrument_id).split('.')[0]
        position = self._engine.account.get_position(symbol)

        if position and not position.is_flat:
            # Create closing order
            side = OrderSide.SELL if position.is_long else OrderSide.BUY
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=side,
                quantity=abs(position.quantity),
                reduce_only=True,
            )
            self.submit_order(order)

    # ═══════════════════════════════════════════════════════════════════════
    # LIFECYCLE METHODS (to be implemented by subclasses)
    # ═══════════════════════════════════════════════════════════════════════

    def on_start(self) -> None:
        """
        Called when strategy starts.

        Override to initialize indicators, subscribe to data, etc.
        """
        pass

    @abstractmethod
    def on_bar(self, bar: Bar) -> None:
        """
        Called for each price bar.

        This is the main entry point for strategy logic.

        Parameters
        ----------
        bar : Bar
            Current price bar
        """
        pass

    def on_stop(self) -> None:
        """
        Called when strategy stops.

        Override to cleanup, close positions, etc.
        """
        pass

    def on_order_filled(self, event: FillEvent) -> None:
        """
        Called when an order is filled.

        Override to handle fills (e.g., create SL/TP orders).

        Parameters
        ----------
        event : FillEvent
            Fill event with order and fill details
        """
        pass

    def on_position_opened(self, position: Position) -> None:
        """
        Called when a position is opened.

        Parameters
        ----------
        position : Position
            Newly opened position
        """
        pass

    def on_position_closed(self, position: Position) -> None:
        """
        Called when a position is closed.

        Parameters
        ----------
        position : Position
            Closed position with realized P&L
        """
        pass

    def on_reset(self) -> None:
        """
        Called to reset strategy state.

        Override to reset internal state variables.
        """
        pass
