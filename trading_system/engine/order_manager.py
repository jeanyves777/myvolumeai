"""
Order Management System.

Handles order lifecycle:
- Order creation (market, limit, stop)
- Order submission and validation
- Order matching and fill simulation
- Bracket/OCO order groups
- Position tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
import uuid

from ..core.models import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Fill, Position, Account, Instrument, Bar
)
from ..core.events import FillEvent, OrderEvent


@dataclass
class OrderGroup:
    """
    Group of related orders (bracket, OCO, etc.)

    Attributes:
        group_id: Unique group identifier
        orders: List of order IDs in the group
        group_type: 'OCO' (one-cancels-other), 'BRACKET', etc.
        parent_order_id: Parent order that triggered this group
    """
    group_id: str
    orders: List[str]
    group_type: str = "OCO"
    parent_order_id: Optional[str] = None


class OrderManager:
    """
    Manages order lifecycle and execution simulation.

    Responsibilities:
    - Create and validate orders
    - Simulate order fills based on market data
    - Handle bracket orders (entry + SL + TP)
    - Handle OCO orders (one-cancels-other)
    - Track order state changes
    - Generate fill events
    """

    def __init__(
        self,
        account: Account,
        slippage_pct: float = 0.1,
        commission_per_contract: float = 0.65,
        fill_probability: float = 1.0
    ):
        """
        Initialize Order Manager.

        Parameters
        ----------
        account : Account
            Trading account for position and balance tracking
        slippage_pct : float
            Simulated slippage as percentage of price (default 0.1%)
        commission_per_contract : float
            Commission per contract (default $0.65)
        fill_probability : float
            Probability of limit order fills (default 1.0 = always fill)
        """
        self.account = account
        self.slippage_pct = slippage_pct
        self.commission_per_contract = commission_per_contract
        self.fill_probability = fill_probability

        # Order storage
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.cancelled_orders: Dict[str, Order] = {}

        # Order groups (OCO, Bracket)
        self.order_groups: Dict[str, OrderGroup] = {}

        # Event callbacks
        self._on_fill_callbacks: List[Callable[[FillEvent], None]] = []
        self._on_order_update_callbacks: List[Callable[[OrderEvent], None]] = []

    # ═══════════════════════════════════════════════════════════════════════
    # ORDER CREATION
    # ═══════════════════════════════════════════════════════════════════════

    def create_market_order(
        self,
        instrument: Instrument,
        side: OrderSide,
        quantity: int,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> Order:
        """Create a market order"""
        order = Order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )
        return order

    def create_limit_order(
        self,
        instrument: Instrument,
        side: OrderSide,
        quantity: int,
        price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> Order:
        """Create a limit order"""
        order = Order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=instrument.make_price(price),
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )
        return order

    def create_stop_market_order(
        self,
        instrument: Instrument,
        side: OrderSide,
        quantity: int,
        stop_price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> Order:
        """Create a stop-market order"""
        order = Order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP_MARKET,
            stop_price=instrument.make_price(stop_price),
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )
        return order

    def create_stop_limit_order(
        self,
        instrument: Instrument,
        side: OrderSide,
        quantity: int,
        stop_price: float,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> Order:
        """Create a stop-limit order"""
        order = Order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP_LIMIT,
            price=instrument.make_price(limit_price),
            stop_price=instrument.make_price(stop_price),
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )
        return order

    # ═══════════════════════════════════════════════════════════════════════
    # BRACKET ORDERS
    # ═══════════════════════════════════════════════════════════════════════

    def create_bracket_order(
        self,
        instrument: Instrument,
        side: OrderSide,
        quantity: int,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> Tuple[Order, Order, Order]:
        """
        Create a bracket order (entry + take profit + stop loss).

        Parameters
        ----------
        instrument : Instrument
            Instrument to trade
        side : OrderSide
            Entry order side (BUY or SELL)
        quantity : int
            Order quantity
        take_profit_price : float
            Take profit limit price
        stop_loss_price : float
            Stop loss trigger price

        Returns
        -------
        Tuple[Order, Order, Order]
            Entry order, take profit order, stop loss order
        """
        # Entry order (market)
        entry_order = self.create_market_order(instrument, side, quantity)

        # Exit side is opposite of entry
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        # Take profit (limit order)
        tp_order = self.create_limit_order(
            instrument, exit_side, quantity, take_profit_price, reduce_only=True
        )
        tp_order.parent_order_id = entry_order.client_order_id

        # Stop loss (stop-market order)
        sl_order = self.create_stop_market_order(
            instrument, exit_side, quantity, stop_loss_price, reduce_only=True
        )
        sl_order.parent_order_id = entry_order.client_order_id

        # Link TP and SL as OCO
        tp_order.linked_orders = [sl_order.client_order_id]
        sl_order.linked_orders = [tp_order.client_order_id]

        # Create order group
        group = OrderGroup(
            group_id=str(uuid.uuid4())[:8],
            orders=[tp_order.client_order_id, sl_order.client_order_id],
            group_type="OCO",
            parent_order_id=entry_order.client_order_id,
        )
        self.order_groups[group.group_id] = group

        return entry_order, tp_order, sl_order

    # ═══════════════════════════════════════════════════════════════════════
    # ORDER SUBMISSION
    # ═══════════════════════════════════════════════════════════════════════

    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for execution.

        Parameters
        ----------
        order : Order
            Order to submit

        Returns
        -------
        bool
            True if order was accepted, False if rejected
        """
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.orders[order.client_order_id] = order
            self._notify_order_update(order)
            return False

        # Store order
        self.orders[order.client_order_id] = order
        self.pending_orders[order.client_order_id] = order

        # Update status
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.utcnow()
        order.updated_at = datetime.utcnow()

        # For market orders, we'll fill immediately on next bar
        # For limit/stop orders, they wait for trigger conditions

        self._notify_order_update(order)
        return True

    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission"""
        # Check quantity
        if order.quantity <= 0:
            return False

        # Check price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                return False

        # Check stop price for stop orders
        if order.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                return False

        # Check reduce_only orders have a position to reduce
        if order.reduce_only:
            position = self.account.get_position(order.instrument.symbol)
            if position is None or position.is_flat:
                return False

            # Check we're reducing, not increasing
            if order.side == OrderSide.BUY and position.is_long:
                return False
            if order.side == OrderSide.SELL and position.is_short:
                return False

        return True

    # ═══════════════════════════════════════════════════════════════════════
    # ORDER PROCESSING
    # ═══════════════════════════════════════════════════════════════════════

    def process_bar(self, bar: Bar) -> List[FillEvent]:
        """
        Process a bar and check for order fills.

        This is the main entry point for order simulation.
        Called by the backtesting engine for each bar.

        Parameters
        ----------
        bar : Bar
            Current price bar

        Returns
        -------
        List[FillEvent]
            List of fill events generated
        """
        fills = []
        orders_to_remove = []

        for order_id, order in list(self.pending_orders.items()):
            # Skip if instrument doesn't match
            if order.instrument.symbol != bar.symbol:
                continue

            # Skip if bar is before the order's first eligible timestamp
            # This prevents SL/TP from filling on the same bar as entry
            if order.first_eligible_bar_timestamp is not None:
                bar_ts = bar.timestamp
                eligible_ts = order.first_eligible_bar_timestamp

                # Normalize both timestamps to be comparable
                # Strip timezone info for comparison (both should be same timezone anyway)
                if hasattr(bar_ts, 'replace') and hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is not None:
                    bar_ts_naive = bar_ts.replace(tzinfo=None)
                else:
                    bar_ts_naive = bar_ts

                if hasattr(eligible_ts, 'replace') and hasattr(eligible_ts, 'tzinfo') and eligible_ts.tzinfo is not None:
                    eligible_ts_naive = eligible_ts.replace(tzinfo=None)
                else:
                    eligible_ts_naive = eligible_ts

                # Skip this order if bar is at or before the eligible timestamp
                if bar_ts_naive <= eligible_ts_naive:
                    continue

            fill = self._try_fill_order(order, bar)
            if fill:
                fills.append(fill)
                orders_to_remove.append(order_id)

        # Move filled orders (with defensive check)
        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                order = self.pending_orders.pop(order_id)
                self.filled_orders[order_id] = order

        return fills

    def _try_fill_order(self, order: Order, bar: Bar) -> Optional[FillEvent]:
        """
        Attempt to fill an order based on bar data.

        Parameters
        ----------
        order : Order
            Order to try filling
        bar : Bar
            Current price bar

        Returns
        -------
        Optional[FillEvent]
            Fill event if order was filled, None otherwise
        """
        fill_price = None

        if order.order_type == OrderType.MARKET:
            # Market orders fill at open with slippage
            fill_price = self._apply_slippage(bar.open, order.side)

        elif order.order_type == OrderType.LIMIT:
            # Limit buy fills if low <= price
            # Limit sell fills if high >= price
            if order.side == OrderSide.BUY and bar.low <= order.price:
                fill_price = min(order.price, bar.open)
            elif order.side == OrderSide.SELL and bar.high >= order.price:
                fill_price = max(order.price, bar.open)

        elif order.order_type == OrderType.STOP_MARKET:
            # Stop buy triggers if high >= stop_price
            # Stop sell triggers if low <= stop_price
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                fill_price = self._apply_slippage(max(order.stop_price, bar.open), order.side)
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                fill_price = self._apply_slippage(min(order.stop_price, bar.open), order.side)

        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop triggers, then acts as limit order
            triggered = False
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                triggered = True
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                triggered = True

            if triggered:
                # Check limit price
                if order.side == OrderSide.BUY and bar.low <= order.price:
                    fill_price = min(order.price, bar.open)
                elif order.side == OrderSide.SELL and bar.high >= order.price:
                    fill_price = max(order.price, bar.open)

        if fill_price is not None:
            return self._execute_fill(order, fill_price, bar.timestamp)

        return None

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to fill price"""
        slippage = price * (self.slippage_pct / 100)
        if side == OrderSide.BUY:
            return price + slippage  # Pay more when buying
        else:
            return price - slippage  # Get less when selling

    def _execute_fill(
        self,
        order: Order,
        fill_price: float,
        timestamp: datetime
    ) -> FillEvent:
        """
        Execute an order fill.

        Parameters
        ----------
        order : Order
            Order being filled
        fill_price : float
            Execution price
        timestamp : datetime
            Fill timestamp

        Returns
        -------
        FillEvent
            Fill event
        """
        # Calculate commission
        commission = self.commission_per_contract * order.quantity

        # Create fill
        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            timestamp=timestamp,
            commission=commission,
        )

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.filled_at = timestamp
        order.updated_at = timestamp

        # Update account
        self.account.process_fill(fill)

        # Handle linked orders (OCO)
        self._handle_linked_orders(order)

        # Create fill event
        fill_event = FillEvent(
            timestamp=timestamp,
            fill=fill,
            order=order,
        )

        # Notify callbacks
        self._notify_fill(fill_event)
        self._notify_order_update(order)

        return fill_event

    def _handle_linked_orders(self, filled_order: Order) -> None:
        """
        Handle linked orders (OCO) when one is filled.
        Cancels all linked orders.
        """
        for linked_id in filled_order.linked_orders:
            if linked_id in self.pending_orders:
                linked_order = self.pending_orders.pop(linked_id)
                linked_order.status = OrderStatus.CANCELLED
                linked_order.updated_at = datetime.utcnow()
                self.cancelled_orders[linked_id] = linked_order
                self._notify_order_update(linked_order)

    # ═══════════════════════════════════════════════════════════════════════
    # ORDER CANCELLATION
    # ═══════════════════════════════════════════════════════════════════════

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Parameters
        ----------
        order_id : str
            Order ID to cancel

        Returns
        -------
        bool
            True if cancelled, False if not found or already filled
        """
        if order_id not in self.pending_orders:
            return False

        order = self.pending_orders.pop(order_id)
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        self.cancelled_orders[order_id] = order

        self._notify_order_update(order)
        return True

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all pending orders, optionally filtered by symbol.

        Parameters
        ----------
        symbol : str, optional
            Only cancel orders for this symbol

        Returns
        -------
        int
            Number of orders cancelled
        """
        orders_to_cancel = []

        for order_id, order in self.pending_orders.items():
            if symbol is None or order.instrument.symbol == symbol:
                orders_to_cancel.append(order_id)

        cancelled = 0
        for order_id in orders_to_cancel:
            if self.cancel_order(order_id):
                cancelled += 1

        return cancelled

    # ═══════════════════════════════════════════════════════════════════════
    # EVENT CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════

    def on_fill(self, callback: Callable[[FillEvent], None]) -> None:
        """Register callback for fill events"""
        self._on_fill_callbacks.append(callback)

    def on_order_update(self, callback: Callable[[OrderEvent], None]) -> None:
        """Register callback for order state changes"""
        self._on_order_update_callbacks.append(callback)

    def _notify_fill(self, event: FillEvent) -> None:
        """Notify all fill callbacks"""
        for callback in self._on_fill_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in fill callback: {e}")

    def _notify_order_update(self, order: Order) -> None:
        """Notify all order update callbacks"""
        event = OrderEvent(
            timestamp=order.updated_at,
            order=order,
        )
        for callback in self._on_order_update_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in order update callback: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # QUERY METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all pending orders, optionally filtered by symbol"""
        orders = list(self.pending_orders.values())
        if symbol:
            orders = [o for o in orders if o.instrument.symbol == symbol]
        return orders

    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders"""
        return list(self.filled_orders.values())

    def has_pending_orders(self, symbol: Optional[str] = None) -> bool:
        """Check if there are pending orders"""
        if symbol:
            return any(o.instrument.symbol == symbol for o in self.pending_orders.values())
        return len(self.pending_orders) > 0
