"""
Core data models for the trading system.

These classes represent the fundamental building blocks:
- Bar: OHLCV price data
- Quote: Bid/Ask quotes
- Order: Trading orders with various types
- Position: Open positions with P&L tracking
- Trade: Executed trades
- Fill: Order fill events
- Account: Trading account with balance tracking
- Instrument: Tradeable instruments (stocks, options, etc.)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
import uuid


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class OrderSide(Enum):
    """Order side (buy or sell)"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types supported by the system"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "PENDING"          # Created but not submitted
    SUBMITTED = "SUBMITTED"      # Submitted to exchange
    ACCEPTED = "ACCEPTED"        # Accepted by exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"            # Completely filled
    CANCELLED = "CANCELLED"      # Cancelled by user or system
    REJECTED = "REJECTED"        # Rejected by exchange
    EXPIRED = "EXPIRED"          # Time expired


class TimeInForce(Enum):
    """Order time-in-force options"""
    GTC = "GTC"      # Good Till Cancelled
    DAY = "DAY"      # Day order
    IOC = "IOC"      # Immediate or Cancel
    FOK = "FOK"      # Fill or Kill


class PositionSide(Enum):
    """Position side"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class InstrumentType(Enum):
    """Types of tradeable instruments"""
    STOCK = "STOCK"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"


class OptionType(Enum):
    """Option contract types"""
    CALL = "CALL"
    PUT = "PUT"


# ═══════════════════════════════════════════════════════════════════════
# MARKET DATA MODELS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Bar:
    """
    OHLCV bar data representing price action for a time period.

    Attributes:
        symbol: Instrument symbol
        timestamp: Bar timestamp (UTC)
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
        ts_event: Event timestamp in nanoseconds (for compatibility)
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    ts_event: Optional[int] = None  # Nanoseconds timestamp

    def __post_init__(self):
        if self.ts_event is None:
            self.ts_event = int(self.timestamp.timestamp() * 1_000_000_000)

    @property
    def mid_price(self) -> float:
        """Return mid price (average of open and close)"""
        return (self.open + self.close) / 2

    @property
    def range(self) -> float:
        """Return bar range (high - low)"""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Return bar body (abs of open - close)"""
        return abs(self.open - self.close)

    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish (close > open)"""
        return self.close > self.open

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }


@dataclass
class Quote:
    """
    Bid/Ask quote data.

    Attributes:
        symbol: Instrument symbol
        timestamp: Quote timestamp
        bid_price: Best bid price
        ask_price: Best ask price
        bid_size: Bid size
        ask_size: Ask size
    """
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int = 0
    ask_size: int = 0

    @property
    def mid_price(self) -> float:
        """Return mid price"""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Return bid-ask spread"""
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        """Return spread as percentage of mid price"""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0


# ═══════════════════════════════════════════════════════════════════════
# INSTRUMENT MODELS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Instrument:
    """
    Base tradeable instrument.

    Attributes:
        symbol: Unique symbol identifier
        instrument_type: Type of instrument
        currency: Quote currency
        multiplier: Contract multiplier (100 for options)
        tick_size: Minimum price increment
        min_quantity: Minimum order quantity
        exchange: Exchange/venue name
    """
    symbol: str
    instrument_type: InstrumentType = InstrumentType.STOCK
    currency: str = "USD"
    multiplier: int = 1
    tick_size: float = 0.01
    min_quantity: int = 1
    exchange: str = "UNKNOWN"

    @property
    def id(self) -> str:
        """Return instrument ID (symbol.exchange)"""
        return f"{self.symbol}.{self.exchange}"

    def make_price(self, price: float) -> float:
        """Round price to tick size"""
        ticks = round(price / self.tick_size)
        return ticks * self.tick_size


@dataclass
class OptionContract(Instrument):
    """
    Option contract instrument.

    Attributes:
        underlying_symbol: Underlying instrument symbol
        option_type: CALL or PUT
        strike_price: Strike price
        expiration: Expiration datetime
    """
    underlying_symbol: str = ""
    option_type: OptionType = OptionType.CALL
    strike_price: float = 0.0
    expiration: Optional[datetime] = None
    expiration_ns: Optional[int] = None  # For compatibility

    def __post_init__(self):
        self.instrument_type = InstrumentType.OPTION
        self.multiplier = 100  # Standard options multiplier
        if self.expiration and self.expiration_ns is None:
            self.expiration_ns = int(self.expiration.timestamp() * 1_000_000_000)

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT

    @property
    def days_to_expiry(self) -> int:
        """Calculate days until expiration"""
        if self.expiration is None:
            return 0
        delta = self.expiration - datetime.now()
        return max(0, delta.days)


# ═══════════════════════════════════════════════════════════════════════
# ORDER MODELS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Order:
    """
    Trading order with full lifecycle management.

    Attributes:
        instrument: Instrument to trade
        side: BUY or SELL
        quantity: Order quantity
        order_type: MARKET, LIMIT, STOP_MARKET, STOP_LIMIT
        price: Limit price (for LIMIT and STOP_LIMIT orders)
        stop_price: Stop trigger price (for STOP orders)
        time_in_force: GTC, DAY, IOC, FOK
        client_order_id: Unique client-assigned ID
        status: Current order status
        filled_quantity: Quantity filled so far
        avg_fill_price: Average fill price
        reduce_only: Only reduce position, don't increase
        parent_order_id: Parent order ID for bracket orders
        linked_orders: List of linked order IDs (for OCO)
    """
    instrument: Instrument
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    reduce_only: bool = False
    parent_order_id: Optional[str] = None
    linked_orders: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    # For realistic backtesting: orders can only fill on bars AFTER this timestamp
    # This prevents SL/TP from triggering on the same bar as entry
    first_eligible_bar_timestamp: Optional[datetime] = None

    @property
    def is_pending(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_cancelled(self) -> bool:
        return self.status == OrderStatus.CANCELLED

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_order_id': self.client_order_id,
            'symbol': self.instrument.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_at': self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════
# FILL / TRADE MODELS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Fill:
    """
    Order fill event representing partial or complete execution.

    Attributes:
        order: The order that was filled
        fill_price: Execution price
        fill_quantity: Quantity filled in this event
        timestamp: Fill timestamp
        commission: Commission paid
        fill_id: Unique fill identifier
    """
    order: Order
    fill_price: float
    fill_quantity: int
    timestamp: datetime
    commission: float = 0.0
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def fill_value(self) -> float:
        """Calculate total fill value (price * quantity * multiplier)"""
        return self.fill_price * self.fill_quantity * self.order.instrument.multiplier

    @property
    def side(self) -> OrderSide:
        return self.order.side

    @property
    def symbol(self) -> str:
        return self.order.instrument.symbol

    def to_dict(self) -> Dict[str, Any]:
        return {
            'fill_id': self.fill_id,
            'order_id': self.order.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'fill_price': self.fill_price,
            'fill_quantity': self.fill_quantity,
            'fill_value': self.fill_value,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class Trade:
    """
    Completed trade (entry + exit) with P&L calculation.

    Attributes:
        instrument: Traded instrument
        side: Trade direction (LONG entry = BUY)
        entry_price: Entry fill price
        exit_price: Exit fill price
        quantity: Trade quantity
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        entry_fill_id: Entry fill ID
        exit_fill_id: Exit fill ID
        commission: Total commission paid
        signal: Market signal at entry (BULLISH/BEARISH/NEUTRAL)
        exit_reason: Reason for exit (TP/SL/TIME/FORCE)
    """
    instrument: Instrument
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    entry_fill_id: str = ""
    exit_fill_id: str = ""
    commission: float = 0.0
    signal: str = ""  # BULLISH, BEARISH, or NEUTRAL
    exit_reason: str = ""  # TP, SL, TIME, FORCE
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def pnl(self) -> float:
        """Calculate realized P&L"""
        if self.side == PositionSide.LONG:
            gross = (self.exit_price - self.entry_price) * self.quantity * self.instrument.multiplier
        else:
            gross = (self.entry_price - self.exit_price) * self.quantity * self.instrument.multiplier
        return gross - self.commission

    @property
    def pnl_pct(self) -> float:
        """Calculate P&L as percentage of entry value"""
        entry_value = self.entry_price * self.quantity * self.instrument.multiplier
        return (self.pnl / entry_value * 100) if entry_value > 0 else 0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'symbol': self.instrument.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'signal': self.signal,
            'exit_reason': self.exit_reason,
        }


# ═══════════════════════════════════════════════════════════════════════
# POSITION MODEL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """
    Open position with real-time P&L tracking.

    Attributes:
        instrument: Position instrument
        quantity: Position quantity (positive = long, negative = short)
        avg_price: Average entry price
        entry_time: Position open timestamp
        realized_pnl: Realized P&L from partial closes
        signal: Market signal at entry (BULLISH/BEARISH/NEUTRAL)
        exit_reason: Reason for exit (TP/SL/TIME/FORCE) - set by strategy before close
    """
    instrument: Instrument
    quantity: int = 0
    avg_price: float = 0.0
    entry_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    signal: str = ""  # BULLISH, BEARISH, or NEUTRAL
    exit_reason: str = ""  # TP, SL, TIME, FORCE
    _fills: List[Fill] = field(default_factory=list)

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def market_value(self) -> float:
        """Calculate position market value at average price"""
        return abs(self.quantity) * self.avg_price * self.instrument.multiplier

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L given current price"""
        if self.is_flat:
            return 0.0

        if self.is_long:
            return (current_price - self.avg_price) * self.quantity * self.instrument.multiplier
        else:
            return (self.avg_price - current_price) * abs(self.quantity) * self.instrument.multiplier

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage"""
        if self.market_value == 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / self.market_value * 100)

    def update_from_fill(self, fill: Fill) -> Optional[Trade]:
        """
        Update position from a fill event.
        Returns a Trade if position was closed.
        """
        self._fills.append(fill)

        fill_qty = fill.fill_quantity if fill.side == OrderSide.BUY else -fill.fill_quantity

        # Opening or adding to position
        if self.is_flat or (self.is_long and fill_qty > 0) or (self.is_short and fill_qty < 0):
            total_value = (self.avg_price * abs(self.quantity) + fill.fill_price * abs(fill_qty))
            self.quantity += fill_qty
            if self.quantity != 0:
                self.avg_price = total_value / abs(self.quantity)
            if self.entry_time is None:
                self.entry_time = fill.timestamp
            return None

        # Reducing or closing position
        else:
            old_qty = self.quantity
            old_avg = self.avg_price

            close_qty = min(abs(fill_qty), abs(self.quantity))

            # Calculate realized P&L for closed portion
            if self.is_long:
                pnl = (fill.fill_price - old_avg) * close_qty * self.instrument.multiplier
            else:
                pnl = (old_avg - fill.fill_price) * close_qty * self.instrument.multiplier

            self.realized_pnl += pnl
            self.quantity += fill_qty

            # If position is closed, create a Trade record
            if abs(self.quantity) < abs(old_qty):
                trade = Trade(
                    instrument=self.instrument,
                    side=PositionSide.LONG if old_qty > 0 else PositionSide.SHORT,
                    entry_price=old_avg,
                    exit_price=fill.fill_price,
                    quantity=close_qty,
                    entry_time=self.entry_time or fill.timestamp,
                    exit_time=fill.timestamp,
                    commission=fill.commission,
                    signal=self.signal,
                    exit_reason=self.exit_reason,
                )

                # Reset if fully closed
                if self.is_flat:
                    self.avg_price = 0.0
                    self.entry_time = None
                    self.signal = ""
                    self.exit_reason = ""

                return trade

            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.instrument.symbol,
            'quantity': self.quantity,
            'side': self.side.value,
            'avg_price': self.avg_price,
            'market_value': self.market_value,
            'realized_pnl': self.realized_pnl,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
        }


# ═══════════════════════════════════════════════════════════════════════
# ACCOUNT MODEL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Account:
    """
    Trading account with balance and position tracking.

    Attributes:
        account_id: Unique account identifier
        currency: Account currency
        initial_balance: Starting balance
        cash_balance: Current cash balance
        positions: Dict of open positions by symbol
        trades: List of completed trades
    """
    account_id: str = "DEFAULT"
    currency: str = "USD"
    initial_balance: float = 100000.0
    cash_balance: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)
    orders: Dict[str, Order] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        """Calculate total realized P&L"""
        return sum(t.pnl for t in self.trades)

    @property
    def equity(self) -> float:
        """Calculate total equity (cash + position values)"""
        # In backtesting, equity is cash + unrealized P&L
        # For simplicity, we track it as cash (which is updated on fills)
        return self.cash_balance

    @property
    def total_return_pct(self) -> float:
        """Calculate total return percentage"""
        if self.initial_balance == 0:
            return 0.0
        return ((self.equity - self.initial_balance) / self.initial_balance) * 100

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.is_winner)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if not t.is_winner)

    @property
    def win_rate(self) -> float:
        if len(self.trades) == 0:
            return 0.0
        return (self.winning_trades / len(self.trades)) * 100

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)

    def get_or_create_position(self, instrument: Instrument) -> Position:
        """Get existing position or create new one"""
        if instrument.symbol not in self.positions:
            self.positions[instrument.symbol] = Position(instrument=instrument)
        return self.positions[instrument.symbol]

    def is_net_long(self, symbol: str) -> bool:
        """Check if net long for symbol"""
        pos = self.positions.get(symbol)
        return pos.is_long if pos else False

    def is_net_short(self, symbol: str) -> bool:
        """Check if net short for symbol"""
        pos = self.positions.get(symbol)
        return pos.is_short if pos else False

    def process_fill(self, fill: Fill) -> Optional[Trade]:
        """Process a fill and update account state"""
        self.fills.append(fill)

        # Get or create position
        position = self.get_or_create_position(fill.order.instrument)

        # Update position and check if a trade was closed
        trade = position.update_from_fill(fill)

        # Update cash balance
        if fill.side == OrderSide.BUY:
            self.cash_balance -= fill.fill_value + fill.commission
        else:
            self.cash_balance += fill.fill_value - fill.commission

        # Record completed trade
        if trade:
            self.trades.append(trade)

        return trade

    def to_dict(self) -> Dict[str, Any]:
        return {
            'account_id': self.account_id,
            'currency': self.currency,
            'initial_balance': self.initial_balance,
            'cash_balance': self.cash_balance,
            'equity': self.equity,
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'total_trades': len(self.trades),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'positions': {k: v.to_dict() for k, v in self.positions.items() if not v.is_flat},
        }
