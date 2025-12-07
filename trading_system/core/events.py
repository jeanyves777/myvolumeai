"""
Event system for the trading engine.

Events are the primary mechanism for communication between components.
The backtesting engine generates events, and strategies respond to them.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from .models import Bar, Quote, Order, Fill, Position, OrderStatus


class EventType(Enum):
    """Types of events in the trading system"""
    BAR = "BAR"                      # New bar data
    QUOTE = "QUOTE"                  # New quote data
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_FILLED = "ORDER_FILLED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CHANGED = "POSITION_CHANGED"
    POSITION_CLOSED = "POSITION_CLOSED"
    TRADE_COMPLETED = "TRADE_COMPLETED"


@dataclass
class Event:
    """Base event class"""
    event_type: EventType = field(default=EventType.BAR)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Any = None

    def __str__(self) -> str:
        return f"{self.event_type.value} @ {self.timestamp}"


@dataclass
class BarEvent(Event):
    """Event triggered when new bar data is received"""
    bar: Optional[Bar] = None

    def __post_init__(self):
        self.event_type = EventType.BAR
        if self.bar and self.timestamp is None:
            self.timestamp = self.bar.timestamp


@dataclass
class QuoteEvent(Event):
    """Event triggered when new quote data is received"""
    quote: Optional[Quote] = None

    def __post_init__(self):
        self.event_type = EventType.QUOTE
        if self.quote and self.timestamp is None:
            self.timestamp = self.quote.timestamp


@dataclass
class OrderEvent(Event):
    """Event triggered on order state changes"""
    order: Optional[Order] = None
    message: str = ""

    def __post_init__(self):
        if self.order:
            status_to_event = {
                OrderStatus.SUBMITTED: EventType.ORDER_SUBMITTED,
                OrderStatus.ACCEPTED: EventType.ORDER_ACCEPTED,
                OrderStatus.REJECTED: EventType.ORDER_REJECTED,
                OrderStatus.CANCELLED: EventType.ORDER_CANCELLED,
                OrderStatus.FILLED: EventType.ORDER_FILLED,
            }
            self.event_type = status_to_event.get(self.order.status, EventType.ORDER_SUBMITTED)


@dataclass
class FillEvent(Event):
    """Event triggered when an order is filled"""
    fill: Optional[Fill] = None
    order: Optional[Order] = None  # For compatibility with strategy handlers

    def __post_init__(self):
        self.event_type = EventType.ORDER_FILLED
        if self.fill and self.timestamp is None:
            self.timestamp = self.fill.timestamp
        if self.fill and not self.order:
            self.order = self.fill.order

    @property
    def client_order_id(self) -> str:
        return self.order.client_order_id if self.order else ""

    @property
    def order_side(self):
        return self.order.side if self.order else None

    @property
    def last_px(self) -> float:
        return self.fill.fill_price if self.fill else 0.0

    @property
    def last_qty(self) -> int:
        return self.fill.fill_quantity if self.fill else 0


@dataclass
class PositionEvent(Event):
    """Event triggered on position changes"""
    position: Optional[Position] = None
    previous_quantity: int = 0

    def __post_init__(self):
        if self.position:
            if self.previous_quantity == 0 and not self.position.is_flat:
                self.event_type = EventType.POSITION_OPENED
            elif self.position.is_flat:
                self.event_type = EventType.POSITION_CLOSED
            else:
                self.event_type = EventType.POSITION_CHANGED


@dataclass
class TradeEvent(Event):
    """Event triggered when a trade is completed (entry + exit)"""
    trade: Any = None  # Trade object

    def __post_init__(self):
        self.event_type = EventType.TRADE_COMPLETED
