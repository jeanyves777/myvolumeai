"""
Alpaca API Client for Paper Trading

Handles:
- Live market data streaming (stocks and options)
- Paper trading order execution
- Account information
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass
import pytz

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        OrderType,
        OrderStatus,
        AssetClass,
    )
    from alpaca.data.live import StockDataStream, OptionDataStream, CryptoDataStream
    from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest,
        OptionBarsRequest,
        StockLatestQuoteRequest,
        OptionLatestQuoteRequest,
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-py not installed. Run: pip install alpaca-py")


EST = pytz.timezone('America/New_York')


@dataclass
class Quote:
    """Represents a market quote."""
    symbol: str
    bid: float
    ask: float
    mid: float
    timestamp: datetime


@dataclass
class Bar:
    """Represents a price bar."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime


class AlpacaClient:
    """
    Client for Alpaca API supporting both trading and market data.
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: If True, use paper trading endpoint
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package not installed. Run: pip install alpaca-py")

        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper

        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper
        )

        # Initialize data clients
        self.stock_data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        self.option_data_client = OptionHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        self.crypto_data_client = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )

        # Streaming clients (initialized on demand)
        self._stock_stream: Optional[StockDataStream] = None
        self._option_stream: Optional[OptionDataStream] = None
        self._crypto_stream: Optional[CryptoDataStream] = None

        # Callbacks for real-time data
        self._bar_callbacks: Dict[str, List[Callable]] = {}
        self._quote_callbacks: Dict[str, List[Callable]] = {}

    # ==================== Account Info ====================

    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            'id': account.id,
            'status': account.status,
            'currency': account.currency,
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'account_blocked': account.account_blocked,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()
        return [{
            'symbol': pos.symbol,
            'qty': float(pos.qty),
            'side': pos.side,
            'avg_entry_price': float(pos.avg_entry_price),
            'market_value': float(pos.market_value),
            'unrealized_pl': float(pos.unrealized_pl),
            'unrealized_plpc': float(pos.unrealized_plpc) * 100,  # Convert to percentage
            'current_price': float(pos.current_price),
        } for pos in positions]

    # ==================== Order Management ====================

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,  # 'buy' or 'sell'
        asset_class: str = 'us_equity'  # 'us_equity' or 'us_option'
    ) -> Dict[str, Any]:
        """
        Submit a market order.

        Args:
            symbol: The symbol to trade (for options, use OCC format)
            qty: Number of shares/contracts
            side: 'buy' or 'sell'
            asset_class: 'us_equity' or 'us_option'

        Returns:
            Order details dict
        """
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = self.trading_client.submit_order(order_request)
        return self._order_to_dict(order)

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        asset_class: str = 'us_equity'
    ) -> Dict[str, Any]:
        """Submit a limit order."""
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
        )

        order = self.trading_client.submit_order(order_request)
        return self._order_to_dict(order)

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        asset_class: str = 'us_equity'
    ) -> Dict[str, Any]:
        """
        Submit a bracket order with take-profit and stop-loss.

        Note: For options, Alpaca may not support bracket orders directly.
        We may need to manage SL/TP manually.
        """
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            order_class='bracket',
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
        )

        order = self.trading_client.submit_order(order_request)
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns number cancelled."""
        cancelled = self.trading_client.cancel_orders()
        return len(cancelled) if cancelled else 0

    def get_orders(
        self,
        status: str = 'open',  # 'open', 'closed', 'all'
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get orders by status."""
        request = GetOrdersRequest(
            status=status,
            limit=limit,
        )
        orders = self.trading_client.get_orders(request)
        return [self._order_to_dict(o) for o in orders]

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific order by ID."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return self._order_to_dict(order)
        except Exception:
            return None

    def _order_to_dict(self, order) -> Dict[str, Any]:
        """Convert Alpaca order object to dict."""
        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': float(order.qty) if order.qty else 0,
            'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
            'side': str(order.side.value),
            'type': str(order.type.value),
            'status': str(order.status.value),
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'stop_price': float(order.stop_price) if order.stop_price else None,
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            'created_at': order.created_at,
            'filled_at': order.filled_at,
        }

    # ==================== Market Data ====================

    def get_latest_stock_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for a stock."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_data_client.get_stock_latest_quote(request)
            if symbol in quotes:
                q = quotes[symbol]
                return Quote(
                    symbol=symbol,
                    bid=float(q.bid_price),
                    ask=float(q.ask_price),
                    mid=(float(q.bid_price) + float(q.ask_price)) / 2,
                    timestamp=q.timestamp,
                )
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
        return None

    def get_latest_option_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for an option (OCC symbol format)."""
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.option_data_client.get_option_latest_quote(request)
            if symbol in quotes:
                q = quotes[symbol]
                return Quote(
                    symbol=symbol,
                    bid=float(q.bid_price),
                    ask=float(q.ask_price),
                    mid=(float(q.bid_price) + float(q.ask_price)) / 2,
                    timestamp=q.timestamp,
                )
        except Exception as e:
            print(f"Error getting option quote for {symbol}: {e}")
        return None

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = '1Min',  # '1Min', '5Min', '1Hour', '1Day'
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        """Get historical bars for a stock."""
        if start is None:
            start = datetime.now(EST) - timedelta(hours=1)

        tf = TimeFrame.Minute
        if timeframe == '5Min':
            tf = TimeFrame.Minute
            # Note: Alpaca uses multiplier for 5-min bars
        elif timeframe == '1Hour':
            tf = TimeFrame.Hour
        elif timeframe == '1Day':
            tf = TimeFrame.Day

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            bars = self.stock_data_client.get_stock_bars(request)
            result = []
            if symbol in bars:
                for b in bars[symbol]:
                    result.append(Bar(
                        symbol=symbol,
                        open=float(b.open),
                        high=float(b.high),
                        low=float(b.low),
                        close=float(b.close),
                        volume=int(b.volume),
                        timestamp=b.timestamp,
                    ))
            return result
        except Exception as e:
            print(f"Error getting bars for {symbol}: {e}")
            return []

    def get_option_chain(
        self,
        underlying: str,
        expiration_date: Optional[datetime] = None
    ) -> List[str]:
        """
        Get available options contracts for underlying.

        Returns list of OCC symbol strings.
        """
        # Alpaca options data requires specific API calls
        # This is a simplified version - full implementation would query the options chain
        try:
            # For now, construct expected OCC symbols based on typical format
            # Full implementation would use Alpaca's options chain endpoint
            return []
        except Exception as e:
            print(f"Error getting option chain: {e}")
            return []

    # ==================== Streaming ====================

    async def start_stock_stream(self, symbols: List[str], on_bar: Callable = None, on_quote: Callable = None):
        """Start streaming stock data."""
        self._stock_stream = StockDataStream(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        if on_bar:
            async def bar_handler(bar):
                on_bar(Bar(
                    symbol=bar.symbol,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume),
                    timestamp=bar.timestamp,
                ))

            self._stock_stream.subscribe_bars(bar_handler, *symbols)

        if on_quote:
            async def quote_handler(quote):
                on_quote(Quote(
                    symbol=quote.symbol,
                    bid=float(quote.bid_price),
                    ask=float(quote.ask_price),
                    mid=(float(quote.bid_price) + float(quote.ask_price)) / 2,
                    timestamp=quote.timestamp,
                ))

            self._stock_stream.subscribe_quotes(quote_handler, *symbols)

        await self._stock_stream._run_forever()

    def stop_streams(self):
        """Stop all data streams."""
        if self._stock_stream:
            self._stock_stream.close()
        if self._option_stream:
            self._option_stream.close()
        if self._crypto_stream:
            self._crypto_stream.close()

    # ==================== Crypto Market Data ====================

    def get_latest_crypto_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get latest quote for a cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD', 'ETH/USD')

        Returns:
            Quote object or None if error
        """
        try:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.crypto_data_client.get_crypto_latest_quote(request)
            if symbol in quotes:
                q = quotes[symbol]
                return Quote(
                    symbol=symbol,
                    bid=float(q.bid_price),
                    ask=float(q.ask_price),
                    mid=(float(q.bid_price) + float(q.ask_price)) / 2,
                    timestamp=q.timestamp,
                )
        except Exception as e:
            print(f"Error getting crypto quote for {symbol}: {e}")
        return None

    def get_crypto_bars(
        self,
        symbol: str,
        timeframe: str = '1Min',  # '1Min', '5Min', '1Hour', '1Day'
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Bar]:
        """
        Get historical bars for a cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD', 'ETH/USD')
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')
            start: Start datetime (default: 24 hours ago)
            end: End datetime (default: now)
            limit: Maximum number of bars

        Returns:
            List of Bar objects
        """
        if start is None:
            start = datetime.now(pytz.UTC) - timedelta(hours=24)

        tf = TimeFrame.Minute
        if timeframe == '5Min':
            tf = TimeFrame.Minute
        elif timeframe == '1Hour':
            tf = TimeFrame.Hour
        elif timeframe == '1Day':
            tf = TimeFrame.Day

        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            bars = self.crypto_data_client.get_crypto_bars(request)
            result = []
            if symbol in bars:
                for b in bars[symbol]:
                    result.append(Bar(
                        symbol=symbol,
                        open=float(b.open),
                        high=float(b.high),
                        low=float(b.low),
                        close=float(b.close),
                        volume=int(b.volume) if b.volume else 0,
                        timestamp=b.timestamp,
                    ))
            return result
        except Exception as e:
            print(f"Error getting crypto bars for {symbol}: {e}")
            return []

    def get_multi_crypto_bars(
        self,
        symbols: List[str],
        timeframe: str = '1Min',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, List[Bar]]:
        """
        Get historical bars for multiple cryptocurrencies.

        Args:
            symbols: List of crypto symbols
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars per symbol

        Returns:
            Dict mapping symbol to list of bars
        """
        if start is None:
            start = datetime.now(pytz.UTC) - timedelta(hours=24)

        tf = TimeFrame.Minute
        if timeframe == '5Min':
            tf = TimeFrame.Minute
        elif timeframe == '1Hour':
            tf = TimeFrame.Hour
        elif timeframe == '1Day':
            tf = TimeFrame.Day

        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            bars_data = self.crypto_data_client.get_crypto_bars(request)
            result = {}
            for symbol in symbols:
                result[symbol] = []
                if symbol in bars_data:
                    for b in bars_data[symbol]:
                        result[symbol].append(Bar(
                            symbol=symbol,
                            open=float(b.open),
                            high=float(b.high),
                            low=float(b.low),
                            close=float(b.close),
                            volume=int(b.volume) if b.volume else 0,
                            timestamp=b.timestamp,
                        ))
            return result
        except Exception as e:
            print(f"Error getting multi crypto bars: {e}")
            return {s: [] for s in symbols}

    # ==================== Crypto Order Management ====================

    def submit_crypto_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # 'buy' or 'sell'
        notional: Optional[float] = None  # Dollar amount instead of qty
    ) -> Dict[str, Any]:
        """
        Submit a market order for cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            qty: Quantity to trade (fractional allowed)
            side: 'buy' or 'sell'
            notional: Optional dollar amount (if provided, qty is ignored)

        Returns:
            Order details dict
        """
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        if notional:
            order_request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=order_side,
                time_in_force=TimeInForce.GTC,
            )
        else:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
            )

        order = self.trading_client.submit_order(order_request)
        return self._order_to_dict(order)

    def submit_crypto_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float
    ) -> Dict[str, Any]:
        """
        Submit a limit order for cryptocurrency.

        Args:
            symbol: Crypto symbol
            qty: Quantity (fractional allowed)
            side: 'buy' or 'sell'
            limit_price: Limit price

        Returns:
            Order details dict
        """
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
        )

        order = self.trading_client.submit_order(order_request)
        return self._order_to_dict(order)

    # ==================== Crypto Streaming ====================

    async def start_crypto_stream(
        self,
        symbols: List[str],
        on_bar: Callable = None,
        on_quote: Callable = None
    ):
        """
        Start streaming crypto data.

        Args:
            symbols: List of crypto symbols to stream
            on_bar: Callback for bar updates
            on_quote: Callback for quote updates
        """
        self._crypto_stream = CryptoDataStream(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        if on_bar:
            async def bar_handler(bar):
                on_bar(Bar(
                    symbol=bar.symbol,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume) if bar.volume else 0,
                    timestamp=bar.timestamp,
                ))

            self._crypto_stream.subscribe_bars(bar_handler, *symbols)

        if on_quote:
            async def quote_handler(quote):
                on_quote(Quote(
                    symbol=quote.symbol,
                    bid=float(quote.bid_price),
                    ask=float(quote.ask_price),
                    mid=(float(quote.bid_price) + float(quote.ask_price)) / 2,
                    timestamp=quote.timestamp,
                ))

            self._crypto_stream.subscribe_quotes(quote_handler, *symbols)

        await self._crypto_stream._run_forever()

    # ==================== Options Helpers ====================

    @staticmethod
    def format_occ_symbol(
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str  # 'C' or 'P'
    ) -> str:
        """
        Format option symbol in OCC format.

        Example: COIN251219C00350000
        = COIN, Dec 19 2025, Call, $350.00 strike
        """
        exp_str = expiration.strftime('%y%m%d')
        strike_str = f"{int(strike * 1000):08d}"
        return f"{underlying}{exp_str}{option_type.upper()}{strike_str}"

    @staticmethod
    def parse_occ_symbol(occ_symbol: str) -> Dict[str, Any]:
        """Parse OCC symbol into components."""
        # OCC format: UNDERLYING + YYMMDD + C/P + 00000000 (strike * 1000)
        # Find where the date starts (first digit after letters)
        i = 0
        while i < len(occ_symbol) and not occ_symbol[i].isdigit():
            i += 1

        underlying = occ_symbol[:i]
        date_str = occ_symbol[i:i+6]
        option_type = occ_symbol[i+6]
        strike_str = occ_symbol[i+7:]

        return {
            'underlying': underlying,
            'expiration': datetime.strptime(date_str, '%y%m%d'),
            'option_type': 'CALL' if option_type == 'C' else 'PUT',
            'strike': int(strike_str) / 1000,
        }


def test_connection(api_key: str, api_secret: str, paper: bool = True) -> bool:
    """Test API connection by fetching account info."""
    try:
        client = AlpacaClient(api_key, api_secret, paper)
        account = client.get_account()
        print(f"Connected to Alpaca ({('Paper' if paper else 'Live')})")
        print(f"  Account ID: {account['id']}")
        print(f"  Cash: ${account['cash']:,.2f}")
        print(f"  Buying Power: ${account['buying_power']:,.2f}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
