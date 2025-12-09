"""
Kraken Futures API Client

Handles:
- REST API for order management and account info
- Market data (OHLC, tickers, order book)
- Authentication (HMAC-SHA512)

Supports both demo (paper) and production environments:
- Demo: demo-futures.kraken.com
- Production: futures.kraken.com

ETH Perpetual symbol: PI_ETHUSD
"""

import hashlib
import hmac
import base64
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import pytz

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not installed. Run: pip install requests")


UTC = pytz.UTC


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
    """Represents a price bar (OHLC)."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


@dataclass
class Position:
    """Represents a futures position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    unrealized_pnl: float
    margin_used: float
    leverage: float


@dataclass
class Order:
    """Represents an order."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'lmt', 'mkt', 'stp', 'take_profit'
    size: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    filled_size: float
    status: str  # 'open', 'filled', 'cancelled'
    timestamp: datetime


class KrakenFuturesClient:
    """
    Client for Kraken Futures API.

    Supports ETH perpetual futures (PI_ETHUSD) trading with margin.

    API Documentation: https://docs.futures.kraken.com/
    """

    # API Base URLs
    DEMO_BASE_URL = "https://demo-futures.kraken.com"
    PROD_BASE_URL = "https://futures.kraken.com"

    # API paths
    API_PATH = "/derivatives/api/v3"

    # ETH perpetual symbol
    ETH_PERPETUAL = "PI_ETHUSD"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        demo: bool = True,
        timeout: int = 30
    ):
        """
        Initialize Kraken Futures client.

        Args:
            api_key: Kraken Futures API key
            api_secret: Kraken Futures API secret (base64 encoded)
            demo: If True, use demo environment (default)
            timeout: Request timeout in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package not installed. Run: pip install requests")

        self.api_key = api_key
        self.api_secret = api_secret
        self.demo = demo
        self.timeout = timeout

        self.base_url = self.DEMO_BASE_URL if demo else self.PROD_BASE_URL
        self.session = requests.Session()

    # ==================== Authentication ====================

    def _get_nonce(self) -> str:
        """Get nonce for API requests (timestamp in milliseconds)."""
        return str(int(time.time() * 1000))

    def _sign_request(
        self,
        endpoint: str,
        post_data: str = "",
        nonce: str = None
    ) -> Tuple[str, str]:
        """
        Sign a request using HMAC-SHA512.

        Args:
            endpoint: API endpoint path
            post_data: URL-encoded POST data
            nonce: Request nonce

        Returns:
            Tuple of (nonce, signature)
        """
        if nonce is None:
            nonce = self._get_nonce()

        # Message = postData + nonce + endpoint
        message = post_data + nonce + endpoint

        # SHA-256 hash of message
        sha256_hash = hashlib.sha256(message.encode('utf-8')).digest()

        # HMAC-SHA512 with base64-decoded secret
        try:
            secret_decoded = base64.b64decode(self.api_secret)
        except Exception:
            # If secret is not base64 encoded, use as-is
            secret_decoded = self.api_secret.encode('utf-8')

        signature = hmac.new(
            secret_decoded,
            sha256_hash,
            hashlib.sha512
        ).digest()

        # Base64 encode signature
        signature_b64 = base64.b64encode(signature).decode('utf-8')

        return nonce, signature_b64

    def _get_headers(
        self,
        endpoint: str,
        post_data: str = "",
        nonce: str = None
    ) -> Dict[str, str]:
        """Get headers for authenticated request."""
        nonce, signature = self._sign_request(endpoint, post_data, nonce)

        return {
            'Content-Type': 'application/x-www-form-urlencoded',
            'APIKey': self.api_key,
            'Nonce': nonce,
            'Authent': signature,
        }

    def _public_request(
        self,
        endpoint: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make a public (unauthenticated) GET request.

        Args:
            endpoint: API endpoint (e.g., '/tickers')
            params: Query parameters

        Returns:
            JSON response
        """
        url = f"{self.base_url}{self.API_PATH}{endpoint}"

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in public request to {endpoint}: {e}")
            return {'result': 'error', 'error': str(e)}

    def _private_request(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        method: str = 'POST'
    ) -> Dict[str, Any]:
        """
        Make a private (authenticated) request.

        Args:
            endpoint: API endpoint (e.g., '/sendorder')
            data: POST data
            method: HTTP method ('POST' or 'GET')

        Returns:
            JSON response
        """
        url = f"{self.base_url}{self.API_PATH}{endpoint}"

        # Prepare POST data
        post_data = ""
        if data:
            post_data = urllib.parse.urlencode(data)

        # Full endpoint path for signing
        full_endpoint = f"{self.API_PATH}{endpoint}"
        headers = self._get_headers(full_endpoint, post_data)

        try:
            if method == 'POST':
                response = self.session.post(
                    url,
                    data=data,
                    headers=headers,
                    timeout=self.timeout
                )
            else:
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout
                )

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in private request to {endpoint}: {e}")
            return {'result': 'error', 'error': str(e)}

    # ==================== Account Info ====================

    def get_accounts(self) -> Dict[str, Any]:
        """
        Get account information including balances and margin.

        Returns:
            Account info dict with keys:
            - fi_ethusd: ETH futures account info
            - cash: Available cash balance
            - portfolioValue: Total portfolio value
            - initialMargin: Used margin
            - availableMargin: Available margin
        """
        result = self._private_request('/accounts', method='GET')

        if result.get('result') == 'success':
            return result.get('accounts', {})
        return {}

    def get_wallets(self) -> List[Dict[str, Any]]:
        """
        Get wallet balances.

        Returns:
            List of wallet dicts with balance info
        """
        result = self._private_request('/accounts', method='GET')

        if result.get('result') == 'success':
            accounts = result.get('accounts', {})
            wallets = []

            # Extract cash wallet
            if 'cash' in accounts:
                wallets.append({
                    'currency': 'USD',
                    'balance': accounts.get('cash', {}).get('balance', 0),
                    'available': accounts.get('cash', {}).get('availableMargin', 0)
                })

            # Extract futures accounts
            for key, value in accounts.items():
                if key.startswith('fi_') or key.startswith('pi_'):
                    wallets.append({
                        'currency': key,
                        'balance': value.get('balance', 0),
                        'pnl': value.get('pnl', 0),
                        'margin': value.get('initialMargin', 0)
                    })

            return wallets
        return []

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary with balances and positions.

        Returns:
            Dict with:
            - equity: Total account equity
            - available_margin: Available for trading
            - used_margin: Margin in use
            - portfolio_value: Total portfolio value
            - pnl: Unrealized P&L
        """
        accounts = self.get_accounts()

        if not accounts:
            return {}

        # Get cash account
        cash = accounts.get('cash', {})

        # Get ETH futures account if exists
        fi_eth = accounts.get('fi_ethusd', {})

        return {
            'equity': cash.get('portfolioValue', 0),
            'available_margin': cash.get('availableMargin', 0),
            'used_margin': cash.get('initialMargin', 0),
            'portfolio_value': cash.get('portfolioValue', 0),
            'pnl': fi_eth.get('pnl', 0),
            'balance': cash.get('balance', 0)
        }

    # ==================== Positions ====================

    def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        result = self._private_request('/openpositions', method='GET')

        if result.get('result') == 'success':
            positions = []
            for pos in result.get('openPositions', []):
                positions.append(Position(
                    symbol=pos.get('symbol', ''),
                    side='long' if pos.get('side') == 'long' else 'short',
                    size=abs(float(pos.get('size', 0))),
                    entry_price=float(pos.get('price', 0)),
                    mark_price=float(pos.get('markPrice', 0)),
                    liquidation_price=float(pos.get('liquidationThreshold', 0)),
                    unrealized_pnl=float(pos.get('pnl', 0)),
                    margin_used=float(pos.get('initialMargin', 0)),
                    leverage=float(pos.get('leverage', 1))
                ))
            return positions
        return []

    def get_position(self, symbol: str = None) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)

        Returns:
            Position object or None
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        positions = self.get_positions()
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None

    # ==================== Market Data ====================

    def get_tickers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all ticker information.

        Returns:
            Dict mapping symbol to ticker data
        """
        result = self._public_request('/tickers')

        if result.get('result') == 'success':
            tickers = {}
            for ticker in result.get('tickers', []):
                symbol = ticker.get('symbol', '')
                tickers[symbol] = {
                    'symbol': symbol,
                    'bid': float(ticker.get('bid', 0)),
                    'ask': float(ticker.get('ask', 0)),
                    'last': float(ticker.get('last', 0)),
                    'mark_price': float(ticker.get('markPrice', 0)),
                    'index_price': float(ticker.get('indexPrice', 0)),
                    'volume_24h': float(ticker.get('vol24h', 0)),
                    'open_interest': float(ticker.get('openInterest', 0)),
                    'funding_rate': float(ticker.get('fundingRate', 0)),
                }
            return tickers
        return {}

    def get_ticker(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """
        Get ticker for a specific symbol.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)

        Returns:
            Ticker dict or None
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        tickers = self.get_tickers()
        return tickers.get(symbol.upper())

    def get_quote(self, symbol: str = None) -> Optional[Quote]:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)

        Returns:
            Quote object or None
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        ticker = self.get_ticker(symbol)
        if ticker:
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else ticker.get('last', 0)

            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                timestamp=datetime.now(UTC)
            )
        return None

    def get_orderbook(
        self,
        symbol: str = None,
        depth: int = 10
    ) -> Dict[str, List[List[float]]]:
        """
        Get order book for a symbol.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)
            depth: Number of levels

        Returns:
            Dict with 'bids' and 'asks' lists [[price, size], ...]
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        result = self._public_request('/orderbook', params={'symbol': symbol})

        if result.get('result') == 'success':
            orderbook = result.get('orderBook', {})
            return {
                'bids': [[float(b[0]), float(b[1])] for b in orderbook.get('bids', [])[:depth]],
                'asks': [[float(a[0]), float(a[1])] for a in orderbook.get('asks', [])[:depth]]
            }
        return {'bids': [], 'asks': []}

    def get_ohlc(
        self,
        symbol: str = None,
        interval: str = '1m',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Bar]:
        """
        Get OHLC (candlestick) data.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)
            interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start: Start datetime (default: 24 hours ago)
            end: End datetime (default: now)

        Returns:
            List of Bar objects
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        # Map interval to Kraken format
        interval_map = {
            '1m': '1m', '1min': '1m', '1Min': '1m',
            '5m': '5m', '5min': '5m', '5Min': '5m',
            '15m': '15m', '15min': '15m', '15Min': '15m',
            '1h': '1h', '1hour': '1h', '1Hour': '1h',
            '4h': '4h', '4hour': '4h',
            '1d': '1d', '1day': '1d', '1Day': '1d',
        }
        kraken_interval = interval_map.get(interval, '1m')

        # Kraken uses tick types instead of traditional OHLC endpoint
        # We'll use the candles endpoint
        params = {
            'symbol': symbol,
            'interval': kraken_interval,
        }

        if start:
            params['from'] = int(start.timestamp())
        if end:
            params['to'] = int(end.timestamp())

        # Use the history endpoint for OHLC
        result = self._public_request('/history', params=params)

        if result.get('result') == 'success':
            bars = []
            candles = result.get('candles', [])

            for candle in candles:
                bars.append(Bar(
                    symbol=symbol,
                    open=float(candle.get('open', 0)),
                    high=float(candle.get('high', 0)),
                    low=float(candle.get('low', 0)),
                    close=float(candle.get('close', 0)),
                    volume=float(candle.get('volume', 0)),
                    timestamp=datetime.fromtimestamp(
                        candle.get('time', 0) / 1000, tz=UTC
                    )
                ))
            return bars

        # Fallback: Try alternative endpoint
        return self._get_ohlc_fallback(symbol, kraken_interval, start, end)

    def _get_ohlc_fallback(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> List[Bar]:
        """
        Fallback method for OHLC data using charts endpoint.
        """
        params = {
            'symbol': symbol,
            'tick_type': 'trade',
            'resolution': interval,
        }

        if start:
            params['from'] = int(start.timestamp())
        if end:
            params['to'] = int(end.timestamp())

        result = self._public_request('/charts', params=params)

        bars = []
        if result.get('result') == 'success':
            candles = result.get('candles', [])
            for candle in candles:
                bars.append(Bar(
                    symbol=symbol,
                    open=float(candle.get('open', 0)),
                    high=float(candle.get('high', 0)),
                    low=float(candle.get('low', 0)),
                    close=float(candle.get('close', 0)),
                    volume=float(candle.get('volume', 0)),
                    timestamp=datetime.fromtimestamp(
                        candle.get('time', 0) / 1000, tz=UTC
                    )
                ))
        return bars

    def get_bars(
        self,
        symbol: str = None,
        timeframe: str = '1Min',
        limit: int = 100
    ) -> List[Bar]:
        """
        Get historical bars (convenience method matching Alpaca interface).

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour')
            limit: Number of bars to fetch

        Returns:
            List of Bar objects
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        # Map timeframe
        tf_map = {
            '1Min': '1m', '5Min': '5m', '15Min': '15m',
            '1Hour': '1h', '4Hour': '4h', '1Day': '1d'
        }
        interval = tf_map.get(timeframe, '1m')

        # Calculate start time based on limit
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15,
            '1h': 60, '4h': 240, '1d': 1440
        }
        minutes = interval_minutes.get(interval, 1) * limit

        start = datetime.now(UTC) - timedelta(minutes=minutes)

        return self.get_ohlc(symbol, interval, start)

    # ==================== Order Management ====================

    def submit_order(
        self,
        symbol: str = None,
        side: str = 'buy',
        order_type: str = 'mkt',
        size: float = 0,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        post_only: bool = False
    ) -> Dict[str, Any]:
        """
        Submit an order.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)
            side: 'buy' or 'sell'
            order_type: 'mkt', 'lmt', 'stp', 'take_profit'
            size: Order size in contracts
            limit_price: Limit price (required for 'lmt' orders)
            stop_price: Stop/trigger price (required for 'stp' orders)
            reduce_only: If True, only reduce position
            post_only: If True, order must be maker only

        Returns:
            Order response dict
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        data = {
            'orderType': order_type,
            'symbol': symbol,
            'side': side,
            'size': size,
        }

        if limit_price is not None:
            data['limitPrice'] = limit_price

        if stop_price is not None:
            data['stopPrice'] = stop_price

        if reduce_only:
            data['reduceOnly'] = 'true'

        if post_only:
            data['postOnly'] = 'true'

        return self._private_request('/sendorder', data=data)

    def submit_market_order(
        self,
        symbol: str = None,
        side: str = 'buy',
        size: float = 0,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Submit a market order.

        Args:
            symbol: Futures symbol
            side: 'buy' or 'sell'
            size: Order size in contracts
            reduce_only: If True, only reduce position

        Returns:
            Order response
        """
        return self.submit_order(
            symbol=symbol,
            side=side,
            order_type='mkt',
            size=size,
            reduce_only=reduce_only
        )

    def submit_limit_order(
        self,
        symbol: str = None,
        side: str = 'buy',
        size: float = 0,
        limit_price: float = 0,
        reduce_only: bool = False,
        post_only: bool = False
    ) -> Dict[str, Any]:
        """
        Submit a limit order.

        Args:
            symbol: Futures symbol
            side: 'buy' or 'sell'
            size: Order size in contracts
            limit_price: Limit price
            reduce_only: If True, only reduce position
            post_only: If True, order must be maker only

        Returns:
            Order response
        """
        return self.submit_order(
            symbol=symbol,
            side=side,
            order_type='lmt',
            size=size,
            limit_price=limit_price,
            reduce_only=reduce_only,
            post_only=post_only
        )

    def submit_stop_order(
        self,
        symbol: str = None,
        side: str = 'sell',
        size: float = 0,
        stop_price: float = 0,
        limit_price: Optional[float] = None,
        reduce_only: bool = True
    ) -> Dict[str, Any]:
        """
        Submit a stop-loss order.

        Args:
            symbol: Futures symbol
            side: 'buy' or 'sell'
            size: Order size in contracts
            stop_price: Trigger price
            limit_price: Optional limit price (if None, market order on trigger)
            reduce_only: If True, only reduce position

        Returns:
            Order response
        """
        order_type = 'stp' if limit_price is None else 'stop_limit'

        return self.submit_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            stop_price=stop_price,
            limit_price=limit_price,
            reduce_only=reduce_only
        )

    def submit_take_profit_order(
        self,
        symbol: str = None,
        side: str = 'sell',
        size: float = 0,
        limit_price: float = 0,
        reduce_only: bool = True
    ) -> Dict[str, Any]:
        """
        Submit a take-profit order.

        Args:
            symbol: Futures symbol
            side: 'buy' or 'sell'
            size: Order size in contracts
            limit_price: Take profit price
            reduce_only: If True, only reduce position

        Returns:
            Order response
        """
        return self.submit_order(
            symbol=symbol,
            side=side,
            order_type='take_profit',
            size=size,
            limit_price=limit_price,
            reduce_only=reduce_only
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order by ID.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancel response
        """
        return self._private_request('/cancelorder', data={'order_id': order_id})

    def cancel_all_orders(self, symbol: str = None) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Args:
            symbol: Optional - only cancel orders for this symbol

        Returns:
            Cancel response
        """
        data = {}
        if symbol:
            data['symbol'] = symbol

        return self._private_request('/cancelallorders', data=data)

    def get_orders(self, symbol: str = None) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Optional - filter by symbol

        Returns:
            List of Order objects
        """
        result = self._private_request('/openorders', method='GET')

        if result.get('result') == 'success':
            orders = []
            for o in result.get('openOrders', []):
                if symbol and o.get('symbol', '').upper() != symbol.upper():
                    continue

                orders.append(Order(
                    order_id=o.get('order_id', ''),
                    symbol=o.get('symbol', ''),
                    side=o.get('side', ''),
                    order_type=o.get('orderType', ''),
                    size=float(o.get('size', 0)),
                    limit_price=float(o.get('limitPrice', 0)) if o.get('limitPrice') else None,
                    stop_price=float(o.get('stopPrice', 0)) if o.get('stopPrice') else None,
                    filled_size=float(o.get('filledSize', 0)),
                    status='open',
                    timestamp=datetime.fromtimestamp(
                        o.get('receivedTime', 0) / 1000, tz=UTC
                    )
                ))
            return orders
        return []

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get a specific order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None
        """
        orders = self.get_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None

    # ==================== Trade History ====================

    def get_fills(
        self,
        symbol: str = None,
        start: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade fills/executions.

        Args:
            symbol: Optional symbol filter
            start: Start datetime
            limit: Max number of fills

        Returns:
            List of fill dicts
        """
        data = {}
        if start:
            data['lastFillTime'] = int(start.timestamp() * 1000)

        result = self._private_request('/fills', data=data, method='GET')

        if result.get('result') == 'success':
            fills = []
            for fill in result.get('fills', [])[:limit]:
                if symbol and fill.get('symbol', '').upper() != symbol.upper():
                    continue
                fills.append({
                    'fill_id': fill.get('fill_id', ''),
                    'order_id': fill.get('order_id', ''),
                    'symbol': fill.get('symbol', ''),
                    'side': fill.get('side', ''),
                    'size': float(fill.get('size', 0)),
                    'price': float(fill.get('price', 0)),
                    'fee': float(fill.get('fee', 0)),
                    'timestamp': datetime.fromtimestamp(
                        fill.get('fillTime', 0) / 1000, tz=UTC
                    )
                })
            return fills
        return []

    # ==================== Leverage ====================

    def set_leverage(self, symbol: str = None, leverage: float = 5) -> Dict[str, Any]:
        """
        Set leverage for a symbol.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)
            leverage: Leverage multiplier (1-50 typically)

        Returns:
            Response dict
        """
        if symbol is None:
            symbol = self.ETH_PERPETUAL

        # Note: Kraken Futures leverage is set per account, not per position
        # This may vary based on API version
        return self._private_request(
            '/leveragepreferences',
            data={'symbol': symbol, 'maxLeverage': leverage}
        )

    def get_leverage(self, symbol: str = None) -> float:
        """
        Get current leverage setting.

        Args:
            symbol: Futures symbol (default: PI_ETHUSD)

        Returns:
            Current leverage multiplier
        """
        position = self.get_position(symbol)
        if position:
            return position.leverage
        return 1.0

    # ==================== Utility Methods ====================

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            tickers = self.get_tickers()
            if tickers:
                print(f"Connected to Kraken Futures ({'Demo' if self.demo else 'Live'})")
                print(f"  Available symbols: {len(tickers)}")
                if self.ETH_PERPETUAL in tickers:
                    eth = tickers[self.ETH_PERPETUAL]
                    print(f"  ETH Price: ${eth.get('last', 0):,.2f}")
                return True
            return False
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    def test_auth(self) -> bool:
        """Test authenticated API access."""
        try:
            accounts = self.get_accounts()
            if accounts:
                summary = self.get_account_summary()
                print(f"Authenticated to Kraken Futures ({'Demo' if self.demo else 'Live'})")
                print(f"  Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}")
                print(f"  Available Margin: ${summary.get('available_margin', 0):,.2f}")
                return True
            return False
        except Exception as e:
            print(f"Auth test failed: {e}")
            return False


def test_kraken_client(api_key: str, api_secret: str, demo: bool = True) -> bool:
    """
    Test Kraken Futures client connection.

    Args:
        api_key: API key
        api_secret: API secret
        demo: Use demo environment

    Returns:
        True if successful
    """
    client = KrakenFuturesClient(api_key, api_secret, demo=demo)

    print("\n=== Testing Kraken Futures Client ===")
    print(f"Environment: {'Demo' if demo else 'Production'}")
    print(f"Base URL: {client.base_url}")

    # Test public endpoints
    print("\n--- Public API Test ---")
    if client.test_connection():
        quote = client.get_quote()
        if quote:
            print(f"  ETH Quote: Bid=${quote.bid:.2f} Ask=${quote.ask:.2f}")

    # Test private endpoints
    print("\n--- Private API Test ---")
    client.test_auth()

    return True
