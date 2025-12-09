"""
Interactive Brokers API Client for Paper Trading

Handles:
- Market data streaming (stocks and options)
- Paper trading order execution
- Account information
- Option chain retrieval
- Alpaca fallback for market data when IB data unavailable

Requires: ib_insync library
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass
import pytz
import time as time_module

try:
    from ib_insync import IB, Stock, Option, Contract, Order, MarketOrder, LimitOrder, StopOrder
    from ib_insync import util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("Warning: ib_insync not installed. Run: pip install ib_insync")

# Try to import Alpaca for fallback market data
try:
    from .alpaca_client import AlpacaClient
    ALPACA_FALLBACK_AVAILABLE = True
except ImportError:
    ALPACA_FALLBACK_AVAILABLE = False

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


@dataclass
class OptionGreeks:
    """Represents option Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    implied_volatility: float = 0.0


class IBClient:
    """
    Client for Interactive Brokers API supporting both trading and market data.

    Uses ib_insync for TWS/IB Gateway connection.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        """
        Initialize IB client.

        Args:
            host: TWS/IB Gateway host (default localhost)
            port: TWS/IB Gateway port (7497 for paper, 7496 for live)
            client_id: Unique client ID for this connection
        """
        if not IB_AVAILABLE:
            raise ImportError("ib_insync package not installed. Run: pip install ib_insync")

        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False

        # Cache for contracts
        self._contract_cache: Dict[str, Contract] = {}
        self._option_chain_cache: Dict[str, List[Contract]] = {}

        # Alpaca fallback client for market data
        self._alpaca_client: Optional[AlpacaClient] = None
        self._alpaca_fallback_enabled = False
        if ALPACA_FALLBACK_AVAILABLE:
            try:
                from ..config import PaperTradingConfig
                config = PaperTradingConfig.load()
                if config.is_configured():
                    self._alpaca_client = AlpacaClient(
                        api_key=config.api_key,
                        api_secret=config.api_secret,
                        paper=True
                    )
                    self._alpaca_fallback_enabled = True
                    print("Alpaca fallback for market data: ENABLED")
            except Exception as e:
                print(f"Alpaca fallback not available: {e}")

    def connect(self) -> bool:
        """Connect to TWS/IB Gateway."""
        try:
            if self.connected:
                return True

            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            print(f"Connected to IB on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        """Disconnect from TWS/IB Gateway."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("Disconnected from IB")

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            return {}

        account_values = self.ib.accountValues()
        info = {}
        for av in account_values:
            if av.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'AvailableFunds']:
                info[av.tag] = float(av.value)
        return info

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        if not self.connected:
            return []

        positions = []
        for pos in self.ib.positions():
            positions.append({
                'symbol': pos.contract.symbol,
                'quantity': pos.position,
                'avg_cost': pos.avgCost,
                'market_value': pos.position * pos.avgCost,
                'contract': pos.contract
            })
        return positions

    def _get_stock_contract(self, symbol: str) -> Contract:
        """Get or create a stock contract."""
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]

        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        self._contract_cache[symbol] = contract
        return contract

    def _get_option_contract(self, symbol: str, expiry: str, strike: float, right: str) -> Optional[Contract]:
        """
        Get or create an option contract.

        Args:
            symbol: Underlying symbol (e.g., 'MARA')
            expiry: Expiration date YYYYMMDD format
            strike: Strike price
            right: 'C' for call, 'P' for put
        """
        cache_key = f"{symbol}_{expiry}_{strike}_{right}"
        if cache_key in self._contract_cache:
            return self._contract_cache[cache_key]

        try:
            contract = Option(symbol, expiry, strike, right, 'SMART')
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                self._contract_cache[cache_key] = contract
                return contract
        except Exception as e:
            print(f"Error qualifying option contract: {e}")
        return None

    def get_latest_stock_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for a stock. Falls back to Alpaca if IB data unavailable."""
        if not self.connected:
            return None

        try:
            contract = self._get_stock_contract(symbol)
            ticker = self.ib.reqMktData(contract, '', False, False)

            # Wait for data with timeout - IB needs time to stream data
            for _ in range(10):  # Try for up to 5 seconds
                self.ib.sleep(0.5)
                if (ticker.bid and ticker.bid > 0) or (ticker.last and ticker.last > 0):
                    break

            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            last = ticker.last if ticker.last and ticker.last > 0 else 0.0
            close = ticker.close if ticker.close and ticker.close > 0 else 0.0

            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            elif last > 0:
                mid = last
            elif close > 0:
                mid = close
            else:
                mid = 0.0

            self.ib.cancelMktData(contract)

            # If no IB data, try Alpaca fallback
            if mid == 0.0 and self._alpaca_fallback_enabled and self._alpaca_client:
                print(f"IB data unavailable for {symbol}, using Alpaca fallback...")
                return self._get_alpaca_stock_quote(symbol)

            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                timestamp=datetime.now(EST)
            )
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            # Try Alpaca fallback on error
            if self._alpaca_fallback_enabled and self._alpaca_client:
                print(f"Trying Alpaca fallback for {symbol}...")
                return self._get_alpaca_stock_quote(symbol)
        return None

    def _get_alpaca_stock_quote(self, symbol: str) -> Optional[Quote]:
        """Get stock quote from Alpaca as fallback."""
        if not self._alpaca_client:
            return None

        try:
            alpaca_quote = self._alpaca_client.get_latest_stock_quote(symbol)
            if alpaca_quote:
                return Quote(
                    symbol=symbol,
                    bid=alpaca_quote.bid,
                    ask=alpaca_quote.ask,
                    mid=alpaca_quote.mid,
                    timestamp=datetime.now(EST)
                )
        except Exception as e:
            print(f"Alpaca fallback error for {symbol}: {e}")
        return None

    def get_latest_option_quote(self, symbol: str, expiry: str, strike: float, right: str) -> Optional[Quote]:
        """Get latest quote for an option. Falls back to Alpaca if IB data unavailable."""
        if not self.connected:
            return None

        # Build OCC option symbol for Alpaca fallback
        # Format: MARA251213C00013000 (symbol + YYMMDD + C/P + strike*1000 padded to 8 digits)
        expiry_formatted = expiry[2:]  # Convert YYYYMMDD to YYMMDD
        occ_symbol = f"{symbol}{expiry_formatted}{right}{int(strike*1000):08d}"

        try:
            contract = self._get_option_contract(symbol, expiry, strike, right)
            if not contract:
                # Try Alpaca fallback if contract not found
                if self._alpaca_fallback_enabled and self._alpaca_client:
                    print(f"IB option contract not found, trying Alpaca for {occ_symbol}...")
                    return self._get_alpaca_option_quote(occ_symbol)
                return None

            ticker = self.ib.reqMktData(contract, '', False, False)

            # Wait for data with timeout
            for _ in range(6):  # Try for up to 3 seconds
                self.ib.sleep(0.5)
                if (ticker.bid and ticker.bid > 0) or (ticker.last and ticker.last > 0):
                    break

            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0

            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            elif ticker.last and ticker.last > 0:
                mid = ticker.last
            else:
                mid = 0.0

            self.ib.cancelMktData(contract)

            # If no IB data, try Alpaca fallback
            if mid == 0.0 and self._alpaca_fallback_enabled and self._alpaca_client:
                print(f"IB option data unavailable for {occ_symbol}, using Alpaca fallback...")
                return self._get_alpaca_option_quote(occ_symbol)

            return Quote(
                symbol=occ_symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                timestamp=datetime.now(EST)
            )
        except Exception as e:
            print(f"Error getting option quote: {e}")
            # Try Alpaca fallback on error
            if self._alpaca_fallback_enabled and self._alpaca_client:
                print(f"Trying Alpaca fallback for {occ_symbol}...")
                return self._get_alpaca_option_quote(occ_symbol)
        return None

    def _get_alpaca_option_quote(self, occ_symbol: str) -> Optional[Quote]:
        """Get option quote from Alpaca as fallback."""
        if not self._alpaca_client:
            return None

        try:
            alpaca_quote = self._alpaca_client.get_latest_option_quote(occ_symbol)
            if alpaca_quote:
                return Quote(
                    symbol=occ_symbol,
                    bid=alpaca_quote.bid,
                    ask=alpaca_quote.ask,
                    mid=alpaca_quote.mid,
                    timestamp=datetime.now(EST)
                )
        except Exception as e:
            print(f"Alpaca fallback error for {occ_symbol}: {e}")
        return None

    def get_option_greeks(self, symbol: str, expiry: str, strike: float, right: str) -> Optional[OptionGreeks]:
        """Get option Greeks."""
        if not self.connected:
            return None

        try:
            contract = self._get_option_contract(symbol, expiry, strike, right)
            if not contract:
                return None

            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1.0)  # Greeks need a bit more time

            if ticker.modelGreeks:
                return OptionGreeks(
                    delta=ticker.modelGreeks.delta or 0.0,
                    gamma=ticker.modelGreeks.gamma or 0.0,
                    theta=ticker.modelGreeks.theta or 0.0,
                    vega=ticker.modelGreeks.vega or 0.0,
                    implied_volatility=ticker.modelGreeks.impliedVol or 0.0
                )

            self.ib.cancelMktData(contract)
        except Exception as e:
            print(f"Error getting option Greeks: {e}")
        return None

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = '1 min',
        duration: str = '1 D',
        end_time: datetime = None
    ) -> List[Bar]:
        """
        Get historical bars for a stock. Falls back to Alpaca if IB data unavailable.

        Args:
            symbol: Stock symbol
            timeframe: Bar size ('1 min', '5 mins', '1 hour', etc.)
            duration: Duration string ('1 D', '1 W', etc.)
            end_time: End datetime (default: now)
        """
        if not self.connected:
            return []

        try:
            contract = self._get_stock_contract(symbol)
            end_dt = end_time or datetime.now()

            bars_data = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            bars = []
            for bar in bars_data:
                bars.append(Bar(
                    symbol=symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=int(bar.volume),
                    timestamp=bar.date if isinstance(bar.date, datetime) else datetime.now(EST)
                ))

            # If no IB bars, try Alpaca fallback
            if len(bars) == 0 and self._alpaca_fallback_enabled and self._alpaca_client:
                print(f"IB bars unavailable for {symbol}, using Alpaca fallback...")
                return self._get_alpaca_stock_bars(symbol, timeframe)

            return bars
        except Exception as e:
            print(f"Error getting bars for {symbol}: {e}")
            # Try Alpaca fallback on error
            if self._alpaca_fallback_enabled and self._alpaca_client:
                print(f"Trying Alpaca fallback for {symbol} bars...")
                return self._get_alpaca_stock_bars(symbol, timeframe)
        return []

    def _get_alpaca_stock_bars(self, symbol: str, timeframe: str = '1 min') -> List[Bar]:
        """Get stock bars from Alpaca as fallback."""
        if not self._alpaca_client:
            return []

        try:
            # Convert IB timeframe to Alpaca timeframe
            if timeframe == '1 min':
                alpaca_tf = '1Min'
            elif timeframe == '5 mins':
                alpaca_tf = '5Min'
            elif timeframe == '15 mins':
                alpaca_tf = '15Min'
            elif timeframe == '1 hour':
                alpaca_tf = '1Hour'
            else:
                alpaca_tf = '1Min'

            alpaca_bars = self._alpaca_client.get_stock_bars(symbol, timeframe=alpaca_tf, limit=100)
            if alpaca_bars:
                bars = []
                for ab in alpaca_bars:
                    bars.append(Bar(
                        symbol=symbol,
                        open=ab.open,
                        high=ab.high,
                        low=ab.low,
                        close=ab.close,
                        volume=int(ab.volume),
                        timestamp=ab.timestamp if hasattr(ab, 'timestamp') else datetime.now(EST)
                    ))
                return bars
        except Exception as e:
            print(f"Alpaca fallback error for {symbol} bars: {e}")
        return []

    def get_option_chain(self, symbol: str, expiry_date: str = None) -> List[Dict]:
        """
        Get option chain for a symbol.

        Args:
            symbol: Underlying symbol
            expiry_date: Optional specific expiry (YYYYMMDD format)

        Returns:
            List of option contracts with strike, expiry, right
        """
        if not self.connected:
            return []

        cache_key = f"{symbol}_{expiry_date or 'all'}"
        if cache_key in self._option_chain_cache:
            return self._option_chain_cache[cache_key]

        try:
            contract = self._get_stock_contract(symbol)
            chains = self.ib.reqSecDefOptParams(
                contract.symbol,
                '',
                contract.secType,
                contract.conId
            )

            options = []
            for chain in chains:
                if chain.exchange != 'SMART':
                    continue

                for expiry in chain.expirations:
                    if expiry_date and expiry != expiry_date:
                        continue

                    for strike in chain.strikes:
                        options.append({
                            'symbol': symbol,
                            'expiry': expiry,
                            'strike': strike,
                            'rights': ['C', 'P']
                        })

            self._option_chain_cache[cache_key] = options
            return options
        except Exception as e:
            print(f"Error getting option chain: {e}")
        return []

    def find_atm_option(self, symbol: str, underlying_price: float, expiry: str, right: str = 'C') -> Optional[Dict]:
        """
        Find the ATM option for a given underlying price.

        Args:
            symbol: Underlying symbol
            underlying_price: Current underlying price
            expiry: Expiration date YYYYMMDD
            right: 'C' for call, 'P' for put

        Returns:
            Dict with option details or None
        """
        chain = self.get_option_chain(symbol, expiry)
        if not chain:
            return None

        # Find closest strike
        best_match = None
        best_diff = float('inf')

        for opt in chain:
            if opt['expiry'] != expiry:
                continue

            diff = abs(opt['strike'] - underlying_price)
            if diff < best_diff:
                best_diff = diff
                best_match = {
                    'symbol': symbol,
                    'expiry': expiry,
                    'strike': opt['strike'],
                    'right': right,
                    'underlying_price': underlying_price
                }

        return best_match

    # ========== ORDER METHODS ==========

    def submit_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,  # 'BUY' or 'SELL'
        is_option: bool = False,
        expiry: str = None,
        strike: float = None,
        right: str = None
    ) -> Optional[str]:
        """
        Submit a market order.

        Returns order ID or None if failed.
        """
        if not self.connected:
            return None

        try:
            if is_option:
                contract = self._get_option_contract(symbol, expiry, strike, right)
            else:
                contract = self._get_stock_contract(symbol)

            if not contract:
                return None

            order = MarketOrder(side, quantity)
            trade = self.ib.placeOrder(contract, order)

            # Wait for order to be acknowledged
            self.ib.sleep(0.5)

            return str(trade.order.orderId)
        except Exception as e:
            print(f"Error submitting market order: {e}")
        return None

    def submit_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: float,
        is_option: bool = False,
        expiry: str = None,
        strike: float = None,
        right: str = None
    ) -> Optional[str]:
        """Submit a limit order."""
        if not self.connected:
            return None

        try:
            if is_option:
                contract = self._get_option_contract(symbol, expiry, strike, right)
            else:
                contract = self._get_stock_contract(symbol)

            if not contract:
                return None

            order = LimitOrder(side, quantity, limit_price)
            trade = self.ib.placeOrder(contract, order)

            self.ib.sleep(0.5)

            return str(trade.order.orderId)
        except Exception as e:
            print(f"Error submitting limit order: {e}")
        return None

    def submit_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: float,
        is_option: bool = False,
        expiry: str = None,
        strike: float = None,
        right: str = None
    ) -> Optional[str]:
        """Submit a stop order."""
        if not self.connected:
            return None

        try:
            if is_option:
                contract = self._get_option_contract(symbol, expiry, strike, right)
            else:
                contract = self._get_stock_contract(symbol)

            if not contract:
                return None

            order = StopOrder(side, quantity, stop_price)
            trade = self.ib.placeOrder(contract, order)

            self.ib.sleep(0.5)

            return str(trade.order.orderId)
        except Exception as e:
            print(f"Error submitting stop order: {e}")
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        if not self.connected:
            return False

        try:
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    return True
        except Exception as e:
            print(f"Error canceling order: {e}")
        return False

    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status."""
        if not self.connected:
            return None

        try:
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    return trade.orderStatus.status
        except Exception as e:
            print(f"Error getting order status: {e}")
        return None

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        if not self.connected:
            return []

        orders = []
        for trade in self.ib.openTrades():
            orders.append({
                'order_id': str(trade.order.orderId),
                'symbol': trade.contract.symbol,
                'side': trade.order.action,
                'quantity': trade.order.totalQuantity,
                'order_type': trade.order.orderType,
                'status': trade.orderStatus.status
            })
        return orders
