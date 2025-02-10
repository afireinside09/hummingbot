import logging
from decimal import Decimal
from typing import Dict, List

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class AtomUsdScalpDCAStrategy(ScriptStrategyBase):
    markets = {"coinbase_advanced_trade": {"ATOM-USD"}}
    
    # Strategy parameters
    trading_pair = "ATOM-USD"
    exchange = "coinbase_advanced_trade"
    
    # Position management
    max_order_amount_usd = Decimal("35.0")  # Maximum USD per order
    min_profitability = Decimal("0.0050")   # 0.50% minimum profit target to cover 0.07% maker fee
    max_positions = 5                        # Maximum number of DCA positions
    dca_step_size = Decimal("0.02")         # 2% price decrease for each DCA level
    
    # Tracking variables
    active_buy_orders: List[LimitOrder] = []
    active_sell_orders: List[LimitOrder] = []
    dca_positions: Dict[float, Decimal] = {}  # Price -> Amount mapping
    
    # Add new tracking variables
    total_position_value: Decimal = Decimal("0")
    total_position_amount: Decimal = Decimal("0")
    
    def __init__(self, connectors: Dict):
        super().__init__(connectors)
        self.ready = False
    
    def on_start(self):
        self.ready = True
        self.logger().info("Strategy started - Trading ATOM-USD with DCA and dynamic basis adjustment")
        self.logger().info(f"Initial parameters: Max order size=${self.max_order_amount_usd}, "
                        f"Profit target={self.min_profitability*100}%, Max positions={self.max_positions}")
    
    def on_stop(self):
        self.cancel_all_orders()
        self.ready = False
    
    def on_tick(self):
        if not self.ready:
            return
        
        # Get current market prices
        mid_price = self.get_mid_price(self.exchange, self.trading_pair)
        
        if not mid_price:
            return
        
        # Manage existing positions
        self.manage_positions(mid_price)
        
        # Place new orders if we have capacity
        if len(self.dca_positions) < self.max_positions:
            self.place_orders(mid_price)
    
    def manage_positions(self, current_price: Decimal):
        """Manage existing positions and place sell orders"""
        self.logger().info(f"Managing positions - Current price: ${current_price}")
        
        # Cancel existing sell orders if price has moved significantly
        for order in self.active_sell_orders:
            if abs(float(order.price) - float(current_price)) / float(current_price) > 0.02:
                self.logger().info(f"Cancelling outdated sell order at ${order.price} due to price movement")
                self.cancel(self.exchange, self.trading_pair, order.client_order_id)
        
        for entry_price, amount in list(self.dca_positions.items()):
            target_sell_price = Decimal(entry_price) * (1 + self.min_profitability)
            self.logger().info(f"Position: {amount} ATOM @ ${entry_price} - Target sell: ${target_sell_price}")
            
            existing_sell = False
            for order in self.active_sell_orders:
                if abs(float(order.price) - float(target_sell_price)) < 0.01:
                    existing_sell = True
                    break
            
            if not existing_sell:
                self.logger().info(f"Placing new sell order: {amount} ATOM @ ${target_sell_price}")
                self.place_sell_order(
                    self.exchange,
                    self.trading_pair,
                    amount,
                    target_sell_price,
                )
    
    def place_orders(self, current_price: Decimal):
        """Place new buy orders for DCA"""
        available_balance = self.get_available_balance("USD")
        self.logger().info(f"Checking for new buy opportunities - Available USD: ${available_balance}")
        
        if available_balance < self.max_order_amount_usd:
            self.logger().info("Insufficient balance for new positions")
            return
        
        buy_price = current_price * Decimal("0.999")
        self.logger().info(f"Initial buy price target: ${buy_price} (0.1% below market)")
        
        if self.dca_positions:
            highest_entry = max(self.dca_positions.keys())
            self.logger().info(f"Existing position detected - Highest entry: ${highest_entry}")
            
            if float(current_price) < highest_entry * (1 - float(self.dca_step_size)):
                buy_price = Decimal(str(highest_entry)) * (1 - self.dca_step_size)
                self.logger().info(f"DCA opportunity - New buy price: ${buy_price} "
                               f"({self.dca_step_size*100}% below highest entry)")
        
        order_amount = self.max_order_amount_usd / buy_price
        self.logger().info(f"Placing buy order: {order_amount} ATOM @ ${buy_price}")
        
        self.place_buy_order(
            self.exchange,
            self.trading_pair,
            order_amount,
            buy_price,
        )
    
    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order filled events"""
        if event.trade_type == TradeType.BUY:
            self.logger().info(f"Buy order filled: {event.amount} ATOM @ ${event.price}")
            
            position_value = Decimal(str(event.amount)) * Decimal(str(event.price))
            self.total_position_value += position_value
            self.total_position_amount += Decimal(str(event.amount))
            
            avg_entry_price = self.total_position_value / self.total_position_amount
            self.logger().info(f"Updated position - Total amount: {self.total_position_amount} ATOM, "
                           f"Average entry: ${avg_entry_price}")
            
            self.dca_positions[float(avg_entry_price)] = self.total_position_amount
            
            # Remove old entry prices
            entry_prices = list(self.dca_positions.keys())
            for price in entry_prices:
                if price != float(avg_entry_price):
                    self.logger().info(f"Consolidating position - Removing entry at ${price}")
                    del self.dca_positions[price]
            
            self.logger().info("Cancelling existing sell orders to update targets")
            self.cancel_all_orders()
            
            self.active_buy_orders = [
                order for order in self.active_buy_orders 
                if order.client_order_id != event.order_id
            ]
            
        else:  # SELL
            self.logger().info(f"Sell order filled: {event.amount} ATOM @ ${event.price}")
            sold_amount = Decimal(str(event.amount))
            
            for price in list(self.dca_positions.keys()):
                if abs(price - float(event.price)) / price < 0.02:
                    self.total_position_amount -= sold_amount
                    self.total_position_value -= sold_amount * Decimal(str(price))
                    
                    if self.total_position_amount <= Decimal("0"):
                        self.logger().info("Position fully closed - Resetting tracking variables")
                        self.total_position_amount = Decimal("0")
                        self.total_position_value = Decimal("0")
                        del self.dca_positions[price]
                    else:
                        self.logger().info(f"Partial position close - Remaining: {self.total_position_amount} ATOM")
                        self.dca_positions[price] = self.total_position_amount
                    break
            
            self.active_sell_orders = [
                order for order in self.active_sell_orders
                if order.client_order_id != event.order_id
            ]
    
    def place_buy_order(self, exchange: str, trading_pair: str, amount: Decimal, price: Decimal):
        """Place a buy order and track it"""
        order = self.buy(exchange, trading_pair, amount, OrderType.LIMIT, price)
        if order:
            self.active_buy_orders.append(order)
    
    def place_sell_order(self, exchange: str, trading_pair: str, amount: Decimal, price: Decimal):
        """Place a sell order and track it"""
        order = self.sell(exchange, trading_pair, amount, OrderType.LIMIT, price)
        if order:
            self.active_sell_orders.append(order)
    
    def cancel_all_orders(self):
        """Cancel all active orders"""
        for exchange in self.markets:
            for trading_pair in self.markets[exchange]:
                self.cancel_all(exchange, trading_pair)
        self.active_buy_orders = []
        self.active_sell_orders = [] 