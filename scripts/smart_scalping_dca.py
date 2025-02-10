import logging
import os
from collections import deque
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class SmartScalpingDCAConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    exchange: str = Field("coinbase_advanced_trade", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange where the bot will trade"))
    trading_pair: str = Field("ATOM-USD", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair in which the bot will place orders"))
    order_amount: Decimal = Field(15, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order amount (denominated in base asset)"))
    min_profitability: Decimal = Field(0.002, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum profitability target (1.0 = 100%)"))
    max_positions: int = Field(5, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximum number of open positions"))
    order_refresh_time: int = Field(15, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order refresh time (in seconds)"))
    position_distance: Decimal = Field(0.002, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Distance between positions (1.0 = 100%)"))


class SmartScalpingDCA(ScriptStrategyBase):
    """
    A smart scalping strategy that:
    1. Places buy orders at strategic price levels below the current price
    2. Updates cost basis when orders are filled
    3. Places sell orders at profitable levels considering fees
    4. Uses DCA to manage multiple positions
    5. Ensures minimum profitability accounting for maker fees
    """

    markets = {}
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: SmartScalpingDCAConfig):
        super().__init__(connectors)
        self.config = config
        self.create_timestamp = 0
        self.positions = deque(maxlen=self.config.max_positions)
        self.maker_fee = Decimal("0.0006")  # 0.06%
        self.taker_fee = Decimal("0.0120")  # 1.20%
        self.min_gas = Decimal("0.02")  # Minimum SOL to keep for gas
        self.logger().info("Strategy initialized with config: %s", vars(config))
        
    @classmethod
    def init_markets(cls, config: SmartScalpingDCAConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.logger().info("------- New Tick Started -------")
            
            # Check current balance and update positions if needed
            connector = self.connectors[self.config.exchange]
            base_balance = connector.get_available_balance(self.config.trading_pair.split("-")[0])
            
            if base_balance == Decimal("0"):
                if self.positions:
                    self.logger().info("No balance found - clearing positions")
                    self.positions.clear()
            elif not self.positions:
                # Initialize position with current balance and market price
                current_price = connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice)
                self.positions.append((current_price, base_balance))
                self.logger().info(f"Initialized position from balance: {base_balance} @ {current_price}")
            elif abs(base_balance - sum(amount for _, amount in self.positions)) > Decimal("1e-8"):
                # Balance changed - update position size
                cost_basis = self.calculate_cost_basis()
                if cost_basis:
                    self.positions.clear()
                    self.positions.append((cost_basis, base_balance))
                    self.logger().info(f"Updated position to match balance: {base_balance} @ {cost_basis}")
            
            self.logger().info("Current positions: %s", list(self.positions))
            
            self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal()
            self.logger().info("Created proposal: %s", 
                            [(o.order_side, float(o.price), float(o.amount)) for o in proposal])
            
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.logger().info("Adjusted proposal: %s", 
                            [(o.order_side, float(o.price), float(o.amount)) for o in proposal_adjusted])
            
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.config.order_refresh_time + self.current_timestamp
            self.logger().info("Next tick scheduled for timestamp: %s", self.create_timestamp)
            self.logger().info("------- Tick Completed -------\n")

    def calculate_cost_basis(self) -> Optional[Decimal]:
        """Calculate cost basis from positions or recent trades, considering only multiples of order_amount"""
        if not self.positions:
            self.logger().info("No positions to calculate cost basis")
            return None
            
        # First try from positions
        total_amount = sum(amount for _, amount in self.positions)
        
        # Calculate how many complete units we have
        num_units = int(total_amount / self.config.order_amount)
        if num_units == 0:
            self.logger().info("Position size %s too small for minimum order amount %s", 
                             float(total_amount), float(self.config.order_amount))
            return None
            
        # Only consider complete units for cost basis
        considered_amount = Decimal("0")
        considered_value = Decimal("0")
        remaining_units = num_units
        
        # Process positions from newest to oldest
        for price, amount in reversed(self.positions):
            if remaining_units <= 0:
                break
                
            # Calculate how many complete units we can take from this position
            position_units = int(amount / self.config.order_amount)
            units_to_take = min(position_units, remaining_units)
            
            if units_to_take > 0:
                amount_to_take = units_to_take * self.config.order_amount
                considered_amount += amount_to_take
                considered_value += amount_to_take * price
                remaining_units -= units_to_take
        
        if considered_amount > 0:
            cost_basis = considered_value / considered_amount
            self.logger().info("Calculated cost basis from positions: %s (using %s/%s total amount)", 
                             float(cost_basis), float(considered_amount), float(total_amount))
            return cost_basis
            
        # If no valid positions, try from recent trades
        connector = self.connectors[self.config.exchange]
        trades = connector.get_my_trades(self.config.trading_pair)
        
        buy_value = Decimal("0")
        buy_amount = Decimal("0")
        
        # Process trades from newest to oldest
        for trade in reversed(trades):
            if trade.trade_type == TradeType.BUY:
                # Only consider amounts that contribute to complete units
                current_total = buy_amount + trade.amount
                current_units = int(current_total / self.config.order_amount)
                previous_units = int(buy_amount / self.config.order_amount)
                
                if current_units > previous_units:
                    # Calculate the portion of this trade that contributes to complete units
                    usable_amount = (current_units * self.config.order_amount) - buy_amount
                    usable_amount = min(usable_amount, trade.amount)
                    
                    buy_amount += usable_amount
                    buy_value += usable_amount * trade.price
                    
                    # If we have enough for our current position, stop processing trades
                    if buy_amount >= (num_units * self.config.order_amount):
                        break
        
        if buy_amount >= self.config.order_amount:
            cost_basis = buy_value / buy_amount
            self.logger().info("Calculated cost basis from trades: %s (using %s total amount)", 
                             float(cost_basis), float(buy_amount))
            return cost_basis
            
        return None

    def create_proposal(self) -> List[OrderCandidate]:
        proposal = []
        current_price = self.connectors[self.config.exchange].get_price_by_type(
            self.config.trading_pair, 
            PriceType.MidPrice
        )
        self.logger().info("Current market price: %s", float(current_price))
        
        cost_basis = self.calculate_cost_basis()
        
        # First priority: Create sell orders for existing positions
        if cost_basis is not None and self.positions:
            min_profitable_price = cost_basis * (
                Decimal("1") + self.config.min_profitability + self.maker_fee + self.taker_fee
            )
            total_position_amount = sum(amount for _, amount in self.positions)
            
            # Calculate sellable amount as multiple of order_amount
            num_units = int(total_position_amount / self.config.order_amount)
            sellable_amount = num_units * self.config.order_amount
            
            if sellable_amount > Decimal("0"):
                self.logger().info("Creating sell order at price %s (cost basis: %s) for amount %s (from total %s)", 
                                float(min_profitable_price), float(cost_basis), 
                                float(sellable_amount), float(total_position_amount))
                sell_order = OrderCandidate(
                    trading_pair=self.config.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.SELL,
                    amount=sellable_amount,
                    price=min_profitable_price
                )
                proposal.append(sell_order)
            else:
                self.logger().info("Position size %s too small for minimum order amount %s", 
                                float(total_position_amount), float(self.config.order_amount))
            return proposal  # Return early - don't place buy orders while we have positions to sell
        
        # Second priority: Place buy orders only if price has dropped enough
        if len(self.positions) < self.config.max_positions:
            # Check available balance for buying
            connector = self.connectors[self.config.exchange]
            base_balance = connector.get_available_balance(self.config.trading_pair.split("-")[0])
            
            # Only proceed if we have enough balance for at least one order_amount plus gas
            if base_balance >= (self.config.order_amount + self.min_gas):
                lowest_price = None
                if self.positions:
                    lowest_price = min(price for price, _ in self.positions)
                    target_buy_price = lowest_price * (Decimal("1") - self.config.position_distance)
                    self.logger().info("Lowest position price: %s, target buy price: %s", 
                                    float(lowest_price), float(target_buy_price))
                else:
                    target_buy_price = current_price * (Decimal("1") - self.config.position_distance)
                    self.logger().info("No positions - target buy price: %s", float(target_buy_price))
                
                # Only place buy order if current price is at or below target price
                if current_price <= target_buy_price:
                    self.logger().info("Current price %s at or below target %s - creating buy order", 
                                    float(current_price), float(target_buy_price))
                    buy_order = OrderCandidate(
                        trading_pair=self.config.trading_pair,
                        is_maker=True,
                        order_type=OrderType.LIMIT,
                        order_side=TradeType.BUY,
                        amount=self.config.order_amount,
                        price=current_price
                    )
                    proposal.append(buy_order)
                else:
                    self.logger().info("Current price %s above target %s - no buy orders", 
                                    float(current_price), float(target_buy_price))
            else:
                self.logger().info("Insufficient balance %s for order amount %s plus gas %s", 
                                float(base_balance), float(self.config.order_amount), float(self.min_gas))

        return proposal

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connectors[self.config.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.config.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )

    def cancel_all_orders(self):
        active_orders = self.get_active_orders(connector_name=self.config.exchange)
        self.logger().info("Cancelling %d active orders", len(active_orders))
        for order in active_orders:
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)
            self.logger().info("Cancelled order %s", order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (
            f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} "
            f"{self.config.exchange} at {round(event.price, 2)}"
        )
        
        if event.trade_type == TradeType.BUY:
            self.logger().info("Buy order filled - Adding to positions")
            self.logger().info("Previous positions: %s", list(self.positions))
            self.positions.append((event.price, event.amount))
            self.logger().info("Updated positions: %s", list(self.positions))
        else:  # SELL
            self.logger().info("Sell order filled - Clearing all positions")
            self.logger().info("Previous positions: %s", list(self.positions))
            self.positions.clear()
            self.logger().info("Positions cleared")
        
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg) 