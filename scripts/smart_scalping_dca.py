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
        self.maker_fee = Decimal("0.0007")  # 0.07% maker fee
        self.taker_fee = Decimal("0.0121")  # 1.21% taker fee
        self.logger().info("Strategy initialized with config: %s", vars(config))
        self.logger().info(f"Fees - Maker: {float(self.maker_fee * 100)}%, Taker: {float(self.taker_fee * 100)}%")
        # Initialize positions from existing balance
        self.initialize_positions_from_balance()

    def initialize_positions_from_balance(self):
        """Initialize positions based on current balance and position price"""
        connector = self.connectors[self.config.exchange]
        base_balance = connector.get_available_balance(self.config.trading_pair.split("-")[0])
        
        if base_balance > Decimal("0"):
            # Get position information from connector
            position_price = connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice)
            self.logger().info(f"Found existing balance: {base_balance} at estimated price: {position_price}")
            
            # Calculate number of positions based on order_amount
            num_positions = int(base_balance / self.config.order_amount)
            remaining_amount = base_balance % self.config.order_amount
            
            # Add full positions
            for _ in range(num_positions):
                self.positions.append((position_price, self.config.order_amount))
            
            # Add remaining amount if significant
            if remaining_amount > Decimal("0"):
                self.positions.append((position_price, remaining_amount))
                
            self.logger().info(f"Initialized {len(self.positions)} positions from existing balance")
            cost_basis = self.calculate_cost_basis()
            if cost_basis:
                self.logger().info(f"Initial cost basis: {float(cost_basis)}")

    def get_trading_pair_balance(self) -> tuple[Decimal, Decimal]:
        """Get base and quote balance for the trading pair"""
        base, quote = self.config.trading_pair.split("-")
        connector = self.connectors[self.config.exchange]
        base_balance = connector.get_available_balance(base)
        quote_balance = connector.get_available_balance(quote)
        return base_balance, quote_balance

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.logger().info("------- New Tick Started -------")
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
        if not self.positions:
            self.logger().info("No positions to calculate cost basis")
            return None
        total_amount = sum(amount for _, amount in self.positions)
        total_value = sum(price * amount for price, amount in self.positions)
        cost_basis = total_value / total_amount if total_amount > 0 else None
        self.logger().info("Calculated cost basis: %s", float(cost_basis) if cost_basis else None)
        return cost_basis

    def create_proposal(self) -> List[OrderCandidate]:
        proposal = []
        current_price = self.connectors[self.config.exchange].get_price_by_type(
            self.config.trading_pair, 
            PriceType.MidPrice
        )
        self.logger().info("Current market price: %s", float(current_price))
        
        # Get current balances
        base_balance, quote_balance = self.get_trading_pair_balance()
        self.logger().info(f"Current balances - Base: {float(base_balance)}, Quote: {float(quote_balance)}")
        
        # Recalculate positions if they don't match balance
        total_position_amount = sum(amount for _, amount in self.positions)
        if abs(total_position_amount - base_balance) > Decimal("1e-10"):
            self.logger().info("Position amount mismatch detected. Reinitializing positions...")
            self.positions.clear()
            self.initialize_positions_from_balance()
        
        cost_basis = self.calculate_cost_basis()
        
        # Calculate buy orders
        if len(self.positions) < self.config.max_positions:
            positions_to_add = self.config.max_positions - len(self.positions)
            
            # If we have no positions, place the first buy order
            if not self.positions:
                self.logger().info("No positions - Creating initial buy order")
                if quote_balance >= (current_price * self.config.order_amount):
                    buy_order = OrderCandidate(
                        trading_pair=self.config.trading_pair,
                        is_maker=True,
                        order_type=OrderType.LIMIT,
                        order_side=TradeType.BUY,
                        amount=self.config.order_amount,
                        price=current_price * (Decimal("1") - self.config.position_distance)
                    )
                    proposal.append(buy_order)
                    self.logger().info("Creating first buy order at price %s", float(buy_order.price))
                else:
                    self.logger().info(f"Insufficient quote balance ({float(quote_balance)}) for initial position")
            
            # Only DCA if current price is below cost basis
            elif cost_basis is not None and current_price < cost_basis:
                self.logger().info("Price below cost basis - Creating %d DCA orders", positions_to_add)
                lowest_position_price = min(price for price, _ in self.positions)
                
                for i in range(positions_to_add):
                    # Check if we have enough quote balance for this order
                    buy_price = lowest_position_price * (Decimal("1") - self.config.position_distance * (i + 1))
                    required_quote = buy_price * self.config.order_amount
                    
                    if quote_balance < required_quote:
                        self.logger().info(f"Insufficient quote balance for DCA order {i+1}")
                        break
                    
                    # Only add order if it would improve our average position
                    weighted_average = (cost_basis * sum(amount for _, amount in self.positions) + 
                                     buy_price * self.config.order_amount) / (
                                         sum(amount for _, amount in self.positions) + self.config.order_amount)
                    
                    if weighted_average < cost_basis:
                        self.logger().info("Creating DCA buy order %d at price %s (improves average from %s to %s)", 
                                         i + 1, float(buy_price), float(cost_basis), float(weighted_average))
                        buy_order = OrderCandidate(
                            trading_pair=self.config.trading_pair,
                            is_maker=True,
                            order_type=OrderType.LIMIT,
                            order_side=TradeType.BUY,
                            amount=self.config.order_amount,
                            price=buy_price
                        )
                        proposal.append(buy_order)
                    else:
                        self.logger().info("Skipping DCA order at price %s as it would not improve average", 
                                         float(buy_price))

        # Calculate sell orders based on positions
        if cost_basis is not None and base_balance > Decimal("0"):
            # Include maker fees for both entry and exit
            min_profitable_price = cost_basis * (
                Decimal("1") + self.config.min_profitability + self.maker_fee * Decimal("2")
            )
            # Add buffer for potential taker fills
            taker_buffer_price = min_profitable_price * (Decimal("1") + self.taker_fee)
            
            self.logger().info("Creating sell order at price %s (cost basis: %s, with taker buffer: %s)", 
                            float(min_profitable_price), float(cost_basis), float(taker_buffer_price))
            sell_order = OrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=base_balance,
                price=taker_buffer_price  # Use price that ensures profit even with taker fees
            )
            proposal.append(sell_order)

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
            f"{self.config.exchange} at {round(event.price, 2)} "
            f"({'taker' if event.is_taker else 'maker'} fee: "
            f"{float(self.taker_fee * 100 if event.is_taker else self.maker_fee * 100)}%)"
        )
        
        if event.trade_type == TradeType.BUY:
            self.logger().info("Buy order filled - Adding to positions")
            self.logger().info("Previous positions: %s", list(self.positions))
            # Account for fees in position price
            adjusted_price = event.price * (
                Decimal("1") + (self.taker_fee if event.is_taker else self.maker_fee)
            )
            self.positions.append((adjusted_price, event.amount))
            self.logger().info("Updated positions: %s", list(self.positions))
        else:  # SELL
            self.logger().info("Sell order filled - Clearing all positions")
            self.logger().info("Previous positions: %s", list(self.positions))
            self.positions.clear()
            self.logger().info("Positions cleared")
        
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg) 