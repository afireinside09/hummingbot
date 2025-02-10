import logging
import os
from decimal import Decimal
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class TechnicalMarketMakingConfig(BaseClientModel):
    """
    Configuration parameters for Technical Market Making strategy
    """
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    
    # Basic parameters
    exchange: str = Field("coinbase_advanced_trade", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange name"))
    trading_pair: str = Field("ATOM-USD", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair"))
    order_amount: Decimal = Field(20, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order amount"))
    
    # Spread parameters
    base_bid_spread: Decimal = Field(0.008, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Base bid spread"))
    base_ask_spread: Decimal = Field(0.008, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Base ask spread"))
    
    # Order refresh parameters
    order_refresh_time: int = Field(30, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order refresh time in seconds"))
    
    # Technical analysis parameters
    bb_length: int = Field(20, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bollinger Bands period"))
    bb_std: float = Field(2.0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bollinger Bands standard deviation"))
    
    macd_fast: int = Field(12, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "MACD fast length"))
    macd_slow: int = Field(26, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "MACD slow length"))
    macd_signal: int = Field(9, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "MACD signal length"))
    
    # Data collection
    price_data_length: int = Field(100, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Number of price data points to maintain"))


class TechnicalMarketMaking(ScriptStrategyBase):
    """
    A market making strategy that adjusts spreads and order sizes based on technical indicators:
    - Bollinger Bands: Adjusts spreads based on price volatility
    - MACD: Adjusts order sizes based on trend strength and direction
    """
    
    markets = {}
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: TechnicalMarketMakingConfig):
        super().__init__(connectors)
        self.config = config
        self.ready_to_trade = False
        self.last_timestamp = 0
        self.price_data = []
        
        # Log configuration details
        self.logger().info("Strategy Configuration:")
        self.logger().info(f"Exchange: {config.exchange}")
        self.logger().info(f"Trading Pair: {config.trading_pair}")
        self.logger().info(f"Order Amount: {config.order_amount}")
        self.logger().info(f"Base Spreads - Bid: {config.base_bid_spread}, Ask: {config.base_ask_spread}")
        self.logger().info(f"Order Refresh Time: {config.order_refresh_time} seconds")
        self.logger().info("\nTechnical Indicators Configuration:")
        self.logger().info(f"Bollinger Bands - Period: {config.bb_length}, StdDev: {config.bb_std}")
        self.logger().info(f"MACD - Fast: {config.macd_fast}, Slow: {config.macd_slow}, Signal: {config.macd_signal}")
        self.logger().info(f"Price Data Length: {config.price_data_length} points")
        
    @classmethod
    def init_markets(cls, config: TechnicalMarketMakingConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    def on_start(self):
        self.logger().info("Strategy started.")
        self.ready_to_trade = False
        
    def on_tick(self):
        """Called on every tick (1 second by default)"""
        current_price = self.connectors[self.config.exchange].get_price_by_type(
            self.config.trading_pair,
            PriceType.MidPrice
        )
        
        self.logger().debug(f"Current mid price: {current_price}")
        
        self.price_data.append(float(current_price))
        if len(self.price_data) > self.config.price_data_length:
            self.price_data.pop(0)
        
        if len(self.price_data) < self.config.price_data_length:
            if not self.ready_to_trade:
                self.logger().info(f"Collecting price data... {len(self.price_data)}/{self.config.price_data_length}")
            return
        
        if not self.ready_to_trade:
            self.logger().info("Price data collection complete. Starting trading operations.")
        self.ready_to_trade = True
        
        if self.last_timestamp <= self.current_timestamp:
            self.logger().debug("Starting order refresh cycle")
            self.cancel_all_orders()
            
            bb_upper, bb_lower = self.calculate_bollinger_bands()
            macd_value, signal_value = self.calculate_macd()
            
            self.logger().debug(f"\nTechnical Indicators:")
            self.logger().debug(f"Bollinger Bands - Upper: {bb_upper:.2f}, Lower: {bb_lower:.2f}")
            self.logger().debug(f"MACD - Value: {macd_value:.4f}, Signal: {signal_value:.4f}")
            
            proposal = self.create_proposal(bb_upper, bb_lower, macd_value, signal_value)
            self.logger().debug(f"Initial proposal created: {[f'{o.order_side} {o.amount:.4f} @ {o.price:.2f}' for o in proposal]}")
            
            adjusted_proposal = self.adjust_proposal_to_budget(proposal)
            self.logger().debug(f"Budget-adjusted proposal: {[f'{o.order_side} {o.amount:.4f} @ {o.price:.2f}' for o in adjusted_proposal]}")
            
            self.place_orders(adjusted_proposal)
            self.last_timestamp = self.current_timestamp + self.config.order_refresh_time

    def calculate_bollinger_bands(self) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        prices = pd.Series(self.price_data)
        sma = prices.rolling(window=self.config.bb_length).mean()
        std = prices.rolling(window=self.config.bb_length).std()
        
        bb_upper = sma + (std * self.config.bb_std)
        bb_lower = sma - (std * self.config.bb_std)
        
        return float(bb_upper.iloc[-1]), float(bb_lower.iloc[-1])

    def calculate_macd(self) -> Tuple[float, float]:
        """Calculate MACD and Signal line"""
        prices = pd.Series(self.price_data)
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.config.macd_slow, adjust=False).mean()
        
        # Calculate MACD and Signal line
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.config.macd_signal, adjust=False).mean()
        
        return float(macd.iloc[-1]), float(signal.iloc[-1])

    def create_proposal(
        self, 
        bb_upper: float, 
        bb_lower: float, 
        macd_value: float, 
        signal_value: float
    ) -> List[OrderCandidate]:
        """Create order proposals based on technical indicators"""
        ref_price = float(self.connectors[self.config.exchange].get_price_by_type(
            self.config.trading_pair,
            PriceType.MidPrice
        ))
        
        # Bollinger Bands spread adjustment
        bb_range = bb_upper - bb_lower
        price_position = (ref_price - bb_lower) / bb_range
        spread_adjustment = abs(price_position - 0.5) * 2
        
        self.logger().debug(f"\nSpread Adjustment Calculation:")
        self.logger().debug(f"Price position in BB range: {price_position:.4f}")
        self.logger().debug(f"Spread adjustment factor: {spread_adjustment:.4f}")
        
        bid_spread = self.config.base_bid_spread * (1 + spread_adjustment)
        ask_spread = self.config.base_ask_spread * (1 + spread_adjustment)
        
        self.logger().debug(f"Adjusted spreads - Bid: {bid_spread:.4f}, Ask: {ask_spread:.4f}")
        
        # MACD size adjustment
        macd_strength = abs(macd_value - signal_value) / abs(signal_value) if signal_value != 0 else 0
        size_adjustment = 1 + (macd_strength * 0.5)
        
        self.logger().debug(f"\nSize Adjustment Calculation:")
        self.logger().debug(f"MACD strength: {macd_strength:.4f}")
        self.logger().debug(f"Size adjustment factor: {size_adjustment:.4f}")
        
        # Adjust order sizes based on trend
        if macd_value > signal_value:
            buy_size = self.config.order_amount * size_adjustment
            sell_size = self.config.order_amount / size_adjustment
            self.logger().debug("MACD indicates uptrend - Increasing buy size, decreasing sell size")
        else:
            buy_size = self.config.order_amount / size_adjustment
            sell_size = self.config.order_amount * size_adjustment
            self.logger().debug("MACD indicates downtrend - Decreasing buy size, increasing sell size")
        
        buy_price = Decimal(str(ref_price)) * (Decimal(1) - Decimal(str(bid_spread)))
        sell_price = Decimal(str(ref_price)) * (Decimal(1) + Decimal(str(ask_spread)))
        
        self.logger().debug(f"\nFinal Order Details:")
        self.logger().debug(f"Buy order: {buy_size:.4f} @ {buy_price:.2f}")
        self.logger().debug(f"Sell order: {sell_size:.4f} @ {sell_price:.2f}")
        
        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal(str(buy_size)),
            price=buy_price
        )
        
        sell_order = OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal(str(sell_size)),
            price=sell_price
        )
        
        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust order sizes to available balance"""
        return self.connectors[self.config.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders from proposal"""
        if not proposal:
            self.logger().warning("No orders to place after budget adjustment")
            return
        
        self.logger().info(f"\nPlacing {len(proposal)} orders:")
        for order in proposal:
            self.logger().info(
                f"Order: {order.order_side} {order.amount:.4f} {order.trading_pair} @ {order.price:.2f}"
            )
            if order.order_side == TradeType.SELL:
                self.sell(
                    connector_name=self.config.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )
            elif order.order_side == TradeType.BUY:
                self.buy(
                    connector_name=self.config.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )

    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        """Log order fills"""
        msg = (
            f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} "
            f"{self.config.exchange} at {round(event.price, 4)}"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg) 