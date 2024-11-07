"""
market agent
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "market agent"

import numpy as np
from multi_agent_system.base.base_agent import BaseAgent
import multi_agent_system.models.market_models as market_models  # noqa: F401


class Market(BaseAgent):
    """
    class for markets

    Args:
        BaseAgent (class): extends BaseAgent
    """

    def setup_agent(self):
        """
        initally executed setup method
        """

        # initialize order books by lead times and products
        self.order_books = {product_type: None for product_type in list(self.products.keys())}
        for order_book in self.order_books:
            self.order_books[order_book] = {lead_time: [] for lead_time in self.products[order_book]}

    def process_msg(self, msg):
        """
        implements message processing

        Args:
            msg (dict): message object
        """

        # append only bids which do not hold zero quantities to order book
        if msg['type'] == 'order_msg' and msg['quantity'] != 0:
            self.__process_order(msg)

    def __process_order(self, msg):
        """
        implements processing of order messages

        Args:
            order_msg (dict): message object
        """
        self.order_books[msg['product_type']][msg['product_lead_time']].append(msg)

    def __logging(self, experiment_time, trades, product):
        """
        log market clearing informations

        Args:
            experiment_time (datetime): clearing time
            trades (list): list with trade messages
            product (int): cleared product
        """
        # iterate only over buys (sells result would be the same)
        quantities = np.array([trade['quantity'] for trade in trades if trade['trade_type'] == 'buy'])
        prices = np.array([trade['price'] for trade in trades if trade['trade_type'] == 'buy'])
        total_quantity = np.sum(quantities)
        if total_quantity > 0:
            mean_price = np.sum(prices * quantities) / total_quantity
        else:
            mean_price = None

        log = {
            'time': experiment_time,
            'product': product,
            'quantity': total_quantity,
            'price': mean_price
        }
        self.trading_table_longtime.append(log)

    def clear(self, product, experiment_time):
        """
        method to execute market clearing process

        Args:
            product (int): product to be cleared
            experiment_time (datetime): clearing time

        Returns:
            msgs (list): list of successful clearing results as list of trade messages
        """
        # get order_book
        order_book = self.order_books[product['product_type']][product['lead_time']]

        # market model
        market_model_inputs = {"order_book": order_book, "market_id": self.name}

        market_attr = {  # noqa: F841
            'model_parameters': self.agent_config['pricing_config']['model_parameters'],
            'model_inputs': market_model_inputs,
        }
        model_type = "market_models." + self.agent_config['pricing_config']['model_type']
        msgs = eval(model_type + "(market_attr)")  # returns list with trade messages

        # delete values from order books after clearing
        self.order_books[product['product_type']][product['lead_time']].clear()

        self.__logging(experiment_time=experiment_time, trades=msgs, product=product)
        return msgs
