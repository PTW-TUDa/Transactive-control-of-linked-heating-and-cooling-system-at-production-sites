"""
models for market clearing
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "models for market clearing"

import pandas as pd
from copy import deepcopy
from multi_agent_system.base.messages import trade_msg
pd.options.mode.chained_assignment = None  # default='warn'


def __double_auction_clearing(market_id, buys, sells):
    """
    clearing within double auction

    Args:
        market_id (str): name of market
        buys (dict): dictionary with buy orders
        sells (dict): dictionary with sell orders

    Returns:
        trades (list): list with trade messages
    """

    trades = []
    for sell_id, sell in sells.copy().items():
        # continue loop if sell was deleted
        if sell_id not in sells:
            continue
        break_loop = False

        # iterate over all buys
        for buy_id, buy in buys.copy().items():

            # continue loop if sell was deleted
            if buy_id not in buys:
                continue

            # stop iteration over buys if price condition is not fulfilled
            if buy['price'] < sell['price']:
                break

            # buy can be fully cleared against sell
            if buy['rest_quantity'] <= sell['rest_quantity']:
                clearing_price = (buy['price']+sell['price'])/2
                clearing_quantity = buy['rest_quantity']
                buy['rest_quantity'] -= clearing_quantity
                sell['rest_quantity'] -= clearing_quantity

                # check for linked buy orders and delete them
                coupled_order = buy['coupled_order']
                try:
                    for order in coupled_order:
                        del buys[order]
                except (KeyError, TypeError):
                    pass
                del buys[buy_id]

            # sell can be fully cleared against buy
            elif buy['rest_quantity'] > sell['rest_quantity']:
                clearing_price = (buy['price']+sell['price'])/2
                clearing_quantity = sell['rest_quantity']
                buy['rest_quantity'] -= clearing_quantity
                sell['rest_quantity'] -= clearing_quantity

                # check for linked sell orders and delete them
                coupled_order = sell['coupled_order']
                try:
                    for order in coupled_order:
                        del sells[order]
                except (KeyError, TypeError):
                    pass
                del sells[sell_id]
                break_loop = True

            # generate buyer msg
            msg_buyer = trade_msg(
                sender_id=market_id,
                reciever_id=buy['sender_id'],
                trade_type='buy',
                product_type=buy['product_type'],
                product_lead_time=buy['product_lead_time'],
                quantity=clearing_quantity,
                price=clearing_price)

            # generate seller msg
            msg_seller = trade_msg(
                sender_id=market_id,
                reciever_id=sell['sender_id'],
                trade_type='sell',
                product_type=sell['product_type'],
                product_lead_time=sell['product_lead_time'],
                quantity=clearing_quantity,
                price=clearing_price)
            trades.extend([msg_buyer, msg_seller])

            # break buy loop if sell has been fully cleared
            if break_loop:
                break

    return trades


def double_auction(parameters):
    """
    double auction with pay-as-bid prices considering minimum ratio of acceptance

    Args:
        parameters (dict): standardized parameter dict holding model parameters/inputs according .json file

    Returns:
        trades (list): list with trade messages
    """

    # create dataframe from order book
    market_id = parameters['model_inputs']['market_id']
    order_book = pd.DataFrame(parameters['model_inputs']['order_book'])

    # terminate function if there are no orders at all
    if order_book.empty:
        return []

    # get buys and sells from order book
    buys = order_book[order_book['order_type'] == 'buy']
    sells = order_book[order_book['order_type'] == 'sell']

    # terminate function if there are no buys or sells
    if buys.empty or sells.empty:
        return []

    # sort dataframes by price and set uuid as index
    buys.sort_values(by=['price'], inplace=True, ignore_index=True, ascending=False)
    sells.sort_values(by=['price'], inplace=True, ignore_index=True)
    buys.set_index('id', inplace=True, drop=False)
    sells.set_index('id', inplace=True, drop=False)

    # add column for rest quantity during clearing process
    buys['rest_quantity'] = buys['quantity']
    sells['rest_quantity'] = sells['quantity']

    # create sorted dicts for clearing and set order id as key
    buys_stored = buys.to_dict('index')
    sells_stored = sells.to_dict('index')

    # iterative clearing until rest quantities are accepted by both sides or no buys/sells left
    while bool(buys_stored) or bool(sells_stored):
        # reset buys, sells, trades for every iteration if market cannot be cleared
        buys = deepcopy(buys_stored)
        sells = deepcopy(sells_stored)

        trades = __double_auction_clearing(market_id=market_id, buys=buys, sells=sells)

        sells_accepted = False
        buys_accepted = False

        # check minimum ratio of acceptance for last buy/sell and delete unaccepted bid for next iteration
        try:
            last_sell = list(sells.values())[0]
            if (
                (1-last_sell['rest_quantity']/last_sell['quantity']) >= last_sell['min_acceptance_ratio']) or (
                    (1-last_sell['rest_quantity']/last_sell['quantity']) == 0):
                sells_accepted = True
            else:
                unaccepted_sell_id = last_sell['id']
                del sells_stored[unaccepted_sell_id]
        except IndexError:
            sells_accepted = True

        try:
            last_buy = list(buys.values())[0]
            if (
                (1-last_buy['rest_quantity']/last_buy['quantity']) >= last_buy['min_acceptance_ratio']) or (
                    (1-last_buy['rest_quantity']/last_buy['quantity']) == 0):
                buys_accepted = True
            else:
                unaccepted_buy_id = last_buy['id']
                del buys_stored[unaccepted_buy_id]
        except IndexError:
            buys_accepted = True

        if sells_accepted and buys_accepted:
            return trades

    # no buys or sells left
    return []


def double_auction_uniform_pricing(parameters):
    """
    double auction with uniform prices considering minimum ratio of acceptance

    Args:
        parameters (dict): standardized parameter dict holding model parameters/inputs according .json file

    Returns:
        trades (list): list with trade messages
    """

    trades = double_auction(parameters=parameters)

    if not trades:
        pass
    else:
        # overwrite pay-as-bid prices with uniform price
        uniform_price = trades[-1]['price']
        for trade in trades:
            trade['price'] = uniform_price

    return trades
