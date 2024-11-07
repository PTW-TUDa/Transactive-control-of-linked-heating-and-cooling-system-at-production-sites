"""
messages passed within multi agent system
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "message passed within multi agent system"


def order_msg(
        sender_id,
        reciever_id,
        order_type,
        product_type,
        product_lead_time,
        quantity,
        price,
        min_acceptance_ratio,
        coupled_order,
        id):
    """
    order message sent by traders to markets

    Args:
        sender_id (str): trader name
        reciever_id (str): market name
        order_type (str): order type, e.g. sell or buy
        product_type (int): product type classified by product duration in seconds
        product_lead_time (int): lead time before product execution in seconds
        quantity (float): traded quantity in kWh
        price (float): bid price in €/kWh
        min_acceptance_ratio (float): ratio of quantity for which clearing is accepted
        coupled_order (list): list of uuids od coupled orders which should be deleted if bid is cleared
        id (uuid): unique uuid of order message

    Returns:
        msg (dict): order message holding relevant information as dictionary
    """

    msg = {
        'type': 'order_msg',
        'sender_id': sender_id,
        'reciever_id': reciever_id,
        'order_type': order_type,
        'product_type': product_type,
        'product_lead_time': product_lead_time,
        'quantity': quantity,
        'price': price,
        'min_acceptance_ratio': min_acceptance_ratio,
        'coupled_order': coupled_order,
        'id': id
    }
    return msg


def trade_msg(sender_id, reciever_id, product_type, product_lead_time, trade_type, quantity, price):
    """
    trade message sent by markets to traders

    Args:
        sender_id (str): market name
        reciever_id (str): trader name
        product_type (int): product type classified by product duration in seconds
        product_lead_time (int): lead time before product execution in seconds
        trade_type (str): trade type, e.g. sell or buy
        quantity (float): cleared quantity in kWh
        price (float): clearing price in €/kWh

    Returns:
        msg (dict): trade message holding relevant information as dictionary
    """

    msg = {
        'type': 'trade_msg',
        'sender_id': sender_id,
        'reciever_id': reciever_id,
        'trade_type': trade_type,
        'product_type': product_type,
        'product_lead_time': product_lead_time,
        'quantity': quantity,
        'price': price
    }
    return msg


def balancing_energy_msg(sender_id, reciever_id, system_id, price_pos, price_neg):
    """
    balacing energy message sent by system operators to traders

    Args:
        sender_id (str): system operator name, usually buffer storage
        reciever_id (str): trader name
        system_id (str): supply system name
        price_pos (float): price for positive balancing energy in €/kWh (higher consumption/less production)
        price_neg (float): price for negative balancing energy in €/kWh (less consumption/higher production)

    Returns:
        msg (dict): balancing energy message holding relevant information as dictionary
    """

    msg = {
        'type': 'balancing_energy_msg',
        'sender_id': sender_id,
        'reciever_id': reciever_id,
        'system_id': system_id,
        'price_pos': price_pos,
        'price_neg': price_neg
    }
    return msg


if __name__ == '__main__':
    pass
