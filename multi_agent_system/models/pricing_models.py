"""
models for pricing assessment
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "models for pricing assessment"

import uuid
import numpy as np
from multi_agent_system.base.messages import order_msg


def cool_producer_pricing(parameters):
    """
    pricing assessment for cooling converter

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """

    duration_hours = parameters['model_inputs']['product_type']/3600
    quantities = np.array(parameters['model_inputs']['quantities']['thermal_energy'])

    cost_electricity = np.array(parameters['model_inputs']['quantities']['electric_energy'])*parameters[
        'model_inputs']['electricity_price']
    cost_operating_hours = parameters['model_parameters']['cost_operating_hours']*duration_hours
    cost_ramp_up = (not parameters['model_inputs']['is_running'])*parameters['model_parameters']['cost_ramp_up']

    # negative buy price
    total_costs = -cost_electricity-cost_operating_hours-cost_ramp_up

    # consider minimum ratio of acceptance regading the minimal
    max_quantity = max(quantities)
    min_acceptance_ratios = []
    for quantity in quantities:
        if quantity*parameters['model_parameters']['min_acceptance_ratio'] >= (
                max_quantity*parameters['model_parameters']['minimal_load']):
            min_acceptance_ratios.append(parameters['model_parameters']['min_acceptance_ratio'])
        else:
            if quantity != 0:
                min_acceptance_ratios.append(min(
                    1,
                    (parameters['model_parameters']['minimal_load']*max_quantity)/quantity))
            else:
                min_acceptance_ratios.append(0)
    # calculate prices for quantities unequal to zero, return 0 for zero bids
    prices = np.minimum(
        np.divide(total_costs, quantities, out=np.zeros_like(total_costs), where=quantities != 0),
        parameters['model_inputs']['positive_market_limit'])
    prices = np.maximum(prices, parameters['model_inputs']['negative_market_limit'])
    uuids = [uuid.uuid4() for _ in range(len(quantities))]

    # identify unexecutable orders if a specific order is excuted because of maximum load
    coupled_orders = []
    for idx, quantity in enumerate(quantities):
        quantities_add = quantities + quantity
        # find indices of quantitites + considered order larger than max quantity
        idx_coupled_orders = np.nonzero(quantities_add > max_quantity)
        # delete self coupling
        idx_coupled_orders = idx_coupled_orders[0][idx_coupled_orders[0] != idx]
        coupled_orders.append([uuids[idx] for idx in idx_coupled_orders])

    msgs = []
    for idx in range(len(quantities)):
        msgs.append(order_msg(
            sender_id=parameters['model_parameters']['name'],
            reciever_id=parameters['model_parameters']['markets'][0],
            order_type='buy',
            product_type=parameters['model_inputs']['product_type'],
            product_lead_time=parameters['model_inputs']['product_lead_time'],
            quantity=abs(quantities[idx]),
            price=prices[idx],
            min_acceptance_ratio=min_acceptance_ratios[idx],
            coupled_order=coupled_orders[idx],
            id=uuids[idx]
        ))

    return msgs


def heat_producer_pricing(parameters):
    """
    pricing assessment for heating converter

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """

    # insert trading model here
    duration_hours = parameters['model_inputs']['product_type']/3600
    quantities = np.array(parameters['model_inputs']['quantities']['thermal_energy'])

    cost_fuel = np.array(parameters['model_inputs']['quantities']['fuel_energy'])*parameters[
        'model_inputs']['fuel_price']

    # demand in external file is negative
    # consider electricity cost only if electricity is produced, e.g. chp
    if np.sum(parameters['model_inputs']['quantities']['electric_energy']) < 0:
        elec_diff = -parameters['model_inputs']['quantities']['electric_energy']+parameters[
            'model_inputs']['electricity_demand']*duration_hours
        cost_electricity = []
        for idx, diff in enumerate(elec_diff):
            if diff > 0:
                cost_electricity.append((parameters['model_inputs']['electricity_demand']*parameters[
                    'model_inputs']['electricity_price']-diff*parameters['model_inputs']['chp_renumeration']))
            else:
                cost_electricity.append((parameters['model_inputs']['quantities']['electric_energy'][idx]*parameters[
                    'model_inputs']['electricity_price']))
    else:
        cost_electricity = [0 for _ in range(len(parameters['model_inputs']['quantities']['electric_energy']))]
    cost_electricity = np.array(cost_electricity)

    cost_operating_hours = parameters['model_parameters']['cost_operating_hours']*duration_hours
    cost_ramp_up = (not parameters['model_inputs']['is_running'])*parameters['model_parameters']['cost_ramp_up']
    total_costs = cost_fuel+cost_electricity+cost_operating_hours+cost_ramp_up

    # consider minimum ratio of acceptance regading the minimal
    max_quantity = max(quantities)
    min_acceptance_ratios = []
    for quantity in quantities:
        if quantity*parameters['model_parameters']['min_acceptance_ratio'] >= max_quantity*parameters[
                'model_parameters']['minimal_load']:
            min_acceptance_ratios.append(parameters['model_parameters']['min_acceptance_ratio'])
        else:
            if quantity != 0:
                min_acceptance_ratios.append(min(
                    1,
                    (parameters['model_parameters']['minimal_load']*max_quantity)/quantity))
            else:
                min_acceptance_ratios.append(0)
    # calculate prices for quantities unequal to zero, return 0 for zero bids
    prices = np.minimum(
        np.divide(total_costs, quantities, out=np.zeros_like(total_costs), where=quantities != 0),
        parameters['model_inputs']['positive_market_limit'])
    prices = np.maximum(prices, parameters['model_inputs']['negative_market_limit'])
    uuids = [uuid.uuid4() for _ in range(len(quantities))]

    # identify unexecutable orders if a specific order is excuted because of maximum load
    coupled_orders = []
    for idx, quantity in enumerate(quantities):
        quantities_add = quantities + quantity
        # find indices of quantitites + considered order larger than max quantity
        idx_coupled_orders = np.nonzero(quantities_add > max_quantity)
        # delete self coupling
        idx_coupled_orders = idx_coupled_orders[0][idx_coupled_orders[0] != idx]
        coupled_orders.append([uuids[idx] for idx in idx_coupled_orders])

    msgs = []
    for idx in range(len(quantities)):
        msgs.append(order_msg(
            sender_id=parameters['model_parameters']['name'],
            reciever_id=parameters['model_parameters']['markets'][0],
            order_type='sell',
            product_type=parameters['model_inputs']['product_type'],
            product_lead_time=parameters['model_inputs']['product_lead_time'],
            quantity=quantities[idx],
            price=prices[idx],
            min_acceptance_ratio=min_acceptance_ratios[idx],
            coupled_order=coupled_orders[idx],
            id=uuids[idx]
        ))

    return msgs


def inherent_storage_pricing(parameters):
    """
    pricing assessment for inherent storages

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    quantities = parameters['model_inputs']['quantities']['thermal_energy']
    diff_quantity = quantities[1]-quantities[0]

    # check if system is connected to 2 market - 0 is heat network
    if len(parameters['model_parameters']['markets']) == 2:
        market_heat = parameters['model_parameters']['markets'][0]
        market_cool = parameters['model_parameters']['markets'][1]
    else:
        market_heat = parameters['model_parameters']['markets'][0]
        market_cool = parameters['model_parameters']['markets'][0]

    # heat demand
    if parameters['model_inputs']['bHeatingMode']:
        diff_quantity_price = np.interp(
            parameters['model_inputs']['quantities']['soc'],
            parameters['model_parameters']['soc_range'],
            [parameters['model_inputs']['positive_market_limit'], 0])
        minimum_quantity_price = parameters['model_inputs']['positive_market_limit']
        market = market_heat
        order_type = 'buy'
    # cool demand
    else:
        diff_quantity_price = np.interp(
            parameters['model_inputs']['quantities']['soc'],
            parameters['model_parameters']['soc_range'],
            [0, parameters['model_inputs']['negative_market_limit']])
        minimum_quantity_price = parameters['model_inputs']['negative_market_limit']
        market = market_cool
        order_type = 'sell'

    if parameters['model_inputs']['shorttime_product']:
        bids = {
            quantities[0]: minimum_quantity_price,
            diff_quantity: diff_quantity_price
        }
    else:
        # one quantity covering demand is traded
        bids = {quantities[0]: minimum_quantity_price}

    msgs = []
    for quantity, price in bids.items():
        msgs.append(order_msg(
            sender_id=parameters['model_parameters']['name'],
            reciever_id=market,
            order_type=order_type,
            product_type=parameters['model_inputs']['product_type'],
            product_lead_time=parameters['model_inputs']['product_lead_time'],
            quantity=abs(quantity),
            price=price,
            min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
            coupled_order=None,
            id=uuid.uuid4()
        ))

    return msgs


def inherent_storage_pricing_one_product(parameters):
    """
    pricing assessment for inherent storages

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    quantities = parameters['model_inputs']['quantities']['thermal_energy']
    diff_quantity = quantities[1]-quantities[0]

    # check if system is connected to 2 market - 0 is heat network
    if len(parameters['model_parameters']['markets']) == 2:
        market_heat = parameters['model_parameters']['markets'][0]
        market_cool = parameters['model_parameters']['markets'][1]
    else:
        market_heat = parameters['model_parameters']['markets'][0]
        market_cool = parameters['model_parameters']['markets'][0]

    # heat demand
    if parameters['model_inputs']['bHeatingMode']:
        diff_quantity_price = np.interp(
            parameters['model_inputs']['quantities']['soc'],
            parameters['model_parameters']['soc_range'],
            [parameters['model_inputs']['positive_market_limit'], 0])
        minimum_quantity_price = parameters['model_inputs']['positive_market_limit']
        market = market_heat
        order_type = 'buy'
    # cool demand
    else:
        diff_quantity_price = np.interp(
            parameters['model_inputs']['quantities']['soc'],
            parameters['model_parameters']['soc_range'],
            [0, parameters['model_inputs']['negative_market_limit']])
        minimum_quantity_price = parameters['model_inputs']['negative_market_limit']
        market = market_cool
        order_type = 'sell'

    bids = {
        quantities[0]: minimum_quantity_price,
        diff_quantity: diff_quantity_price
    }

    msgs = []
    for quantity, price in bids.items():
        msgs.append(order_msg(
            sender_id=parameters['model_parameters']['name'],
            reciever_id=market,
            order_type=order_type,
            product_type=parameters['model_inputs']['product_type'],
            product_lead_time=parameters['model_inputs']['product_lead_time'],
            quantity=abs(quantity),
            price=price,
            min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
            coupled_order=None,
            id=uuid.uuid4()
        ))

    return msgs


def demand_pricing(parameters):
    """
    pricing assessment for demands without inherent storage capacity

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    # waste heat/cool demand
    if parameters['model_inputs']['quantities']['thermal_energy'][0] > 0:
        bids = {
            parameters['model_inputs']['quantities']['thermal_energy'][0] : parameters[
                'model_inputs']['negative_market_limit']}
        order_type = 'sell'
    else:
        bids = {
            parameters['model_inputs']['quantities']['thermal_energy'][0] : parameters[
                'model_inputs']['positive_market_limit']}
        order_type = 'buy'

    msgs = []
    for quantity, price in bids.items():
        msgs.append(order_msg(
            sender_id=parameters['model_parameters']['name'],
            reciever_id=parameters['model_parameters']['markets'][0],
            order_type=order_type,
            product_type=parameters['model_inputs']['product_type'],
            product_lead_time=parameters['model_inputs']['product_lead_time'],
            quantity=abs(quantity),
            price=price,
            min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
            coupled_order=None,
            id=uuid.uuid4()
        ))

    return msgs


def thermal_network_pricing(parameters):
    """
    pricing assessment for thermal network

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    quantities = parameters['model_inputs']['quantities']['thermal_energy']

    # price for complete storage loading -> will be cleared after minimum quantity
    price = np.interp(
        parameters['model_inputs']['quantities']['soc'],
        parameters['model_parameters']['soc_range'],
        [parameters['model_inputs']['positive_market_limit'], parameters['model_inputs']['negative_market_limit']])

    if quantities > 0:
        bids = {abs(quantities): price}
        order_type = 'buy'
    else:
        bids = {abs(quantities): price}
        order_type = 'sell'

    msgs = []
    for quantity, price in bids.items():
        msgs.append(order_msg(
            sender_id=parameters['model_parameters']['name'],
            reciever_id=parameters['model_parameters']['markets'][0],
            order_type=order_type,
            product_type=parameters['model_inputs']['product_type'],
            product_lead_time=parameters['model_inputs']['product_lead_time'],
            quantity=quantity,
            price=price,
            min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
            coupled_order=None,
            id=uuid.uuid4()
        ))

    return msgs


def no_pricing(parameters):
    """
    no pricing assessment

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """

    return []


def storage_pricing(parameters):
    """
    pricing assessment for active storages

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    quantities = parameters['model_inputs']['quantities']

    if not parameters['model_inputs']['shorttime_product']:
        # heat storage
        if parameters['model_parameters']['is_heat_storage']:
            market_limit = parameters['model_inputs']['positive_market_limit']*parameters[
                'model_parameters']['market_limit_ratio']
            # active storage does not trade under 0 (no usage of negative market limit)
            price = np.interp(
                parameters['model_inputs']['quantities']['soc'],
                parameters['model_parameters']['soc_range'],
                [market_limit, 0])
            order_type = 'buy'
        # cold storage
        else:
            market_limit = parameters['model_inputs']['negative_market_limit']*parameters[
                'model_parameters']['market_limit_ratio']
            # active storage does not trade under 0 (no usage of negative market limit)
            price = np.interp(
                parameters['model_inputs']['quantities']['soc'],
                parameters['model_parameters']['soc_range'],
                [0, market_limit])
            order_type = 'sell'
        quantity = np.sum(quantities['thermal_energy'])

    else:
        # buy price from previous trading steps
        stored_energy = quantities['stored_energy']
        costs_stored_energy = np.array([parameters['model_inputs']['energy_costs']], dtype=np.float64)
        price_stored_energy = np.divide(costs_stored_energy, stored_energy, out=np.zeros_like(costs_stored_energy),
                                        where=stored_energy != 0)
        quantity = np.sum(quantities['stored_energy'])+np.sum(quantities['thermal_energy'])
        # heat storage
        if parameters['model_parameters']['is_heat_storage']:
            market_limit = parameters['model_inputs']['positive_market_limit']
            price_traded_energy = parameters['model_inputs']['price_neg']
            order_type = 'sell'
            # calculate price only for quantity unequal to zero
            price = np.mean(
                np.divide((price_stored_energy*stored_energy+price_traded_energy*quantities['thermal_energy']),
                          quantity, out=np.zeros_like(quantities['thermal_energy']), where=quantity != 0))
            price = min(price, market_limit)
        # cold storage
        else:
            market_limit = parameters['model_inputs']['negative_market_limit']
            price_traded_energy = parameters['model_inputs']['price_pos']
            order_type = 'buy'
            # calculate price only for quantity unequal to zero
            price = np.mean(
                np.divide((price_stored_energy*stored_energy+price_traded_energy*quantities['thermal_energy']),
                          quantity, out=np.zeros_like(quantities['thermal_energy']), where=quantity != 0))
            price = max(price, market_limit)

    msgs = []
    msgs.append(order_msg(
        sender_id=parameters['model_parameters']['name'],
        reciever_id=parameters['model_parameters']['markets'][0],
        order_type=order_type,
        product_type=parameters['model_inputs']['product_type'],
        product_lead_time=parameters['model_inputs']['product_lead_time'],
        quantity=abs(quantity),
        price=price,
        min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
        coupled_order=None,
        id=uuid.uuid4()
    ))

    return msgs


def storage_pricing_one_product(parameters):
    """
    pricing assessment for active storages

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    quantities = parameters['model_inputs']['quantities']

    # buy/sell fixed ratio of storage capacity
    # heat storage
    if parameters['model_parameters']['is_heat_storage']:
        market_limit = parameters['model_inputs']['positive_market_limit']*parameters[
            'model_parameters']['market_limit_ratio']
        # active storage does not trade under 0 (no usage of negative market limit)
        price = np.interp(
            parameters['model_inputs']['quantities']['soc'],
            parameters['model_parameters']['soc_range'],
            [market_limit, 0])
        order_type = 'buy'
    # cold storage
    else:
        market_limit = parameters['model_inputs']['negative_market_limit']*parameters[
            'model_parameters']['market_limit_ratio']
        # active storage does not trade under 0 (no usage of negative market limit)
        price = np.interp(
            parameters['model_inputs']['quantities']['soc'],
            parameters['model_parameters']['soc_range'],
            [0, market_limit])
        order_type = 'sell'
    quantity = np.sum(quantities['thermal_energy'])

    thermal_energy_msg = order_msg(
        sender_id=parameters['model_parameters']['name'],
        reciever_id=parameters['model_parameters']['markets'][0],
        order_type=order_type,
        product_type=parameters['model_inputs']['product_type'],
        product_lead_time=parameters['model_inputs']['product_lead_time'],
        quantity=abs(quantity),
        price=price,
        min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
        coupled_order=None,
        id=uuid.uuid4()
        )

    # sell only stored energy
    stored_energy = quantities['stored_energy']
    costs_stored_energy = np.array([parameters['model_inputs']['energy_costs']], dtype=np.float64)
    price_stored_energy = np.divide(costs_stored_energy, stored_energy, out=np.zeros_like(costs_stored_energy),
                                    where=stored_energy != 0)
    quantity = np.sum(quantities['stored_energy'])

    # heat storage
    if parameters['model_parameters']['is_heat_storage']:
        market_limit = parameters['model_inputs']['positive_market_limit']
        order_type = 'sell'
        # calculate price only for quantity unequal to zero
        price = np.mean(
            np.divide((price_stored_energy*stored_energy),
                      quantity, out=np.zeros_like(quantities['stored_energy']), where=quantity != 0))
        price = min(price, market_limit)
    # cold storage
    else:
        market_limit = parameters['model_inputs']['negative_market_limit']
        order_type = 'buy'
        # calculate price only for quantity unequal to zero
        price = np.mean(
            np.divide((price_stored_energy*stored_energy),
                      quantity, out=np.zeros_like(quantities['stored_energy']), where=quantity != 0))
        price = max(price, market_limit)

    stored_energy_msg = order_msg(
        sender_id=parameters['model_parameters']['name'],
        reciever_id=parameters['model_parameters']['markets'][0],
        order_type=order_type,
        product_type=parameters['model_inputs']['product_type'],
        product_lead_time=parameters['model_inputs']['product_lead_time'],
        quantity=abs(quantity),
        price=price,
        min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
        coupled_order=None,
        id=uuid.uuid4()
        )

    msgs = [thermal_energy_msg, stored_energy_msg]
    return msgs


def heat_exchanger_pricing(parameters):
    """
    pricing assessment for heat exchangers

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    bids = []
    # use case 1a - single cold producer
    if not parameters['model_parameters']['additional_producer'] and not parameters[
            'model_parameters']['heating_use_case']:
        bids.append({
            'reciever_id': parameters['model_parameters']['markets'][0],
            'order_type': 'buy',
            'price': parameters['model_parameters']['negative_market_limit']
            })
    # use case 1b - single heat producer
    elif not parameters['model_parameters']['additional_producer'] and parameters[
            'model_parameters']['heating_use_case']:
        bids.append({
            'reciever_id': parameters['model_parameters']['markets'][1],
            'order_type': 'sell',
            'price': parameters['model_inputs']['positive_market_limit']
            })
    # use case 2a - additional cold producer
    # use case 2b - additional heat producer
    else:
        bids.append({
            'reciever_id': parameters['model_parameters']['markets'][0],
            'order_type': 'buy',
            'price': 0
            },
            {
            'reciever_id': parameters['model_parameters']['markets'][1],
            'order_type': 'sell',
            'price': 0
            })

    msgs = []
    for bid in bids:
        msgs.append(order_msg(
            sender_id=parameters['model_parameters']['name'],
            reciever_id=bid['reciever_id'],
            order_type=bid['order_type'],
            product_type=parameters['model_inputs']['product_type'],
            product_lead_time=parameters['model_inputs']['product_lead_time'],
            quantity=abs(parameters['model_inputs']['quantities']['thermal_energy'][0]),
            price=bid['price'],
            min_acceptance_ratio=parameters['model_parameters']['min_acceptance_ratio'],
            coupled_order=None,
            id=uuid.uuid4()
        ))

    return msgs


def heat_pump_pricing(parameters):
    """
    pricing assessment for heat pumps

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (list): list of trade messages
    """
    duration_hours = parameters['model_inputs']['product_type']/3600
    quantities = np.array(parameters['model_inputs']['quantities']['thermal_energy_heat'])

    cost_electricity = np.array(parameters['model_inputs']['quantities']['electric_energy'])*parameters[
        'model_inputs']['electricity_price']
    cost_operating_hours = parameters['model_parameters']['cost_operating_hours']*duration_hours
    cost_ramp_up = (not parameters['model_inputs']['is_running'])*parameters['model_parameters']['cost_ramp_up']
    total_electric_costs = cost_electricity+cost_operating_hours+cost_ramp_up

    # consider minimum ratio of acceptance regading the minimal
    max_quantity = max(quantities)
    min_acceptance_ratios = []
    for quantity in quantities:
        if quantity*parameters['model_parameters']['min_acceptance_ratio'] >= (
                max_quantity*parameters['model_parameters']['minimal_load']):
            min_acceptance_ratios.append(parameters['model_parameters']['min_acceptance_ratio'])
        else:
            if quantity != 0:
                min_acceptance_ratios.append(min(
                    1,
                    (parameters['model_parameters']['minimal_load']*max_quantity)/quantity))
            else:
                min_acceptance_ratios.append(0)
    uuids = [uuid.uuid4() for _ in range(len(quantities))]

    # identify unexecutable orders if a specific order is excuted because of maximum load
    coupled_orders = []
    for idx, quantity in enumerate(quantities):
        quantities_add = quantities + quantity
        # find indices of quantitites + considered order larger than max quantity
        idx_coupled_orders = np.nonzero(quantities_add > max_quantity)
        # delete self coupling
        idx_coupled_orders = idx_coupled_orders[0][idx_coupled_orders[0] != idx]
        coupled_orders.append([uuids[idx] for idx in idx_coupled_orders])

    msgs = []
    # use case 3a - single heat producer
    if not parameters['model_parameters']['additional_producer'] and parameters[
            'model_parameters']['heating_use_case']:
        prices = np.ones(len(quantities))*parameters['model_inputs']['positive_market_limit']

        for idx in range(len(quantities)):
            msgs.append(order_msg(
                sender_id=parameters['model_parameters']['name'],
                reciever_id=parameters['model_parameters']['markets'][0],
                order_type='sell',
                product_type=parameters['model_inputs']['product_type'],
                product_lead_time=parameters['model_inputs']['product_lead_time'],
                quantity=abs(parameters['model_inputs']['quantities']['thermal_energy_heat'][idx]),
                price=prices[idx],
                min_acceptance_ratio=min_acceptance_ratios[idx],
                coupled_order=coupled_orders[idx],
                id=uuids[idx]
            ))

    # use case 3b - single cold producer
    elif not parameters['model_parameters']['additional_producer'] and parameters[
            'model_parameters']['heating_use_case']:
        prices = np.ones(len(quantities))*parameters['model_inputs']['negative_market_limit']

        for idx in range(len(quantities)):
            msgs.append(order_msg(
                sender_id=parameters['model_parameters']['name'],
                reciever_id=parameters['model_parameters']['markets'][1],
                order_type='buy',
                product_type=parameters['model_inputs']['product_type'],
                product_lead_time=parameters['model_inputs']['product_lead_time'],
                quantity=abs(parameters['model_inputs']['quantities']['thermal_energy_cool'][idx]),
                price=prices[idx],
                min_acceptance_ratio=min_acceptance_ratios[idx],
                coupled_order=coupled_orders[idx],
                id=uuids[idx]
            ))

    # use case 4a - additional heat producer or use case 4b - addtional cold producer
    else:
        prices = np.minimum(
            np.divide(total_electric_costs, quantities, out=np.zeros_like(total_electric_costs),
                      where=quantities != 0), parameters['model_inputs']['positive_market_limit'])
        prices = np.maximum(prices, parameters['model_inputs']['negative_market_limit'])

        for idx in range(len(quantities)):
            msgs.append(order_msg(
                sender_id=parameters['model_parameters']['name'],
                reciever_id=parameters['model_parameters']['markets'][0],
                order_type='sell',
                product_type=parameters['model_inputs']['product_type'],
                product_lead_time=parameters['model_inputs']['product_lead_time'],
                quantity=abs(parameters['model_inputs']['quantities']['thermal_energy_heat'][idx]),
                price=prices[idx],
                min_acceptance_ratio=min_acceptance_ratios[idx],
                coupled_order=coupled_orders[idx],
                id=uuids[idx]
            ))
            msgs.append(order_msg(
                sender_id=parameters['model_parameters']['name'],
                reciever_id=parameters['model_parameters']['markets'][1],
                order_type='buy',
                product_type=parameters['model_inputs']['product_type'],
                product_lead_time=parameters['model_inputs']['product_lead_time'],
                quantity=abs(parameters['model_inputs']['quantities']['thermal_energy_cool'][idx]),
                price=0,
                min_acceptance_ratio=min_acceptance_ratios[idx],
                coupled_order=coupled_orders[idx],
                id=uuids[idx]
            ))

    return msgs
