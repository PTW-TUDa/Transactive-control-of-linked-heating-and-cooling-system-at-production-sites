"""
models for quantity assessment
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "models for capacity assessment"

import numpy as np
from scipy.interpolate import griddata


def cooling_utility(parameters):
    """
    utility generating cooling energy

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal and electric energies for bidding process
    """
    # calcalute residual thermal power depending on ambient temperature
    thermal_efficiency = np.interp(
        parameters['model_inputs']['ambient_temperature'],
        parameters['model_parameters']['thermal_efficiencies'][0],
        parameters['model_parameters']['thermal_efficiencies'][1])
    nominal_thermal_power = thermal_efficiency*parameters['model_parameters']['nominal_electric_power']

    # maximum residual power
    cleared_power = max(
        parameters['model_inputs']['cleared_energy_neg'])/(parameters['model_inputs']['trading_time']/3600)
    power_max = nominal_thermal_power - cleared_power
    power_max = np.clip(power_max, 0, nominal_thermal_power)

    # minimum residual power
    if parameters['model_parameters']['minimal_load']*nominal_thermal_power > cleared_power:
        power_min = parameters['model_parameters']['minimal_load']*nominal_thermal_power
    else:
        power_min = 0

    if parameters['model_parameters']['bid_discretization'] != 1:
        thermal_powers = np.linspace(power_min, power_max, num=parameters['model_parameters']['bid_discretization'])
    else:
        thermal_powers = np.array([power_max])

    # consider only sell product allocation
    thermal_powers *= parameters['model_inputs']['product_allocation']

    # calculate electric power
    operating_points = thermal_powers/nominal_thermal_power
    electric_efficiencies = np.interp(
        operating_points,
        parameters['model_parameters']['electric_efficiencies'][0],
        parameters['model_parameters']['electric_efficiencies'][1])
    electric_powers = operating_points*electric_efficiencies*parameters['model_parameters']['nominal_electric_power']

    # calculate energy based on powers and product
    thermal_energy = thermal_powers*parameters['model_inputs']['product_type']/3600
    electric_energy = electric_powers*parameters['model_inputs']['product_type']/3600

    result = {
        'thermal_energy': np.array(thermal_energy, dtype=np.float64),
        'electric_energy': np.array(electric_energy, dtype=np.float64)
        }

    return result


def heating_utility(parameters):
    """
    utility generating heating energy

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal, electric and fuel energies for bidding process
    """
    # calculate residual thermal power depending on return temperature
    thermal_efficiency = np.interp(
        parameters['model_inputs']['fReturnTemperature'],
        parameters['model_parameters']['thermal_efficiencies'][0],
        parameters['model_parameters']['thermal_efficiencies'][1])
    nominal_thermal_power = thermal_efficiency*parameters['model_parameters']['nominal_fuel_power']

    # maximum residual power - trading time in seconds and energy in kWh
    cleared_power = max(
        parameters['model_inputs']['cleared_energy_pos'])/(parameters['model_inputs']['trading_time']/3600)
    power_max = nominal_thermal_power - cleared_power
    power_max = np.clip(power_max, 0, nominal_thermal_power)

    # minimum residual power
    if parameters['model_parameters']['minimal_load']*nominal_thermal_power > cleared_power:
        power_min = parameters['model_parameters']['minimal_load']*nominal_thermal_power
    else:
        power_min = 0

    if parameters['model_parameters']['bid_discretization'] != 1:
        thermal_powers = np.linspace(power_min, power_max, num=parameters['model_parameters']['bid_discretization'])
    else:
        thermal_powers = np.array([power_max])

    # consider product allocation
    thermal_powers *= parameters['model_inputs']['product_allocation']

    # calculate gas power
    fuel_powers = thermal_powers/thermal_efficiency

    # calculate electric power
    operating_points = thermal_powers/nominal_thermal_power
    electric_efficiencies = np.interp(
        operating_points,
        parameters['model_parameters']['electric_efficiencies'][0],
        parameters['model_parameters']['electric_efficiencies'][1])
    electric_powers = -operating_points*electric_efficiencies*parameters['model_parameters']['nominal_fuel_power']

    # calculate energies based on power and product
    thermal_energy = thermal_powers*parameters['model_inputs']['product_type']/3600
    electric_energy = electric_powers*parameters['model_inputs']['product_type']/3600
    fuel_energy = fuel_powers*parameters['model_inputs']['product_type']/3600

    result = {
        'thermal_energy': np.array(thermal_energy, dtype=np.float64),
        'electric_energy': np.array(electric_energy, dtype=np.float64),
        'fuel_energy': np.array(fuel_energy, dtype=np.float64)
        }

    return result


def demand_prescribed(parameters):
    """
    external prescribed heating or cooling demand

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal energy for bidding process
    """
    # positive if waste heat/cool demand, negative if heat demand
    demand = parameters['model_inputs']['demand']
    cleared_power = max(
        np.array(parameters['model_inputs']['cleared_energy_pos'])-np.array(parameters[
            'model_inputs']['cleared_energy_neg']))/(parameters['model_inputs']['trading_time']/3600)

    # consider heating and cooling demands
    if demand > 0:
        thermal_power = max(0, demand-cleared_power)
    else:
        thermal_power = min(0, demand-cleared_power)

    thermal_energy_min = thermal_power*parameters['model_inputs']['product_type']/3600
    thermal_energy = thermal_energy_min * parameters['model_inputs']['product_allocation']

    result = {
        'thermal_energy': np.array([thermal_energy], dtype=np.float64),
        'thermal_energy_min': np.array([thermal_energy_min], dtype=np.float64)
        }
    return result


def demand_building(parameters):
    """
    building demand which is calculated by ambient temperature

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal energies and current soc for bidding process
    """
    actual_temperature = parameters['model_inputs']['fRoomTemperature']
    target_temperature = parameters['model_parameters']['target_temperature']
    mean_temperature = np.mean([target_temperature, actual_temperature])

    # demand means ambient temperature
    demand = (parameters['model_parameters']['heat_capacity']*(target_temperature-actual_temperature))/parameters[
        'model_inputs']['product_type']+(mean_temperature-parameters['model_inputs']['ambient_temperature'])/parameters[
            'model_parameters']['thermal_resistance_to_ambient']
    cleared_power = max(np.array(parameters['model_inputs']['cleared_energy_pos'])-np.array(parameters[
        'model_inputs']['cleared_energy_neg']))/(parameters['model_inputs']['trading_time']/3600)

    # shorttime product
    if parameters['model_inputs']['shorttime_product']:
        residual_power = [0, 0]
    else:
        # residual power - possible [load reduction, load raise]
        residual_power = [
            (parameters['model_parameters']['heat_capacity'] *
             (target_temperature-parameters['model_parameters']['temperature_limits'][0])
             )/parameters['model_inputs']['product_type'],
            (parameters['model_parameters']['heat_capacity'] *
             (parameters['model_parameters']['temperature_limits'][1]-target_temperature)
             )/parameters['model_inputs']['product_type']]

    # determine if system is in heating mode
    if parameters['model_inputs']['bHeatingMode'] == 1:
        power_max = max(demand + residual_power[1] + cleared_power, 0)
        power_min = max(demand - residual_power[0] + cleared_power, 0)
    # cool demand
    else:
        power_max = min(demand - residual_power[0] - cleared_power, 0)
        power_min = min(demand + residual_power[1] - cleared_power, 0)
    thermal_powers = np.array([power_min, power_max])
    thermal_energy_min = thermal_powers[0]*parameters['model_inputs']['product_type']/3600
    thermal_energy = thermal_powers*parameters['model_inputs']['product_type']*parameters[
        'model_inputs']['product_allocation']/3600

    heat_capacity = (
        parameters['model_parameters']['heat_capacity'] *
        (parameters['model_parameters']['temperature_limits'][1] -
         parameters['model_parameters']['temperature_limits'][0]))
    if heat_capacity != 0:
        soc = (
            parameters['model_parameters']['heat_capacity'] *
            (parameters['model_inputs']['fRoomTemperature'] -
             parameters['model_parameters']['temperature_limits'][0])/heat_capacity)
    else:
        soc = 0

    result = {
        'thermal_energy': np.array(thermal_energy, dtype=np.float64),
        'thermal_energy_min': np.array([max(0, thermal_energy_min)], dtype=np.float64),
        'soc': soc,
        }

    return result


def demand_building_one_product(parameters):
    """
    building demand which is calculated by ambient temperature

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal energies and current soc for bidding process
    """
    actual_temperature = parameters['model_inputs']['fRoomTemperature']
    target_temperature = parameters['model_parameters']['target_temperature']
    mean_temperature = np.mean([target_temperature, actual_temperature])

    # demand means ambient temperature
    demand = (parameters['model_parameters']['heat_capacity']*(target_temperature-actual_temperature))/parameters[
        'model_inputs']['product_type']+(mean_temperature-parameters['model_inputs']['ambient_temperature'])/parameters[
            'model_parameters']['thermal_resistance_to_ambient']
    cleared_power = max(np.array(parameters['model_inputs']['cleared_energy_pos'])-np.array(parameters[
        'model_inputs']['cleared_energy_neg']))/(parameters['model_inputs']['trading_time']/3600)

    # residual power - possible [load reduction, load raise]
    residual_power = [
        (parameters['model_parameters']['heat_capacity'] *
            (actual_temperature-parameters['model_parameters']['temperature_limits'][0])) /
        parameters['model_inputs']['product_type'],
        (parameters['model_parameters']['heat_capacity'] *
            (parameters['model_parameters']['temperature_limits'][1]-actual_temperature)) /
        parameters['model_inputs']['product_type']
    ]

    # determine if system is in heating mode
    if parameters['model_inputs']['bHeatingMode'] == 1:
        power_max = max(demand + residual_power[1], 0) + cleared_power
        power_min = max(demand - residual_power[0], 0) + cleared_power
    # cool demand
    else:
        power_max = min(demand - residual_power[0], 0) - cleared_power
        power_min = min(demand + residual_power[1], 0) - cleared_power
    thermal_powers = np.array([power_min, power_max])
    thermal_energy_min = thermal_powers[0]*parameters['model_inputs']['product_type']/3600
    thermal_energy = thermal_powers*parameters['model_inputs']['product_type']*parameters[
        'model_inputs']['product_allocation']/3600

    heat_capacity = (
        parameters['model_parameters']['heat_capacity'] *
        (parameters['model_parameters']['temperature_limits'][1] -
         parameters['model_parameters']['temperature_limits'][0]))
    if heat_capacity != 0:
        soc = (
            parameters['model_parameters']['heat_capacity'] *
            (parameters['model_inputs']['fRoomTemperature'] -
             parameters['model_parameters']['temperature_limits'][0])/heat_capacity)
    else:
        soc = 0

    result = {
        'thermal_energy': np.array(thermal_energy, dtype=np.float64),
        'thermal_energy_min': np.array([max(0, thermal_energy_min)], dtype=np.float64),
        'soc': soc,
        }

    return result


def thermal_network(parameters):
    """
    thermal_network acting as system operator

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal energy and soc for bidding process
    """
    actual_temperature = (
        parameters['model_inputs']['fUpperTemperature']+parameters['model_inputs']['fLowerTemperature'])/2
    target_temperature = parameters['model_parameters']['target_temperature']

    demand_power = (
        parameters['model_parameters']['heat_capacity'] *
        (target_temperature-actual_temperature))/parameters['model_inputs']['product_type']

    heat_capacity = (
        parameters['model_parameters']['heat_capacity'] *
        (parameters['model_parameters']['temperature_limits'][1] -
         parameters['model_parameters']['temperature_limits'][0]))
    soc = (
        parameters['model_parameters']['heat_capacity'] *
        (actual_temperature-parameters['model_parameters']['temperature_limits'][0])/heat_capacity)

    if demand_power > 0:
        thermal_energy = np.array(demand_power)*parameters['model_inputs']['product_type']*parameters[
            'model_inputs']['buy_product_allocation']/3600
    else:
        thermal_energy = np.array(demand_power)*parameters['model_inputs']['product_type']*parameters[
            'model_inputs']['sell_product_allocation']/3600

    result = {
        'thermal_energy': np.array(thermal_energy, dtype=np.float64),
        'soc': soc
        }

    return result


def storage(parameters):
    """
    active heat or cold storage

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding traded thermal energy, stored energy and soc for bidding process
    """

    actual_temperature = (
        parameters['model_inputs']['fUpperTemperature']+parameters['model_inputs']['fLowerTemperature'])/2
    heat_capacity = (
        parameters['model_parameters']['heat_capacity'] *
        (parameters['model_parameters']['temperature_limits'][1] -
         parameters['model_parameters']['temperature_limits'][0]))
    stored_energy = max(0, parameters['model_parameters']['heat_capacity'] *
                        (actual_temperature-parameters['model_parameters']['temperature_limits'][0]))
    soc = stored_energy/heat_capacity

    stored_energy = stored_energy/3600
    cleared_energy = abs(
        np.array(parameters['model_inputs']['cleared_energy_pos']) -
        np.array(parameters['model_inputs']['cleared_energy_neg']))

    # buy fixed ratio of heat capacity - kJ in kWh
    if not parameters['model_inputs']['shorttime_product']:
        if parameters['model_parameters']['is_heat_storage']:
            thermal_energy = (
                heat_capacity/3600/len(cleared_energy) -
                cleared_energy)*parameters['model_inputs']['buy_product_allocation']
        else:
            thermal_energy = (
                heat_capacity/3600/len(cleared_energy) -
                cleared_energy)*parameters['model_inputs']['sell_product_allocation']
    # sell traded energy on shortterm market
    else:
        if parameters['model_parameters']['is_heat_storage']:
            thermal_energy = cleared_energy*parameters['model_inputs']['sell_product_allocation']
        else:
            thermal_energy = cleared_energy*parameters['model_inputs']['buy_product_allocation']

    result = {
        'thermal_energy': np.array(thermal_energy, dtype=np.float64),
        'stored_energy': np.array([stored_energy], dtype=np.float64),
        'soc': soc
        }

    return result


def storage_one_product(parameters):
    """
    active heat or cold storage

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding traded thermal energy, stored energy and soc for bidding process
    """

    actual_temperature = (
        parameters['model_inputs']['fUpperTemperature']+parameters['model_inputs']['fLowerTemperature'])/2
    heat_capacity = (
        parameters['model_parameters']['heat_capacity'] *
        (parameters['model_parameters']['temperature_limits'][1] -
         parameters['model_parameters']['temperature_limits'][0]))
    stored_energy = max(0, parameters['model_parameters']['heat_capacity'] *
                        (actual_temperature-parameters['model_parameters']['temperature_limits'][0]))
    soc = stored_energy/heat_capacity
    cleared_energy = abs(
        np.array(parameters['model_inputs']['cleared_energy_pos']) -
        np.array(parameters['model_inputs']['cleared_energy_neg']))

    # buy fixed ratio of heat capacity - kJ in kWh and sell only stored energy
    if parameters['model_parameters']['is_heat_storage']:
        thermal_energy = (
            heat_capacity/3600/len(cleared_energy) -
            cleared_energy)*parameters['model_inputs']['buy_product_allocation']
    else:
        thermal_energy = (
            heat_capacity/3600/len(cleared_energy) -
            cleared_energy)*parameters['model_inputs']['sell_product_allocation']

    result = {
        'thermal_energy': np.array(thermal_energy, dtype=np.float64),
        'stored_energy': np.array([stored_energy], dtype=np.float64),
        'soc': soc
        }

    return result


def heat_exchanger(parameters):
    """
    heat exchanger

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal energies for bidding process
    """
    c_min_flow = min(
        parameters['model_parameters']['max_capacity_flow_hot'],
        parameters['model_parameters']['max_capacity_flow_cold'])
    c_max_flow = max(
        parameters['model_parameters']['max_capacity_flow_hot'],
        parameters['model_parameters']['max_capacity_flow_cold'])

    # limit power to zero because bids are always greater than zero
    thermal_power_max = c_min_flow*(
        parameters['model_inputs']['fFeedTemperature_hot'] -
        parameters['model_inputs']['fReturnTemperature_cold'])

    ntu = (
        parameters['model_parameters']['heat_transfer_coefficient'] *
        parameters['model_parameters']['heat_exchanger_area']/c_min_flow)
    if c_min_flow == 0:
        eps = 0
    elif c_min_flow/c_max_flow == 1:
        eps = ntu/(1+ntu)
    else:
        eps = (1-np.exp(-(1-c_min_flow/c_max_flow)*ntu))/(1-c_min_flow/c_max_flow*np.exp(
            -(1-c_min_flow/c_max_flow)*ntu))

    # use case heating
    if parameters['model_parameters']['heating_use_case']:
        cleared_power = max(
            np.array(parameters['model_inputs']['cleared_energy_pos']))/(parameters[
                'model_inputs']['trading_time']/3600)
    else:
        cleared_power = max(
            np.array(parameters['model_inputs']['cleared_energy_neg']))/(parameters[
                'model_inputs']['trading_time']/3600)
    thermal_power = max(thermal_power_max*eps - cleared_power, 0)

    thermal_energy = thermal_power*parameters['model_inputs']['product_type']*parameters[
        'model_inputs']['product_allocation']/3600

    result = {
        'thermal_energy': np.array([thermal_energy], dtype=np.float64),
        }
    return result


def heat_pump(parameters):
    """
    heat pump

    Args:
        parameters (dict): dictionary holding model parameters and model inputs

    Returns:
        result (dict): dictionary holding thermal and electric energies for bidding process
    """
    # thermal efficiency
    thermal_efficiencies = np.array(parameters['model_parameters']['thermal_efficiencies'])
    return_temperatures_hot_thermal_grid = thermal_efficiencies[0, 1:]
    feed_temperature_cold_thermal_grid = thermal_efficiencies[1:, 0]
    data_thermal_grid = thermal_efficiencies[1:, 1:]
    return_temperature_hot_thermal_mesh, feed_temperature_cold_thermal_mesh = np.meshgrid(
        return_temperatures_hot_thermal_grid, feed_temperature_cold_thermal_grid)
    points_thermal = np.column_stack([return_temperature_hot_thermal_mesh.ravel(),
                                      feed_temperature_cold_thermal_mesh.ravel()])
    values_thermal = data_thermal_grid.ravel()
    thermal_efficiency = griddata(
        points_thermal, values_thermal,
        (
            parameters['model_inputs']['fReturnTemperature_hot'],
            parameters['model_inputs']['fReturnTemperature_cold']
        ),
        method='linear')
    if np.isnan(thermal_efficiency):
        thermal_efficiency = griddata(
            points_thermal, values_thermal,
            (
                parameters['model_inputs']['fReturnTemperature_hot'],
                parameters['model_inputs']['fReturnTemperature_cold']
            ),
            method='nearest')

    # electric efficiency
    electric_efficiencies = np.array(parameters['model_parameters']['electric_efficiencies'])
    return_temperatures_hot_electric_grid = electric_efficiencies[0, 1:]
    feed_temperature_cold_electric_grid = electric_efficiencies[1:, 0]
    data_electric_grid = electric_efficiencies[1:, 1:]
    return_temperature_hot_electric_mesh, feed_temperature_cold_electric_mesh = np.meshgrid(
        return_temperatures_hot_electric_grid,
        feed_temperature_cold_electric_grid)
    points_electric = np.column_stack([return_temperature_hot_electric_mesh.ravel(),
                                       feed_temperature_cold_electric_mesh.ravel()])
    values_electric = data_electric_grid.ravel()
    electric_efficiency = griddata(
        points_electric, values_electric,
        (
            parameters['model_inputs']['fReturnTemperature_hot'],
            parameters['model_inputs']['fReturnTemperature_cold']
        ),
        method='linear')
    if np.isnan(electric_efficiency):
        electric_efficiency = griddata(
            points_electric, values_electric,
            (
                parameters['model_inputs']['fReturnTemperature_hot'],
                parameters['model_inputs']['fReturnTemperature_cold']
            ),
            method='nearest')

    nominal_cooling_power = thermal_efficiency*parameters['model_parameters']['nominal_cooling_power']
    nominal_electric_power = electric_efficiency*parameters['model_parameters']['nominal_electric_power']
    nominal_heating_power = nominal_cooling_power+nominal_electric_power

    # use case heating
    if parameters['model_parameters']['heating_use_case']:
        cleared_power = max(np.array(parameters['model_inputs']['cleared_energy_pos']))/(parameters[
            'model_inputs']['trading_time']/3600)

        power_max_heat = np.clip(nominal_heating_power-cleared_power, 0, nominal_heating_power)
        power_max_electric = power_max_heat/nominal_heating_power*nominal_electric_power
        power_max_cool = power_max_heat-power_max_electric

        # minimum residual power
        if parameters['model_parameters']['minimal_load']*power_max_heat > cleared_power:
            min_operating_point = parameters['model_parameters']['minimal_load']
        else:
            min_operating_point = 0
    else:
        cleared_power = max(np.array(parameters['model_inputs']['cleared_energy_neg']))/(parameters[
            'model_inputs']['trading_time']/3600)

        power_max_cool = np.clip(nominal_cooling_power-cleared_power, 0, nominal_cooling_power)
        power_max_electric = power_max_cool/nominal_cooling_power*nominal_electric_power
        power_max_heat = power_max_cool+power_max_electric

        # minimum residual power
        if parameters['model_parameters']['minimal_load'] > np.divide(cleared_power,
                                                                      power_max_cool, out=np.zeros_like(power_max_cool),
                                                                      where=power_max_cool != 0):
            min_operating_point = parameters['model_parameters']['minimal_load']
        else:
            min_operating_point = 0

    if parameters['model_parameters']['bid_discretization'] != 1:
        thermal_energy_heat = np.linspace(
            min_operating_point*power_max_heat,
            power_max_heat,
            num=parameters['model_parameters']['bid_discretization']
            )*parameters['model_inputs']['product_type']*parameters['model_inputs']['product_allocation']/3600
        thermal_energy_cool = np.linspace(
            min_operating_point*power_max_cool,
            power_max_cool,
            num=parameters['model_parameters']['bid_discretization']
            )*parameters['model_inputs']['product_type']*parameters['model_inputs']['product_allocation']/3600
        electric_energy = np.linspace(
            min_operating_point*power_max_electric,
            power_max_electric,
            num=parameters['model_parameters']['bid_discretization']
            )*parameters['model_inputs']['product_type']*parameters['model_inputs']['product_allocation']/3600
    else:
        thermal_energy_heat = np.array([power_max_heat])*parameters['model_inputs']['product_type']*parameters[
            'model_inputs']['product_allocation']/3600
        thermal_energy_cool = np.array([power_max_cool])*parameters['model_inputs']['product_type']*parameters[
            'model_inputs']['product_allocation']/3600
        electric_energy = np.array([power_max_electric])*parameters['model_inputs']['product_type']*parameters[
            'model_inputs']['product_allocation']/3600

    result = {
        'thermal_energy_heat': np.array(thermal_energy_heat, dtype=np.float64),
        'thermal_energy_cool': np.array(thermal_energy_cool, dtype=np.float64),
        'electric_energy': np.array(electric_energy, dtype=np.float64)
        }
    return result
