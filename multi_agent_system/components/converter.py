"""
converter agent
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "converter agent"

import numpy as np
from multi_agent_system.components.trader import Trader
from multi_agent_system.base.util import DynamicObject
from multi_agent_system.models import pricing_models, quantity_assessment_models  # noqa: F401


class Converter(Trader):
    """
    converter generating heating or cooling energy

    Args:
        Trader (object): extends trader class
    """

    def setup_agent(self):
        """
        initally executed setup method
        """
        super().setup_agent()

        # initalize dynamic price and demand objects
        self.electricity_prices = DynamicObject(
            filename=self.experiment_config['cost_electricity'],
            sheet_name='electricity_price')
        self.fuel_prices = DynamicObject(filename=self.experiment_config['cost_fuel'], sheet_name='fuel_price')
        self.electricity_demand = DynamicObject(
            filename=self.experiment_config['electricity_demand'],
            sheet_name='electricity_demand')

    def get_state(self, observation, control_step):
        """
        get state from environemt

        Args:
            observation (dict): observed values from environment
            control_step (int): number of control within trading trading time
        """
        super().get_state(observation, control_step)

        # append generated energy to trading table
        last_observation = self.observations[-1 - control_step]
        current_observation = self.observations[-1]
        energy_difference = current_observation['fHeatEnergy'] - \
            last_observation['fHeatEnergy']  # calculate supplied/consumed energy

        if energy_difference < 0:
            # consumed heat, produced cold
            self.trading_table[0]['real_energy_neg'] = abs(energy_difference)
        else:
            # consumed cold, produced heat
            self.trading_table[0]['real_energy_pos'] = abs(energy_difference)

    def trade(self, product):
        """
        method to call trading process

        Args:
            product (int): traded product defined by product duration in seconds

        Returns:
            msgs (list): list of order messages which are processed to market
        """

        # perform capacity assessment
        quantities = self.__quantity_assessment(product=product)

        # pass quantities ot pricing assessment
        msgs = self.__pricing(product=product, quantities=quantities)

        return msgs

    def __quantity_assessment(self, product):
        """
        quantity_assessment

        Args:
            product (int): traded product defined by product duration in seconds

        Returns:
            quantities (dict): dictionary holding tradable quantities and further information for pricing (e.g. soc)
        """
        super()._quantity_assessment(product=product)

        self.physical_parameters['model_inputs']['product_allocation'] = self.product_allocation[
            product['product_type']][product['lead_time']]

        model_type = "quantity_assessment_models." + self.agent_config['model_config']['capacity_model']
        quantities = eval(model_type + "(self.physical_parameters)")
        return quantities

    def __pricing(self, product, quantities):
        """
        pricing

        Args:
            product (int): traded product defined by product duration in seconds
            quantities (dict): dictionary holding tradable quantities and further information for pricing (e.g. soc)

        Returns:
            msgs (list): list of order messages which are processed to market
        """
        super()._pricing(product=product, quantities=quantities)
        last_log = self.trading_table_longtime[-1]

        # append information about minimal load and running system to model inputs
        self.pricing_parameters['model_inputs']['is_running'] = last_log['cleared_energy_pos'] != 0

        # append information about electricity, fuel prices and electricity demand to model inputs
        start_time = self.experiment_time + product['lead_time']
        end_time = start_time + product['product_type']
        df_electricity_price = self.electricity_prices.get_values(start_time=start_time, end_time=end_time)
        df_fuel_price = self.fuel_prices.get_values(start_time=start_time, end_time=end_time)
        df_electricity_demand = self.electricity_demand.get_values(start_time=start_time, end_time=end_time)
        mean_electricity_price = np.mean(df_electricity_price['value'].to_numpy())
        mean_fuel_price = np.mean(df_fuel_price['value'].to_numpy())
        mean_electricity_demand = np.mean(df_electricity_demand['value'].to_numpy())
        self.pricing_parameters['model_inputs']['electricity_price'] = mean_electricity_price
        self.pricing_parameters['model_inputs']['fuel_price'] = mean_fuel_price
        self.pricing_parameters['model_inputs']['electricity_demand'] = mean_electricity_demand
        self.pricing_parameters['model_inputs']['chp_renumeration'] = self.experiment_config['chp_renumeration']

        model_type = "pricing_models." + self.agent_config['model_config']['pricing_model']
        order_msgs = eval(model_type + "(self.pricing_parameters)")
        return order_msgs


if __name__ == '__main__':
    pass
