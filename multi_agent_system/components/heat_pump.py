"""
heat pump agent
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "heat pump agent"

import numpy as np
from multi_agent_system.components.trader import Trader
from multi_agent_system.base.util import DynamicObject
from multi_agent_system.models import pricing_models, quantity_assessment_models  # noqa: F401


class HeatPump(Trader):
    """
    heat pump between to supply systems and markets

    Args:
        Trader (object): extends trader class
    """

    def setup_agent(self):
        """
        initally executed setup method
        """
        super().setup_agent()

        # initialize cop and electricity price object
        self.cop = 1
        self.electricity_prices = DynamicObject(
            filename=self.experiment_config['cost_electricity'],
            sheet_name='electricity_price')

    def process_msg(self, msg):
        """
        implements message processing

        Args:
            msg (dict): message object
        """
        super().process_msg(msg)

        # overwrite results from parent method if consumer is connected to more than one network
        if msg['type'] == 'balancing_energy_msg':
            log = self.trading_table_longtime[-1]
            is_hot_network = msg['system_id'] == self.agent_config['base_config']['connections_markets'][0]
            # heat counter is implemented on producer network and
            if is_hot_network:
                energy_difference = (log['cleared_energy_pos'] - log['real_energy_pos'])
            else:
                energy_difference = -((log['cleared_energy_pos'] - log['real_energy_pos']) * (1 - 1 / self.cop))

            if energy_difference >= 0:
                # produced less/consumed more energy than cleared
                cost_balancing_energy = energy_difference * msg['price_pos']
            else:
                # produced more/consumed less energy than cleared
                cost_balancing_energy = energy_difference * msg['price_neg']

            # update longtime trading log with balancing energy costs
            if energy_difference != 0:
                self.trading_table_longtime[-1]['cost_balancing_energy' + '_' +
                                                msg['system_id']] = cost_balancing_energy / energy_difference
            else:
                self.trading_table_longtime[-1]['cost_balancing_energy' + '_' + msg['system_id']] = 0
            self.trading_table_longtime[-1]['balancing_energy' + '_' + msg['system_id']] = energy_difference

    def get_state(self, observation, control_step):
        """
        get state from environemt

        Args:
            observation (dict): observed values from environment
            control_step (int): number of control within trading trading time
        """
        super().get_state(observation, control_step)

        # get observation for heat energy after last trading period
        last_observation = self.observations[-1 - control_step]
        current_observation = self.observations[-1]
        energy_difference = current_observation['fHeatEnergy'] - \
            last_observation['fHeatEnergy']  # calculate supplied/consumed energy

        if current_observation['fElectricPower'] > 0.01 and current_observation['fHeatFlowRate'] > 0.01:
            self.cop = current_observation['fHeatFlowRate'] / current_observation['fElectricPower']
        else:
            self.cop = 1

        # energy on production side is calculated considering forecasted cop
        self.trading_table[0]['real_energy_pos'] = abs(energy_difference)
        self.trading_table[0]['real_energy_neg'] = abs(energy_difference) * (1 - 1 / self.cop)

    def set_actions(self):
        """
        set actions from agent to environment to supervisory controller (environment)

        Returns:
            action (dict): set controlled variables as dictionary to supervisory controller
        """

        # check for use cases
        heating_use_case = self.agent_config['model_config']['model_parameters']['heating_use_case']
        additional_producer = self.agent_config['model_config']['model_parameters']['additional_producer']

        # use case 1a - single cold producer
        if not heating_use_case and not additional_producer:
            ideal_cop = self.observations[-1]['fReturnTemperature_hot'] / \
                (self.observations[-1]['fReturnTemperature_hot'] - self.observations[-1]['fReturnTemperature_cold'])
            # cleared energy is always at producer side -> heat counter
            cleared_energy = self.trading_table[0]['cleared_energy_neg'] * (ideal_cop / (ideal_cop - 1))
        # use case 1b - single cold producer
        elif heating_use_case and not additional_producer:
            cleared_energy = self.trading_table[0]['cleared_energy_pos']
        # use case 2 - additional producer
        else:
            # execute smaller amount
            if self.trading_table[0]['cleared_energy_pos'] <= self.trading_table[0]['cleared_energy_neg']:
                cleared_energy = self.trading_table[0]['cleared_energy_pos']
            else:
                ideal_cop = self.observations[-1]['fReturnTemperature_hot'] / (
                    self.observations[-1]['fReturnTemperature_hot'] - self.observations[-1]['fReturnTemperature_cold'])
                # cleared energy is always at producer side -> heat counter
                cleared_energy = self.trading_table[0]['cleared_energy_neg'] * (ideal_cop / (ideal_cop - 1))

        # enable power controlled option
        if self.experiment_config['is_power_controlled']:
            bSetStatusOn = 1.0 if cleared_energy != 0 else 0.0
            fSetPoint = abs(cleared_energy) * 3600 / self.trading_time
        else:
            real_energy = self.trading_table[0]['real_energy_pos']
            # calculate cleared energy and actual energy in current timestep
            if abs(real_energy) > abs(cleared_energy):
                energy_difference = 0.01  # minimum value
            else:
                energy_difference = cleared_energy - real_energy
            time_left = (
                (self.trading_time / self.experiment_config['sampling_time']) - self.control_step
                ) * self.experiment_config['sampling_time']

            bSetStatusOn = 1.0 if energy_difference != 0 and cleared_energy != 0 else 0.0
            # calculate mean power for left control time in trading period
            fSetPoint = (abs(energy_difference) * 3600) / (time_left)

        output_vars = self.agent_config['base_config']['env_outputs']
        action = {output_vars['bSetStatusOn']: bSetStatusOn, output_vars['fSetPoint']: fSetPoint}
        return action

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

        # pass quantities to pricing assessment
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

        # append product allocation and use cases to model inputs
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
        self.pricing_parameters['model_inputs']['is_running'] = (
            last_log['cleared_energy_pos'] or last_log['cleared_energy_neg']) != 0

        # append information about electricity prices to model inputs
        start_time = self.experiment_time + product['lead_time']
        end_time = start_time + product['product_type']
        df_electricity_price = self.electricity_prices.get_values(start_time=start_time, end_time=end_time)
        mean_electricity_price = np.mean(df_electricity_price['value'].to_numpy())
        self.pricing_parameters['model_inputs']['electricity_price'] = mean_electricity_price

        model_type = "pricing_models." + self.agent_config['model_config']['pricing_model']
        order_msgs = eval(model_type + "(self.pricing_parameters)")
        return order_msgs


if __name__ == '__main__':
    pass
