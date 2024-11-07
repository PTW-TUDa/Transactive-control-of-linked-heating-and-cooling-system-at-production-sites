"""
consumer agent
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "consumer agent"

import numpy as np
from multi_agent_system.components.trader import Trader
from multi_agent_system.base.util import DynamicObject
from multi_agent_system.models import pricing_models, quantity_assessment_models  # noqa: F401


class Consumer(Trader):
    """
    active heat or cold storage

    Args:
        Trader (object): extends trader class
    """

    def process_msg(self, msg):
        """
        extend method for message processing

        Args:
            msg (dict): message from markets or system operators
        """
        super().process_msg(msg)

        # overwrite results from parent method if consumer is connected to more than one network
        if len(self.agent_config['base_config']['connections_markets']) > 1 and msg['type'] == 'balancing_energy_msg':
            log = self.trading_table_longtime[-1]
            is_hot_network = msg['sender_id'] == self.agent_config['base_config']['connections_markets'][0]

            # difference of cleared to real energy as producer
            pos_energy_diff = log['cleared_energy_pos'] - log['real_energy_pos']
            # difference of cleared to real energy as producer
            neg_energy_diff = log['cleared_energy_neg'] - log['real_energy_neg']

            # considered energy difference depending on heating / cooling use case
            energy_difference = neg_energy_diff if is_hot_network else pos_energy_diff

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

    def setup_agent(self):
        """
        initally executed setup method
        """
        super().setup_agent()

        self.min_demand = 0  # initialize minimum demand

        # additional demand information if demand cannot be forecasted by global parameters, e.g. ambient temperature
        try:
            self.demands = DynamicObject(
                filename=self.agent_config['model_config']['model_parameters']['demand'],
                sheet_name=self.name)
            self.has_demand = True
        except BaseException:
            self.has_demand = False

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

        # update trading table
        if energy_difference > 0:
            self.trading_table[0]['real_energy_neg'] = abs(energy_difference)
        else:
            self.trading_table[0]['real_energy_pos'] = abs(energy_difference)

    def set_actions(self):
        """
        set actions from agent to environment to supervisory controller (environment)

        Returns:
            action (dict): set controlled variables as dictionary to supervisory controller
        """
        action = super().set_actions()

        # calculate cleared energy as producer
        cleared_energy = self.trading_table[0]['cleared_energy_pos'] - self.trading_table[0]['cleared_energy_neg']

        # overwrite cleared energy with minimum demenad if clearing was not successful
        if abs(self.min_demand) > abs(cleared_energy):
            # cleared and real energy as producer
            cleared_energy = -self.min_demand

            # enable power and energy controlled consumer
            if self.experiment_config["is_power_controlled"]:
                action[self.agent_config['base_config']['env_outputs'][
                    'bSetStatusOn']] = 1.0 if cleared_energy != 0 else 0.0
                action[self.agent_config['base_config']['env_outputs'][
                    'fSetPoint']] = abs(cleared_energy) * 3600 / self.trading_time
            else:
                real_energy = self.trading_table[0]['real_energy_pos'] - self.trading_table[0]['real_energy_neg']
                energy_difference = cleared_energy - real_energy

                # time left within this trading period
                time_left = (
                    (self.trading_time / self.experiment_config['sampling_time']) - self.control_step
                    ) * self.experiment_config['sampling_time']

                # set bSetStatusOn - false if energy_difference is 0
                action[self.agent_config['base_config']['env_outputs'][
                    'bSetStatusOn']] = 1.0 if energy_difference != 0 and cleared_energy != 0 else 0.0

                # calculate mean power for left control time in trading period
                action[self.agent_config['base_config']['env_outputs'][
                    'fSetPoint']] = (abs(energy_difference) * 3600) / (time_left)

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

        self.physical_parameters['model_inputs']['product_allocation'] = self.product_allocation[
            product['product_type']][product['lead_time']]

        # pass demand to model if demand exists
        if self.has_demand:
            start_time = self.experiment_time + product['lead_time']
            end_time = start_time + product['product_type']
            df_demand = self.demands.get_values(start_time=start_time, end_time=end_time)
            mean_demand = np.mean(df_demand['value'].to_numpy())
            self.physical_parameters['model_inputs']['demand'] = mean_demand

        # execute model and return quantities
        model_type = "quantity_assessment_models." + self.agent_config['model_config']['capacity_model']
        quantities = eval(model_type + "(self.physical_parameters)")

        # save forecast before execution - consumers always execute forecast for minimum energy
        if self.physical_parameters['model_inputs']['shorttime_product']:
            self.min_demand = quantities['thermal_energy_min'][0]

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

        # execute pricing model and return order messages
        model_type = "pricing_models." + self.agent_config['model_config']['pricing_model']
        order_msgs = eval(model_type + "(self.pricing_parameters)")
        return order_msgs


if __name__ == '__main__':
    pass
