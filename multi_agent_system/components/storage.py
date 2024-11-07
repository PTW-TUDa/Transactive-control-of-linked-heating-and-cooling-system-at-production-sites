"""
active storage agent
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "active storage agent"

from multi_agent_system.components.trader import Trader
from multi_agent_system.models import pricing_models, quantity_assessment_models  # noqa: F401


class Storage(Trader):
    """
    active heat or cold storage

    Args:
        Trader (object): extends trader class
    """

    def setup_agent(self):
        """
        initally executed setup method
        """
        super().setup_agent()

        # define intial costs for stored energy, typically zero
        self.energy_costs = 0

    def process_msg(self, msg):
        """
        extend method for message processing

        Args:
            msg (dict): message from markets or system operators
        """
        super().process_msg(msg)

        # calculate costs for stored energy after billing balancing energy
        if msg['type'] == 'balancing_energy_msg':
            log = self.trading_table_longtime[-1]
            cost = log['cleared_energy_neg'] * log['price_neg'] - log['cleared_energy_pos'] * log['price_pos']
            self.energy_costs += cost

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

        # get last and current temperature and calculate energy difference between observations
        last_temp = (last_observation['fUpperTemperature'] + last_observation['fLowerTemperature']) / 2
        current_temp = (observation['fUpperTemperature'] + observation['fLowerTemperature']) / 2
        energy_difference = self.agent_config['model_config']['model_parameters']['heat_capacity'] / \
            3600 * (current_temp - last_temp)

        if energy_difference > 0:
            # storage has been loading and consumed heat energy
            self.trading_table[0]['real_energy_neg'] = abs(energy_difference)
        else:
            # storage has been unloaded and produced energy
            self.trading_table[0]['real_energy_pos'] = abs(energy_difference)

    def set_actions(self):
        """
        set actions from agent to environment to supervisory controller (environment)

        Returns:
            action (dict): set controlled variables as dictionary to supervisory controller
        """
        action = super().set_actions()

        # query cleared energy as producers
        cleared_energy = self.trading_table[0]['cleared_energy_pos'] - self.trading_table[0]['cleared_energy_neg']

        # cleared as producer for heat storage means unloading and cleared as producer for cold storage means loading
        if cleared_energy >= 0:
            if self.agent_config['model_config']['model_parameters']['is_heat_storage']:
                action[self.agent_config['base_config']['env_outputs']['bLoading']] = 0
            else:
                action[self.agent_config['base_config']['env_outputs']['bLoading']] = 1
        else:
            if self.agent_config['model_config']['model_parameters']['is_heat_storage']:
                action[self.agent_config['base_config']['env_outputs']['bLoading']] = 1
            else:
                action[self.agent_config['base_config']['env_outputs']['bLoading']] = 0

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

        # extend model inputs for active storage
        self.physical_parameters['model_inputs']['buy_product_allocation'] = self.buy_product_allocation[
            product['product_type']][product['lead_time']]
        self.physical_parameters['model_inputs']['sell_product_allocation'] = self.sell_product_allocation[
            product['product_type']][product['lead_time']]

        # execute model and return quantities
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

        # extends model inputs with active storage specific informations, e.g. costs for stored energy
        self.pricing_parameters['model_inputs']['energy_costs'] = self.energy_costs

        # execute pricing model and return order messages
        model_type = "pricing_models." + self.agent_config['model_config']['pricing_model']
        order_msgs = eval(model_type + "(self.pricing_parameters)")
        return order_msgs


if __name__ == '__main__':
    pass
