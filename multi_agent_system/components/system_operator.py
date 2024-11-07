"""
system operator agent
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "buffer storage agent"

from multi_agent_system.components.trader import Trader
from multi_agent_system.base.messages import balancing_energy_msg
from multi_agent_system.models import pricing_models, quantity_assessment_models  # noqa: F401


class SystemOperator(Trader):
    """
    system operator

    Args:
        Trader (object): extends trader class
    """

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

        last_temp = (last_observation['fUpperTemperature'] + last_observation['fLowerTemperature']) / 2
        current_temp = (observation['fUpperTemperature'] + observation['fLowerTemperature']) / 2
        energy_difference = self.agent_config['model_config']['model_parameters']['heat_capacity'] / \
            3600 * (current_temp - last_temp)

        if energy_difference > 0:
            self.trading_table[0]['real_energy_neg'] = abs(energy_difference)  # heat counter implemented as consumer
        else:
            self.trading_table[0]['real_energy_pos'] = abs(energy_difference)

        # add balancing energy demand with 0 to log for postprocessing
        self.trading_table_longtime[-1]['balancing_energy' + '_' +
                                        self.agent_config['base_config']['connections_markets'][0]] = 0
        self.trading_table_longtime[-1]['cost_balancing_energy' + '_' +
                                        self.agent_config['base_config']['connections_markets'][0]] = 0

    def set_actions(self):
        """
        set actions from agent to environment to supervisory controller (environment)

        Returns:
            action (dict): empty dictionary as system_operator sets no actions
        """
        return {}

    def return_balancing_energy_price(self):
        """
        return balancing energy prices for last trading period

        Returns:
            msg (dict): balancing energy message
        """

        '''
        positive balancing energy (supplied by buffer storage) must be buyed by the buffer storage as consumer
        negative balancing energy (supplied by buffer storage) must be sold by the buffer storage as producer
        the buffer storage as consumer -> price_pos
        '''

        # get trading log
        log = self.trading_table[0]

        # generate msgs for connected trades, 0 ist always connected market
        msgs = []
        for trader in self.agent_config['base_config']['connections_traders']:
            msgs.append(balancing_energy_msg(
                sender_id=self.name,
                reciever_id=trader,
                system_id=self.agent_config['base_config']['connections_markets'][0],
                price_pos=max(0, log['price_neg'])+self.experiment_config['positive_market_limit'],
                price_neg=max(0, log['price_pos'])+self.experiment_config['positive_market_limit']
            ))
        return msgs

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

        self.physical_parameters['model_inputs']['buy_product_allocation'] = self.buy_product_allocation[
            product['product_type']][product['lead_time']]
        self.physical_parameters['model_inputs']['sell_product_allocation'] = self.sell_product_allocation[
            product['product_type']][product['lead_time']]

        # # execute pmodel and return quantities
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

        # execute pricing model and return order messages
        model_type = "pricing_models." + self.agent_config['model_config']['pricing_model']
        order_msgs = eval(model_type + "(self.pricing_parameters)")
        return order_msgs


if __name__ == '__main__':
    pass
