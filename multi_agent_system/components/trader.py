"""
trader agent (base)
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "trader agent (base)"

from abc import abstractmethod
import numpy as np
from multi_agent_system.base.base_agent import BaseAgent
from multi_agent_system.base.util import DynamicObject


class Trader(BaseAgent):
    """
    general class for traders

    Args:
        BaseAgent (class): extends BaseAgent
    """

    def setup_agent(self):
        """
        initally executed setup method
        """

        # initialize price objects
        self.ambient_temperature = DynamicObject(
            filename=self.experiment_config['ambient_temperature'],
            sheet_name='ambient_temperature')

        var_name_list = [
            'cleared_energy_pos',  # cleared energy as producer
            'cleared_energy_neg',  # cleared energy as consumer
            'real_energy_pos',  # real energy as producer
            'real_energy_neg',  # real energy as consumer
            'price_pos',  # price for cleared energy as producer
            'price_neg',  # price for cleared energy as consumer
        ]

        # create variable name list for all connected markets and products
        for market in self.agent_config['base_config']['connections_markets']:
            for product in self.products:
                var_name_list.append(str(product) + '_' + market)
                var_name_list.append("price_" + str(product) + '_' + market)

        # create trading table as data frame
        longest_product = max(list(self.products.keys()))
        shortest_product = min(list(self.products.keys()))
        longest_lead_time = max(self.products[longest_product])
        horizon = int((longest_product + longest_lead_time) / shortest_product)
        self.trading_table = [{var_name: 0 for var_name in var_name_list} for _ in range(horizon)]

        # create nested dictionary for product allocation
        if self.type in ['system_operator', 'storage']:
            self.buy_product_allocation = {}
            self.sell_product_allocation = {}
            for product, lead_time, allocation in zip(list(self.products.keys()),
                                                      list(self.products.values()),
                                                      self.agent_config['model_config'][
                                                          'model_parameters']['buy_product_allocation']):
                self.buy_product_allocation[product] = dict(zip(lead_time, allocation))
            for product, lead_time, allocation in zip(list(self.products.keys()),
                                                      list(self.products.values()),
                                                      self.agent_config['model_config'][
                                                          'model_parameters']['sell_product_allocation']):
                self.sell_product_allocation[product] = dict(zip(lead_time, allocation))
        else:
            self.product_allocation = {}
            for product, lead_time, allocation in zip(list(self.products.keys()), list(
                    self.products.values()), self.agent_config['model_config'][
                        'model_parameters']['product_allocation']):
                self.product_allocation[product] = dict(zip(lead_time, allocation))
        self.shorttime_product = list(self.products.keys())[0]

    def process_msg(self, msg):
        """
        implements message processing

        Args:
            msg (dict): message object
        """

        if msg['type'] == 'trade_msg':
            self.__process_clearing(msg)
        elif msg['type'] == 'balancing_energy_msg':
            self.__billing(msg)

    def __billing(self, msg):
        """
        implements billing after recieving balancing energy price

        Args:
            balancing_energy_msg (dict): message object
        """
        # recieve price for balancing energy for last trading period - influences
        # only longtime trading log, because trading log has already been updated
        log = self.trading_table_longtime[-1]

        # Connection to one network: combination of positive and negative clearing, e.g., storages
        energy_difference = (log['cleared_energy_pos'] - log['cleared_energy_neg']) - \
                            (log['real_energy_pos'] - log['real_energy_neg'])

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

    def __process_clearing(self, msg):
        """
        implements processing of clearing results

        Args:
            trade_msg (dict): message object
        """
        product_type = msg['product_type']

        # check for trade type and generate appending string to access trading table correctly
        if msg['trade_type'] == 'buy':
            trade_type = '_neg'
        elif msg['trade_type'] == 'sell':
            trade_type = '_pos'
        else:
            print(['[ERROR] Undefined trade type: ', msg['trade_type']])

        # minimum horizon that is covered by product, minimum 0
        min_horizon = int(msg['product_lead_time'] / self.trading_time)
        # maximum horizon that is covered by product
        max_horizon = int((msg['product_lead_time'] + msg['product_type']) / self.trading_time)
        # split quantity depending on product length
        quantity = msg['quantity'] / (max_horizon - min_horizon)

        for idx in range(min_horizon, max_horizon):
            # calculate mean price and cleared energy
            if self.trading_table[idx]['cleared_energy' + trade_type] != 0:
                self.trading_table[idx]['price' + trade_type] = (
                    self.trading_table[idx]['cleared_energy' + trade_type] *
                    self.trading_table[idx]['price' + trade_type] +
                    quantity * msg['price']) / (
                        self.trading_table[idx]['cleared_energy' + trade_type] + quantity)
                self.trading_table[idx]['price_' + str(product_type) + '_' + msg['sender_id']] = (
                    self.trading_table[idx][str(product_type) + '_' + msg['sender_id']] *
                    self.trading_table[idx]['price_' + str(product_type) + '_' + msg['sender_id']] +
                    quantity * msg['price'])/(
                        self.trading_table[idx][str(product_type) + '_' + msg['sender_id']] + quantity)
            else:
                self.trading_table[idx]['price' + trade_type] = msg['price']
                self.trading_table[idx]['price_' + str(product_type) + '_' + msg['sender_id']] = msg['price']
            self.trading_table[idx]['cleared_energy' +
                                    trade_type] = self.trading_table[idx]['cleared_energy' + trade_type] + quantity
            self.trading_table[idx][str(product_type) + '_' + msg['sender_id']] += quantity

    @abstractmethod
    def get_state(self, observation, control_step):
        """
        get state from environemt

        Args:
            observation (dict): observed values from environment
            control_step (int): number of control within trading trading time
        """
        # append observation to longtime log
        self.observations.append(observation)
        self.experiment_time = observation['time']
        self.scenario_time = observation['scenario_time']
        self.control_step = control_step

        # drop first element of trading table if control step is zero
        if control_step == 0:
            log = self.trading_table[0]
            log['time'] = self.experiment_time
            log['scneario_time'] = self.scenario_time
            self.trading_table_longtime.append(log)
            self.trading_table.append({var_name: 0 for var_name in self.trading_table[0]})
            self.trading_table.pop(0)

    def set_actions(self):
        """
        set actions from agent to environment to supervisory controller (environment)

        Returns:
            action (dict): set controlled variables as dictionary to supervisory controller
        """

        # cleared energy as producer
        cleared_energy = self.trading_table[0]['cleared_energy_pos'] - self.trading_table[0]['cleared_energy_neg']

        # enable power and energy controlled consumer
        if self.experiment_config["is_power_controlled"]:
            bSetStatusOn = 1.0 if cleared_energy != 0 else 0.0
            fSetPoint = abs(cleared_energy) * 3600 / self.trading_time
        else:
            real_energy = self.trading_table[0]['real_energy_pos'] - self.trading_table[0]['real_energy_neg']

            if abs(real_energy) > abs(cleared_energy):
                energy_difference = 0.01  # minimum value
            else:
                energy_difference = cleared_energy - real_energy

            # time left within this trading period
            time_left = (
                (self.trading_time / self.experiment_config['sampling_time']) - self.control_step
                ) * self.experiment_config['sampling_time']

            # set bSetStatusOn - false if energy_difference is 0
            bSetStatusOn = 1.0 if energy_difference != 0 and cleared_energy != 0 else 0.0

            # calculate mean power for left control time in trading period
            fSetPoint = (abs(energy_difference) * 3600) / (time_left)

        output_vars = self.agent_config['base_config']['env_outputs']
        action = {output_vars['bSetStatusOn']: bSetStatusOn, output_vars['fSetPoint']: fSetPoint}
        return action

    @abstractmethod
    def trade(self, product):
        """
        method to call trading process

        Args:
            product (int): traded product defined by product duration in seconds

        Returns:
            msgs (list): list of order messages which are processed to market
        """

    def _quantity_assessment(self, product):
        """
        quantity_assessment

        Args:
            product (int): traded product defined by product duration in seconds

        Returns:
            quantities (dict): dictionary holding tradable quantities and further information for pricing (e.g. soc)
        """
        # minimum horizon that is covered by product, minimum 0
        min_horizon = int(product['lead_time'] / self.trading_time)
        # maximum horizon that is covered by product
        max_horizon = int((product['lead_time'] + product['product_type']) / self.trading_time)

        # get cleared energy and calculate maximum cleared power for these horizons
        cleared_energy_pos = [element['cleared_energy_pos'] for element in self.trading_table]
        cleared_energy_neg = [element['cleared_energy_neg'] for element in self.trading_table]

        # get observation from last step
        observation = self.observations[-1]

        # get static model parameters
        physical_model_parameters = self.agent_config['model_config']['model_parameters']

        # get model inputs at this time step
        physical_model_inputs = {model_input: observation[model_input]
                                 for model_input in self.agent_config['base_config']['env_inputs']}
        physical_model_inputs['cleared_energy_pos'] = cleared_energy_pos[min_horizon:max_horizon]
        physical_model_inputs['cleared_energy_neg'] = cleared_energy_neg[min_horizon:max_horizon]
        physical_model_inputs['trading_time'] = self.trading_time
        physical_model_inputs['product_type'] = product['product_type']

        # get index of product in list
        if product['product_type'] == self.shorttime_product:
            physical_model_inputs['shorttime_product'] = True
        else:
            physical_model_inputs['shorttime_product'] = False

        # get current ambient temperature
        start_time = self.experiment_time + product['lead_time']
        end_time = start_time + product['product_type']
        df_ambient_temperature = self.ambient_temperature.get_values(start_time=start_time, end_time=end_time)
        mean_ambient_temperature = np.mean(df_ambient_temperature['value'].to_numpy())
        physical_model_inputs['ambient_temperature'] = mean_ambient_temperature

        # create dictionary with model parameters
        self.physical_parameters = {
            'model_parameters': physical_model_parameters,
            'model_inputs': physical_model_inputs
        }

    def _pricing(self, product, quantities):
        """
        pricing

        Args:
            product (int): traded product defined by product duration in seconds
            quantities (dict): dictionary holding tradable quantities and further information for pricing (e.g. soc)

        Returns:
            msgs (list): list of order messages which are processed to market
        """

        # get observation from last step
        observation = self.observations[-1]

        # get static model parameters
        trading_model_parameters = self.agent_config['model_config']['model_parameters']
        trading_model_parameters['markets'] = self.agent_config['base_config']['connections_markets']
        trading_model_parameters['name'] = self.name

        # get model inputs at this time step
        trading_model_inputs = {model_input: observation[model_input]
                                for model_input in self.agent_config['base_config']['env_inputs']}
        trading_model_inputs['quantities'] = quantities
        trading_model_inputs['product_type'] = product['product_type']
        trading_model_inputs['product_lead_time'] = product['lead_time']
        trading_model_inputs['positive_market_limit'] = self.experiment_config['positive_market_limit']
        trading_model_inputs['negative_market_limit'] = self.experiment_config['negative_market_limit']
        if product['product_type'] == self.shorttime_product:
            trading_model_inputs['shorttime_product'] = True
        else:
            trading_model_inputs['shorttime_product'] = False

        # minimum horizon that is covered by product, minimum 0
        min_horizon = int(product['lead_time'] / self.trading_time)
        # maximum horizon that is covered by product
        max_horizon = int((product['lead_time'] + product['product_type']) / self.trading_time)

        # get cleared energy and calculate maximum cleared power for these horizons
        cleared_energy_pos = [element['cleared_energy_pos'] for element in self.trading_table]
        cleared_energy_neg = [element['cleared_energy_neg'] for element in self.trading_table]
        price_pos = [element['price_pos'] for element in self.trading_table]
        price_neg = [element['price_neg'] for element in self.trading_table]
        trading_model_inputs['cleared_energy_pos'] = cleared_energy_pos[min_horizon:max_horizon]
        trading_model_inputs['cleared_energy_neg'] = cleared_energy_neg[min_horizon:max_horizon]
        trading_model_inputs['price_pos'] = price_pos[min_horizon:max_horizon]
        trading_model_inputs['price_neg'] = price_neg[min_horizon:max_horizon]

        self.pricing_parameters = {
            'model_parameters': trading_model_parameters,
            'model_inputs': trading_model_inputs
        }


if __name__ == '__main__':
    pass
