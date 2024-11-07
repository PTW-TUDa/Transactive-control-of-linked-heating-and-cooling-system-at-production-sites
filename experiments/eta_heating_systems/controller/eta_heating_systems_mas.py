"""
controller
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "multi agent system based controller"

from typing import Any
import os
import pathlib
import numpy as np
import pandas as pd
from eta_utility.eta_x.agents import RuleBased
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.vec_env import VecEnv
from datetime import datetime, timedelta
from multi_agent_system.components.market import Market
from multi_agent_system.components.converter import Converter
from multi_agent_system.components.consumer import Consumer
from multi_agent_system.components.system_operator import SystemOperator
from multi_agent_system.components.storage import Storage
from multi_agent_system.components.heat_exchanger import HeatExchanger
from multi_agent_system.components.heat_pump import HeatPump
from multi_agent_system.base.util import read_config, DynamicObject


class EtaHeatingSystemsMas(RuleBased):
    """
    Rule-based multi agent system for eta factory supply systems

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
    """

    def __init__(self, policy: type[BasePolicy], env: VecEnv, verbose: int = 1, **kwargs: Any):
        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        # extract action and observation names from the environments state_config
        self.action_names = self.env.envs[0].state_config.actions
        self.observation_names = self.env.envs[0].state_config.observations

        '''
        multi agent system
        '''
        # read mas config json
        from experiments.eta_heating_systems.config.global_ import run_name
        root_path = pathlib.Path(__file__).parents[1]
        config_path = os.path.join(root_path, "config", run_name + ".json")
        self.config = read_config(config_path)

        # read sampling and trading time -> control steps
        self.products = dict(zip(self.config["environment_specific"]["products"]
                             [0], self.config["environment_specific"]["products"][1]))
        self.sampling_time = self.config["environment_specific"]["sampling_time"]
        self.scenario_time = datetime.strptime(
            self.config["environment_specific"]["scenario_time_begin"],
            self.config['environment_specific']['date_format'])
        self.scenario_end_time = datetime.strptime(
            self.config["environment_specific"]["scenario_time_end"],
            self.config['environment_specific']['date_format'])
        self.time = 0
        self.min_product = min(list(self.products.keys()))
        self.ambient_temperature = DynamicObject(
            filename=self.config['environment_specific']['ambient_temperature'],
            sheet_name='ambient_temperature')

        # number of trading steps between shortest and longest product
        # longer products must be multiples of shortest product
        self.num_trading_steps = int(max(list(self.products.keys())) / self.min_product)
        self.trading_step = 0
        self.num_control_steps = int(self.min_product / self.sampling_time)
        self.control_step = 0

        # instantiate agents
        self.traders = {}
        self.markets = {}
        self.agent_inputs = {}
        for agent in self.config['agents']:
            if agent['type'] == 'market':
                self.markets[agent['name']] = Market(agent_name=agent['name'],
                                                     agent_type=agent['type'],
                                                     agent_config=agent['config'],
                                                     experiment_config=self.config['environment_specific'])
            elif agent['type'] == 'converter':
                self.traders[agent['name']] = Converter(agent_name=agent['name'],
                                                        agent_type=agent['type'],
                                                        agent_config=agent['config'],
                                                        experiment_config=self.config['environment_specific'])
            elif agent['type'] == 'consumer':
                self.traders[agent['name']] = Consumer(agent_name=agent['name'],
                                                       agent_type=agent['type'],
                                                       agent_config=agent['config'],
                                                       experiment_config=self.config['environment_specific'])
            elif agent['type'] == 'system_operator':
                self.traders[agent['name']] = SystemOperator(agent_name=agent['name'],
                                                             agent_type=agent['type'],
                                                             agent_config=agent['config'],
                                                             experiment_config=self.config['environment_specific'])
            elif agent['type'] == 'storage':
                self.traders[agent['name']] = Storage(agent_name=agent['name'],
                                                      agent_type=agent['type'],
                                                      agent_config=agent['config'],
                                                      experiment_config=self.config['environment_specific'])
            elif agent['type'] == 'heat_exchanger':
                self.traders[agent['name']] = HeatExchanger(agent_name=agent['name'],
                                                            agent_type=agent['type'],
                                                            agent_config=agent['config'],
                                                            experiment_config=self.config['environment_specific'])
            elif agent['type'] == 'heat_pump':
                self.traders[agent['name']] = HeatPump(agent_name=agent['name'],
                                                       agent_type=agent['type'],
                                                       agent_config=agent['config'],
                                                       experiment_config=self.config['environment_specific'])
            else:
                print('[ERROR] ', agent['name'], ': agent type does not exist.')

            # write agent inputs to dict
            self.agent_inputs[agent['name']] = agent['config']['base_config']['env_inputs']

        # start agents
        for (_, market) in self.markets.items():
            market.setup_agent()
        for (_, trader) in self.traders.items():
            trader.setup_agent()

        self.fHeatEnergy_WMZ300 = 0

    def control_rules(self, observation):
        """
        Controller of the model.

        :param observation: Observation from the environment.
        :returns: actions
        """

        # prepare observations
        observation = self.__prepare_observations(observation=observation)

        # set global actions
        action = self.__set_global_actions(observation=observation)

        # check for benchmark scenario and overwrite bAlgortihmModeActivated
        if not self.config["environment_specific"]["is_benchmark_scenario"]:
            # set actions from multi agent system (bSetStatusOn, fSetPoint)
            action = self.__set_mas_actions(observation=observation, action=action)

        # proceed in time
        self.scenario_time += timedelta(seconds=self.sampling_time)
        self.time += self.sampling_time

        # terminate experiment by last billing balancing energy and result export
        if self.scenario_time == self.scenario_end_time:
            self.__terminate_experiment()

        # return actions to environment
        actions = []
        actions.append(list(action.values()))
        actions = actions[0]
        # print(actions)
        return np.array(actions)

    def __prepare_observations(self, observation):
        """
        prepare observations for control process

        Args:
            observation (list): list with observations from environemnt

        Returns:
            observation (dict): dictionary with prepared observations
        """

        # get observations from env and add information for multi agent system
        observation = dict(zip(self.observation_names, observation))
        observation['scenario_time'] = self.scenario_time
        observation['time'] = self.time
        observation["fFeedTemperature_HNHT"] = 65
        observation["fFeedTemperature_HNLT_Heating"] = 40
        observation["fFeedTemperature_CN"] = 15
        observation["Strategy.localSetParameters.bHeatingModeActivated"] = self.config[
            'environment_specific']['heating_mode']

        # compressor heat counter is not installed as consumer but as producer
        observation["HNLT.CompressorSystem.WMZ251.sensorState.fHeatEnergy"] = -observation[
            "HNLT.CompressorSystem.WMZ251.sensorState.fHeatEnergy"]

        if self.config['environment_specific']['is_live_env']:
            observation["HNHT_HNLT.HeatPump1System.P_el"] = observation["HNHT_HNLT.HeatPump1System.P_el"]/1000
            observation["HNLT_CN.HeatPump2System.P_el"] = observation["HNLT_CN.HeatPump2System.P_el"]/1000
            observation["CN.eChillerSystem.P_el"] = observation["CN.eChillerSystem.P_el"]
            self.fHeatEnergy_WMZ300 += observation[
                "HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatFlowRate"]*self.sampling_time/3600000
            observation["HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatEnergy"] = self.fHeatEnergy_WMZ300
            print(observation)
            print(self.fHeatEnergy_WMZ300)
        return observation

    def __set_global_actions(self, observation):
        """
        set global (mas independet) actions

        Args:
            observation (dict): dictionary with observations from environment

        Returns:
            action (dict): dictionary with global actions
        """
        # intialize action dict
        action = dict.fromkeys(self.action_names, 0)

        # get current ambient temperature
        start_time = self.time
        end_time = start_time + 900
        df_ambient_temperature = self.ambient_temperature.get_values(start_time=start_time, end_time=end_time)
        mean_ambient_temperature = np.mean(df_ambient_temperature['value'].to_numpy())
        # ambient temperature is only set to virtual env
        if not self.config["environment_specific"]["is_live_env"]:
            action["T_ambient"] = mean_ambient_temperature

        # active production and heating mode
        if self.config["environment_specific"]["production_mode"]:
            action["Strategy.localSetParameters.bProductionModeActivated"] = 1
        else:
            action["Strategy.localSetParameters.bProductionModeActivated"] = 0

        if not self.config['environment_specific']['is_live_env']:
            if self.config["environment_specific"]["heating_mode"]:
                action["Strategy.localSetParameters.bHeatingModeActivated"] = 1
            else:
                action["Strategy.localSetParameters.bHeatingModeActivated"] = 0

        # return action array with zeros for benchmark scenario
        if self.config["environment_specific"]["is_benchmark_scenario"]:
            return action

        # chp1 system
        action["HNHT.CHP1System.control.bAlgorithmModeActivated"] = 1
        if self.config['environment_specific']['is_live_env']:
            action["HNHT.CHP1System.setSetPoint.fSetPointAlgorithm"] = observation["fFeedTemperature_HNHT"]
        else:
            action["HNHT.CHP1System.setSetPoint.fSetPointAlgorithm"] = observation["fFeedTemperature_HNHT"] + 273.15
        action["HNHT.CHP1System.RV32x.control.bAlgorithmModeActivated"] = 1
        action["HNHT.CHP1System.RV32x.localSetParameters.nControlModeAlgorithm"] = 3
        # chp2 system
        action["HNHT.CHP2System.control.bAlgorithmModeActivated"] = 1
        if self.config['environment_specific']['is_live_env']:
            action["HNHT.CHP2System.setSetPoint.fSetPointAlgorithm"] = observation["fFeedTemperature_HNHT"]
        else:
            action["HNHT.CHP2System.setSetPoint.fSetPointAlgorithm"] = observation["fFeedTemperature_HNHT"] + 273.15
        action["HNHT.CHP2System.RV32x.control.bAlgorithmModeActivated"] = 1
        action["HNHT.CHP2System.RV32x.localSetParameters.nControlModeAlgorithm"] = 3
        # condensing boiler system
        action["HNHT.CondensingBoilerSystem.control.bAlgorithmModeActivated"] = 1
        if self.config['environment_specific']['is_live_env']:
            action["HNHT.CondensingBoilerSystem.setSetPoint.fSetPointAlgorithm"] = observation[
                "fFeedTemperature_HNHT"]
        else:
            action["HNHT.CondensingBoilerSystem.setSetPoint.fSetPointAlgorithm"] = observation[
                "fFeedTemperature_HNHT"] + 273.15
        action["HNHT.CondensingBoilerSystem.RV331.control.bAlgorithmModeActivated"] = 1
        action["HNHT.CondensingBoilerSystem.RV331.localSetParameters.nControlModeAlgorithm"] = 3
        # vsi storage system
        action["HNHT.VSIStorageSystem.control.bAlgorithmModeActivated"] = 1
        action["HNHT.VSIStorageSystem.setSetPoint.fSetPointAlgorithm"] = 30
        action["HNHT.VSIStorageSystem.SV307.control.bAlgorithmModeActivated"] = 1
        action["HNHT.VSIStorageSystem.SV307.localSetParameters.nControlModeAlgorithm"] = 3
        # central machine heating system
        action["HNHT.CentralMachineHeatingSystem.control.bAlgorithmModeActivated"] = 1  # uncontrollable demand
        action["HNHT.CentralMachineHeatingSystem.setSetPoint.fSetPointAlgorithm"] = 0  # not used
        action["HNHT.CentralMachineHeatingSystem.PU300.control.bAlgorithmModeActivated"] = 1  # uncontrollable demand
        # control thermal power by pump
        action["HNHT.CentralMachineHeatingSystem.PU300.localSetParameters.nControlModeAlgorithm"] = 6
        # static heating system
        action["HNHT.StaticHeatingSystem.control.bAlgorithmModeActivated"] = 1
        if not self.config['environment_specific']['is_live_env']:
            action["HNHT.StaticHeatingSystem.SV.control.bAlgorithmModeActivated"] = 1
            action["HNHT.StaticHeatingSystem.SV.control.bSetStatusOnAlgorithm"] = 1  # SV is always open
            action["HNHT.StaticHeatingSystem.SV.setSetPoint.fSetPointAlgorithm"] = 100  # SV is always open
            action["HNHT.StaticHeatingSystem.SV.localSetParameters.nControlModeAlgorithm"] = 0
        action["HNHT.StaticHeatingSystem.RV350.control.bAlgorithmModeActivated"] = 1
        action["HNHT.StaticHeatingSystem.RV350.localSetParameters.nControlModeAlgorithm"] = 3
        # heat pump system
        action["HNHT_HNLT.HeatPump1System.control.bAlgorithmModeActivated"] = 1
        if self.config['environment_specific']['is_live_env']:
            action["HNHT_HNLT.HeatPump1System.setSetPoint.fSetPointAlgorithm"] = observation[
                "fFeedTemperature_HNHT"]
        else:
            action["HNHT_HNLT.HeatPump1System.setSetPoint.fSetPointAlgorithm"] = observation[
                "fFeedTemperature_HNHT"] + 273.15
        action["HNHT_HNLT.HeatPump1System.RV342.control.bAlgorithmModeActivated"] = 1
        action["HNHT_HNLT.HeatPump1System.RV342.localSetParameters.nControlModeAlgorithm"] = 3
        # heat exchanger 1 system
        action["HNHT_HNLT.HeatExchanger1System.control.bAlgorithmModeActivated"] = 1
        action["HNHT_HNLT.HeatExchanger1System.setSetPoint.fSetPointAlgorithm"] = 0  # not controlled
        action["HNHT_HNLT.HeatExchanger1System.RV315.control.bAlgorithmModeActivated"] = 1
        action["HNHT_HNLT.HeatExchanger1System.RV315.localSetParameters.nControlModeAlgorithm"] = 3
        # hvfa system
        action["HNLT.HVFASystem.control.bAlgorithmModeActivated"] = 1
        action["HNLT.HVFASystem.setSetPoint.fSetPointAlgorithm"] = 70
        action["HNLT.HVFASystem.RVx05.control.bAlgorithmModeActivated"] = 1
        action["HNLT.HVFASystem.RVx05.localSetParameters.nControlModeAlgorithm"] = 3
        # compressor system
        action["HNLT.CompressorSystem.control.bAlgorithmModeActivated"] = 1
        action["HNLT.CompressorSystem.setSetPoint.fSetPointAlgorithm"] = 0  # not controlled
        action["HNLT.CompressorSystem.RV251.control.bAlgorithmModeActivated"] = 1
        action["HNLT.CompressorSystem.RV251.localSetParameters.nControlModeAlgorithm"] = 3
        # inner capillary tube mats
        action["HNLT_CN.InnerCapillaryTubeMats.control.bAlgorithmModeActivated"] = 1
        if not self.config['environment_specific']['is_live_env']:
            action["HNLT_CN.InnerCapillaryTubeMats.SV.control.bAlgorithmModeActivated"] = 1
            action["HNLT_CN.InnerCapillaryTubeMats.SV.control.bSetStatusOnAlgorithm"] = 1  # SV is always open
            action["HNLT_CN.InnerCapillaryTubeMats.SV.setSetPoint.fSetPointAlgorithm"] = 100  # SV is always open
            action["HNLT_CN.InnerCapillaryTubeMats.SV.localSetParameters.nControlModeAlgorithm"] = 0
        else:
            # force valves open by system setpoint
            if self.config["environment_specific"]["heating_mode"]:
                action["HNLT_CN.InnerCapillaryTubeMats.setSetPoint.fSetPointAlgorithm"] = 40
            else:
                action["HNLT_CN.InnerCapillaryTubeMats.setSetPoint.fSetPointAlgorithm"] = 10
        action["HNLT_CN.InnerCapillaryTubeMats.RV500.control.bAlgorithmModeActivated"] = 1
        action["HNLT_CN.InnerCapillaryTubeMats.RV500.localSetParameters.nControlModeAlgorithm"] = 3
        # underfloor heating
        action["HNLT_CN.UnderfloorHeatingSystem.control.bAlgorithmModeActivated"] = 1
        if not self.config['environment_specific']['is_live_env']:
            action["HNLT_CN.UnderfloorHeatingSystem.SV.control.bAlgorithmModeActivated"] = 1
            action["HNLT_CN.UnderfloorHeatingSystem.SV.control.bSetStatusOnAlgorithm"] = 1  # SV is always open
            action["HNLT_CN.UnderfloorHeatingSystem.SV.setSetPoint.fSetPointAlgorithm"] = 100  # SV is always open
            action["HNLT_CN.UnderfloorHeatingSystem.SV.localSetParameters.nControlModeAlgorithm"] = 0
        action["HNLT_CN.UnderfloorHeatingSystem.RV425.control.bAlgorithmModeActivated"] = 1
        action["HNLT_CN.UnderfloorHeatingSystem.RV425.localSetParameters.nControlModeAlgorithm"] = 3
        # heat exchanger 6 system (outer capillary tube mats)
        action["HNLT.OuterCapillaryTubeMats.control.bAlgorithmModeActivated"] = 1
        action["HNLT.OuterCapillaryTubeMats.setSetPoint.fSetPointAlgorithm"] = 0  # not controlled
        action["HNLT.OuterCapillaryTubeMats.RV600.control.bAlgorithmModeActivated"] = 1
        action["HNLT.OuterCapillaryTubeMats.RV600.localSetParameters.nControlModeAlgorithm"] = 3
        # heat pump system
        action["HNLT_CN.HeatPump2System.control.bAlgorithmModeActivated"] = 1
        # internally controlled by heat pump
        action["HNLT_CN.HeatPump2System.setSetPoint.fSetPointAlgorithm"] = observation[
            "fFeedTemperature_HNLT_Heating"] + 273.15
        # eChiller system
        action["CN.eChillerSystem.control.bAlgorithmModeActivated"] = 1
        if self.config['environment_specific']['is_live_env']:
            action["CN.eChillerSystem.setSetPoint.fSetPointAlgorithm"] = observation["fFeedTemperature_CN"]
        else:
            action["CN.eChillerSystem.setSetPoint.fSetPointAlgorithm"] = observation["fFeedTemperature_CN"] + 273.15
        action["CN.eChillerSystem.SV138.control.bAlgorithmModeActivated"] = 1
        action["CN.eChillerSystem.SV138.localSetParameters.nControlModeAlgorithm"] = 3
        # central machine cooling system
        action["CN.CentralMachineCoolingSystem.control.bAlgorithmModeActivated"] = 1
        action["CN.CentralMachineCoolingSystem.setSetPoint.fSetPointAlgorithm"] = 0  # not used
        action["CN.CentralMachineCoolingSystem.PU100.control.bAlgorithmModeActivated"] = 1
        # control thermal power by pump
        action["CN.CentralMachineCoolingSystem.PU100.localSetParameters.nControlModeAlgorithm"] = 6
        # hvfa system
        action["CN.HVFASystem.control.bAlgorithmModeActivated"] = 1
        action["CN.HVFASystem.setSetPoint.fSetPointAlgorithm"] = 40  # maximum pump speed
        action["CN.HVFASystem.RVx05.control.bAlgorithmModeActivated"] = 1
        action["CN.HVFASystem.RVx05.localSetParameters.nControlModeAlgorithm"] = 3

        return action

    def __set_mas_actions(self, observation, action):
        """
        set mas specific actions

        Args:
            dictionary with observations from environment
            action (dict): dictionary with global actions

        Returns:
            action (dict): dictionary with global and mas actions
        """

        # get state
        for agent_name, agent in self.traders.items():
            agent_inputs = self.agent_inputs[agent_name]

            agent_observation = {}
            for input_name, input_value in agent_inputs.items():
                # heat energy is recieved in MWh and must be converted to kWh
                if input_name == 'fHeatEnergy':
                    agent_observation[input_name] = observation[input_value] * 1000
                else:
                    agent_observation[input_name] = observation[input_value]
            agent_observation['scenario_time'] = self.scenario_time
            agent_observation['time'] = self.time
            agent.get_state(agent_observation, self.control_step)

        # trading, clearing, logging
        if self.control_step == 0:
            # tradable products in this trading time step, 0 entry is always balancing energy
            if self.trading_step == 0:
                tradable_products = self.products
            else:
                tradable_products = {
                    product: lead_times for (
                        product, lead_times) in self.products.items() if (
                        self.trading_step) %
                    (product / self.min_product) == 0}

            # descending sorted list of tradable products -> trade and clear longtime product before shorttime product
            for product in sorted(list(tradable_products.keys()), reverse=True):
                # lead time of traded product
                lead_times = self.products[product]

                # iterate over lead time and start with longest lead time
                for lead_time in sorted(lead_times, reverse=True):
                    # return bids from trading process as msgs
                    trader_msgs = [
                        agent.trade(
                            product={
                                'product_type': product,
                                'lead_time': lead_time}) for _,
                        agent in self.traders.items()]

                    # iterate over msgs and mpa them to markets
                    for msg in trader_msgs:
                        for sub_msg in msg:
                            eval("self.markets['" + sub_msg['reciever_id'] + "'].process_msg(msg=sub_msg)")

                    # perform clearing and return market messages
                    market_msgs = [
                        agent.clear(
                            product={
                                'product_type': product,
                                'lead_time': lead_time},
                            experiment_time=self.scenario_time) for _,
                        agent in self.markets.items()]

                    # iterate over market messages and map them to traders
                    for msg in market_msgs:
                        for sub_msg in msg:
                            eval("self.traders['" + sub_msg['reciever_id'] + "'].process_msg(msg=sub_msg)")

            # return balancing energy price from last trading period and pass them to market participants
            balancing_energy_msgs = [agent.return_balancing_energy_price() for _,
                                     agent in self.traders.items() if agent.return_agent_type() == 'system_operator']
            for msg in balancing_energy_msgs:
                for sub_msg in msg:
                    eval("self.traders['" + sub_msg['reciever_id'] + "'].process_msg(msg=sub_msg)")

            # raise trading step
            self.trading_step += 1

        # set actions
        for _, agent in self.traders.items():
            actions = agent.set_actions()
            # map agent actions to global environment actions
            for key, val in actions.items():
                if key.split('.')[-1] == 'fSetPointAlgorithm':
                    if self.config['environment_specific']['is_live_env']:
                        action[key] = val
                    else:
                        action[key] = val * 1000  # thermal power must be converted from kW to W
                else:
                    action[key] = val

                # write bSetStatusOnAlgorithm also to system
                if key.split('.')[-1] == 'bSetStatusOnAlgorithm':
                    key_system = key.split('.', 2)[0] + '.' + key.split('.', 2)[1] + '.control.bSetStatusOnAlgorithm'
                    action[key_system] = val

        # reset control step to execute trading and clearing in next step
        self.control_step += 1
        if self.control_step == self.num_control_steps:
            self.control_step = 0
            if self.config['environment_specific']['is_live_env']:
                self.__logging()

        # reset trading step
        if self.trading_step == self.num_trading_steps:
            self.trading_step = 0

        return action

    def __logging(self):
        """
        save trading results to external file
        """
        # write longtime trading log of agents to excel
        root_path = pathlib.Path(__file__).parents[1]
        from experiments.eta_heating_systems.config.global_ import series_name, run_name
        result_path = os.path.join(root_path, "results", series_name, run_name + ".xlsx")
        writer = pd.ExcelWriter(
            result_path,
            engine="xlsxwriter")
        # iterate over traders
        for agent_name, agent in self.traders.items():
            df = agent.return_trading_table_longtime()
            df.to_excel(writer, sheet_name=agent_name)

        # iterate over markets
        for agent_name, agent in self.markets.items():
            df = agent.return_trading_table_longtime()
            df.to_excel(writer, sheet_name=agent_name)
        writer.close()

    def __terminate_experiment(self):
        """
        terminate experiment by last billing of balancing energy and result export
        """

        if not self.config["environment_specific"]["is_benchmark_scenario"]:
            balancing_energy_msgs = [
                agent.return_balancing_energy_price()
                for _, agent in self.traders.items()
                if agent.return_agent_type() == 'buffer_storage'
            ]
            for msg in balancing_energy_msgs:
                for sub_msg in msg:
                    eval("self.traders['" + sub_msg['reciever_id'] + "'].process_msg(msg=sub_msg)")

        self.__logging()
