"""
base agent
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "base agent"

from abc import abstractmethod
import pandas as pd


class BaseAgent():
    def __init__(self, agent_name, agent_type, agent_config, experiment_config):
        """
        agent base class

        Args:
            agent_name (str): agent name
            agent_type (str): agent type (child class)
            agent_config (dict): dictionary with global information about experimental setup
            experiment_config (dict): dictionary with agent specific information
        """

        self.agent_config = agent_config
        self.type = agent_type
        self.name = agent_name
        self.observations = []  # list for logging observations of every agent sampling time
        self.experiment_config = experiment_config
        self.trading_table_longtime = []
        self.products = dict(zip(self.experiment_config["products"][0], self.experiment_config["products"][1]))
        self.trading_time = min(list(self.products.keys()))

    def return_agent_type(self):
        """
        returns agent type

        Returns:
            agent_type (str): agenty type (child class) as string
        """
        return self.type

    def return_trading_table_longtime(self):
        """
        return longtime trading table

        Returns:
            trading_table_longtime (df): longtime trading table as pandsa dataframe
        """
        df = pd.DataFrame.from_records(self.trading_table_longtime)
        return df

    @abstractmethod
    def setup_agent(self):
        """
        abtract method for setting up agent
        """

    @abstractmethod
    def process_msg(self, msg):
        """
        abtract method for processing incoming messages
        method is called whenever agent recieves a message

        Args:
            msg (object): incoming message
        """
