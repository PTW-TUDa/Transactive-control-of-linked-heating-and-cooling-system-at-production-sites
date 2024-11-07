"""
utility functions within project
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "utility functions within project"

import os
import json
import pandas as pd
import numpy as np


def read_config(file_path):
    """
    Read agent config

    Args:
        file_path (str): file path to json config file

    Returns:
        json_data (dict): json data as dictionary
    """
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


class DynamicObject():
    def __init__(self, filename, sheet_name):
        """
        dynamic object which allows reading dynamic values from xlsx file

        Args:
            filename (str): path to external file from root path
            sheet_name (str): name of excel sheet in workbook
        """

        # check if filename is a string, otherwise static value
        if isinstance(filename, str):
            self.dynamic = True
            df = pd.read_excel(filename, sheet_name=sheet_name)
            df['combined'] = df['date'].astype(str) + ' ' + df['start_time'].astype(str)
            df["datetime"] = pd.to_datetime(df['combined'], format='%Y-%m-%d %H:%M:%S')
            df.drop(['date', 'start_time', 'end_time', 'combined'], axis=1, inplace=True)
        else:
            self.dynamic = False
            data = {'combined': ["01.01.1900 00:00"], 'value': [filename]}
            df = pd.DataFrame.from_dict(data)
            df["datetime"] = pd.to_datetime(df['combined'], format='%d.%m.%Y %H:%M')
            df["seconds"] = 0

        # save values as dataframe
        df.set_index('seconds', inplace=True, drop=True)
        self.df = df

    def is_dynamic(self):
        """
        returns wheter object is dynamic or static

        Returns:
            dynamic (bool): true if object allows dynamic access to values
        """
        return self.dynamic

    def get_values(self, start_time, end_time):
        """
        get value by start and endtime

        Returns:
            value (float): price
        """
        if self.dynamic:
            df = self.df[np.logical_and((self.df.index.get_level_values(0) >= start_time),
                                        (self.df.index.get_level_values(0) < end_time))]
        else:
            df = self.df
        return df


if __name__ == '__main__':
    # test - read config
    current_path = os.getcwd()
    config_path = os.path.join(current_path, "experiments", "eta_heating_systems", "config", "2024_04_09_1_day_s0.json")
    config = read_config(config_path)

    # test - create price_object
    ambient_temperature = DynamicObject(
        filename=config['environment_specific']['ambient_temperature'],
        sheet_name='ambient_temperature')
    start_time = '2024-04-09 07:00:00'
    end_time = '2024-04-09 07:15:00'
    res = ambient_temperature.get_values(start_time=start_time, end_time=end_time)
    print(np.mean(res['value'].to_numpy()))
