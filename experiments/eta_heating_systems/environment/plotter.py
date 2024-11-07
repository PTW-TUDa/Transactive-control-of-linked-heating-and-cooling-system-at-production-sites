import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from multi_agent_system.base.util import read_config, DynamicObject


class Plotter():
    def __init__(self, result_path, config_path, fig_size):

        # global settings
        font_path = 'C:\\Windows\\Fonts\\lmroman10-regular.otf'
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()

        plt.rcParams["font.size"] = 9
        plt.rcParams['lines.linewidth'] = 0.9
        self.fig_size = fig_size
        self.fig_dpi = 400
        self.linestyles = ["solid", "dashed", "dashdot", "dotted",
                           (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
        self.markers = ['o', 's', '^']
        self.result_path = result_path

        # read parameters from self.config file
        self.config = read_config(config_path)
        self.sampling_time = self.config["environment_specific"]["sampling_time"]
        self.scenario_time_begin = datetime.strptime(
            self.config["environment_specific"]["scenario_time_begin"],
            self.config['environment_specific']['date_format'])
        self.scenario_time_end = datetime.strptime(
            self.config["environment_specific"]["scenario_time_end"],
            self.config['environment_specific']['date_format'])

        # get self.config of supply systems
        self.networks = {
            'HNHT': self.config['agents'][0]["config"]['base_config']['connections_traders'],
            'HNLT': self.config['agents'][1]["config"]['base_config']['connections_traders'],
            'CN': self.config['agents'][2]["config"]['base_config']['connections_traders'],
        }

        # define system names for plotting
        self.system_names = {
            "BufferStorage_HNHT": "BS_HNHT",
            "VSIStorageSystem": "ST_HNHT",
            "CHP1System": "CHP1",
            "CHP2System": "CHP2",
            "CondensingBoilerSystem": "CB",
            "StaticHeatingSystem": "BH",
            "CentralMachineHeatingSystem": "CMH",
            "HeatExchanger1System": "HEX1",
            "HeatPump1System": "HP1",
            "BufferStorage_HNLT": "BS_HNLT",
            "HVFASystem_HNLT": "ST_HNLT",
            "OuterCapillaryTubeMats": "OCTM",
            "CompressorSystem": "COMP",
            "UnderfloorHeatingSystem": "UFH",
            "InnerCapillaryTubeMats": "ICTM",
            "HeatPump2System": "HP2",
            "BufferStorage_CN": "BS_CN",
            "HVFASystem_CN": "ST_CN",
            "eChillerSystem": "CU",
            "CentralMachineCoolingSystem": "CMC"
        }

        # load state variables
        self.episode_df = pd.read_excel(self.result_path, sheet_name='StateVars')
        # experiment duration in hours
        self.experiment_duration = len(self.episode_df['scenario_time']) * self.sampling_time / 3600

        # x axis ticks
        self.ticknames = []
        self.tickpos = []
        for step in range(len(self.episode_df)):
            tickdate = self.scenario_time_begin + timedelta(seconds=step * self.sampling_time)
            if tickdate.minute == 0 and tickdate.second == 0:
                self.tickpos.append(step)
                self.ticknames.append(tickdate.strftime("%H:%M"))
            elif tickdate.hour == 0 and tickdate.minute == 0 and tickdate.second == 0:
                self.tickpos.append(step)
                self.ticknames.append(tickdate.strftime("%d.%m"))

    def total_system_figure(self):
        """
        figure containing metrics for overall system

        Returns:
            fig: figure with total system characteristics
        """

        # create figure and subplots
        fig = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi)
        axes = [fig.add_subplot(4, 1, 1), fig.add_subplot(4, 1, 2), fig.add_subplot(4, 1, (3, 4))]
        axes[0].set_title("Performance | Total system", loc="center", fontdict={'fontweight': 'bold'})
        axes[1].set_title("Downtimes | Supply systems", loc="center", fontdict={'fontweight': 'bold'})
        axes[2].set_title("Utilization | Single systems", loc="center", fontdict={'fontweight': 'bold'})

        # adjust layout
        plt.tight_layout()
        fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.95, wspace=0.3, hspace=0.1)

        obj_gas_cost = DynamicObject(filename=self.config["environment_specific"]["cost_fuel"], sheet_name=0)
        obj_electricity_cost = DynamicObject(filename=self.config["environment_specific"]["cost_electricity"],
                                             sheet_name=0)
        chp_renumeration = self.config["environment_specific"]["chp_renumeration"]
        obj_electricity_demand = DynamicObject(
            filename=self.config["environment_specific"]["electricity_demand"], sheet_name=0)

        # get electricity, gas cost and electricity demand as numpy array
        start_time = 0
        end_time = self.config["settings"]["episode_duration"]
        df_electricity_cost = obj_electricity_cost.get_values(
            start_time=start_time, end_time=end_time)
        electricity_cost = df_electricity_cost['value'].to_numpy()
        df_gas_cost = obj_gas_cost.get_values(start_time=start_time, end_time=end_time)
        gas_cost = df_gas_cost['value'].to_numpy()
        df_electricity_demand = obj_electricity_demand.get_values(
            start_time=start_time, end_time=end_time)
        # if not dynamic -> kWh, value is negative in xlsx file
        electricity_demand_ext = abs(df_electricity_demand['value'].to_numpy())

        # if electricity demand is dynamic - calculate average value of state vars depending on discretization
        episode_electricity_generation = self.episode_df['HNHT.CHP1System.P_el'] + \
            self.episode_df['HNHT.CHP2System.P_el']
        episode_electricity_demand = self.episode_df['HNHT_HNLT.HeatPump1System.P_el'] + \
            self.episode_df['HNLT_CN.HeatPump2System.P_el'] + self.episode_df['CN.eChillerSystem.P_el']
        episode_gas_demand = self.episode_df['HNHT.CHP1System.P_gas'] + \
            self.episode_df['HNHT.CHP2System.P_gas'] + self.episode_df['HNHT.CondensingBoilerSystem.P_gas']

        # reshape results based on sampling time in environment and sampling time in electricity demand file
        if obj_electricity_demand.is_dynamic():
            discretization_electricity_demand = (
                df_electricity_demand.index[1] -
                df_electricity_demand.index[0])
            num_rows = len(self.episode_df) / (discretization_electricity_demand / self.sampling_time)
            if num_rows % 2 == 0:
                num_rows = int(num_rows) - 1
            else:
                num_rows = int(num_rows)

            # calculate the number of columns
            num_columns = int(discretization_electricity_demand / self.sampling_time)

            # calculate the total number of elements needed for reshaping
            total_elements = num_rows * num_columns
            electricity_generation_arr = episode_electricity_generation[:total_elements].values.reshape(
                (num_rows, num_columns))
            electricity_generation = electricity_generation_arr.mean(
                axis=1) * discretization_electricity_demand / 3600
            electricity_demand_arr = episode_electricity_demand[:total_elements].values.reshape(
                (num_rows, num_columns))
            electricity_demand = (electricity_demand_ext + electricity_demand_arr.mean(axis=1)
                                  ) * discretization_electricity_demand / 3600
        else:
            electricity_generation = np.mean(
                episode_electricity_generation[:-1]) * self.experiment_duration  # get energy in kWh
            electricity_demand = (
                np.mean(episode_electricity_demand[:-1]) + electricity_demand_ext) * self.experiment_duration

        # if gas cost is dynamic - calculate average value of state vars depending on discretization
        if obj_gas_cost.is_dynamic():
            discretization_gas_cost = (df_gas_cost.index[1] - df_gas_cost.index[0])
            gas_demand_arr = episode_gas_demand.values.reshape(
                (int(len(self.episode_df) / (discretization_gas_cost / self.sampling_time)),
                 int(discretization_gas_cost / self.sampling_time)))
            gas_demand = gas_demand_arr.mean(axis=1) * discretization_gas_cost / 3600
        else:
            gas_demand = np.mean(episode_gas_demand[:-1]) * self.experiment_duration

        # if electricity is fed into grid -> chp renumeration (negative costs) else electricity demand is displaced
        electrical_diff = electricity_demand + electricity_generation
        elec_cost = np.sum(
            np.where(
                electrical_diff < 0,
                electrical_diff *
                chp_renumeration,
                electrical_diff *
                electricity_cost))
        gas_cost = np.sum(gas_demand * gas_cost)
        total_costs = elec_cost + gas_cost

        # generated, consumed and stored heat energy
        if self.config["environment_specific"]['heating_mode']:
            ufh_heat = np.mean(self.episode_df['HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatFlowRate'])
            ict_heat = np.mean(self.episode_df['HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatFlowRate'])
            ufh_cool = 0
            ict_cool = 0
        else:
            ufh_heat = 0
            ict_heat = 0
            ufh_cool = abs(np.mean(self.episode_df['HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatFlowRate']))
            ict_cool = abs(np.mean(self.episode_df['HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatFlowRate']))

        heat_energy_demand = (
            np.mean(self.episode_df['HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatFlowRate']) +
            np.mean(self.episode_df['HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatFlowRate']) +
            ufh_heat +
            ict_heat
        ) * self.experiment_duration

        vsi_temp_begin = (self.episode_df["HNHT.VSIStorageSystem.VSIStorage.localState.fUpperTemperature"].iloc[0] +
                          self.episode_df["HNHT.VSIStorageSystem.VSIStorage.localState.fLowerTemperature"].iloc[0]) / 2
        vsi_temp_end = (self.episode_df["HNHT.VSIStorageSystem.VSIStorage.localState.fUpperTemperature"].iloc[-1] +
                        self.episode_df["HNHT.VSIStorageSystem.VSIStorage.localState.fLowerTemperature"].iloc[-1]) / 2
        hvfa_hnlt_temp_begin = (self.episode_df["HNLT.HVFASystem.HVFAStorage.localState.fUpperTemperature"].iloc[0] +
                                self.episode_df["HNLT.HVFASystem.HVFAStorage.localState.fLowerTemperature"].iloc[0]) / 2
        hvfa_hnlt_temp_end = (self.episode_df["HNLT.HVFASystem.HVFAStorage.localState.fUpperTemperature"].iloc[-1] +
                              self.episode_df["HNLT.HVFASystem.HVFAStorage.localState.fLowerTemperature"].iloc[-1]) / 2
        hvfa_cn_temp_begin = (self.episode_df["CN.HVFASystem.HVFAStorage.localState.fUpperTemperature"].iloc[0] +
                              self.episode_df["CN.HVFASystem.HVFAStorage.localState.fLowerTemperature"].iloc[0]) / 2
        hvfa_cn_temp_end = (self.episode_df["CN.HVFASystem.HVFAStorage.localState.fUpperTemperature"].iloc[-1] +
                            self.episode_df["CN.HVFASystem.HVFAStorage.localState.fLowerTemperature"].iloc[-1]) / 2

        energy_stored_building = (
            4230 * (self.episode_df["HNHT.StaticHeatingSystem.ConsumerTemperature.Celsius"].iloc[-1] -
                    self.episode_df["HNHT.StaticHeatingSystem.ConsumerTemperature.Celsius"].iloc[0]) +
            132000 * (self.episode_df["HNLT_CN.InnerCapillaryTubeMats.setPointState.fOperatingPoint"].iloc[-1] -
                      self.episode_df["HNLT_CN.InnerCapillaryTubeMats.setPointState.fOperatingPoint"].iloc[0]) +
            7200 * (self.episode_df["HNLT_CN.UnderfloorHeatingSystem.ConsumerTemperature.Celsius"].iloc[-1] -
                    self.episode_df["HNLT_CN.UnderfloorHeatingSystem.ConsumerTemperature.Celsius"].iloc[0])
        )

        if self.config["environment_specific"]['heating_mode'] and energy_stored_building > 0:
            heat_energy_stored_building = energy_stored_building
            cool_energy_stored_building = 0
        elif not self.config["environment_specific"]['heating_mode'] and energy_stored_building < 0:
            cool_energy_stored_building = energy_stored_building
            heat_energy_stored_building = 0
        else:
            cool_energy_stored_building = 0
            heat_energy_stored_building = 0

        heat_energy_stored = (
            7000 * 4.12 * (vsi_temp_end - vsi_temp_begin) +
            1000 * 4.12 * (self.episode_df["HNHT.localState.fMidTemperature"].iloc[-1] -
                           self.episode_df["HNHT.localState.fMidTemperature"].iloc[0]) +
            25000 * 4.12 * (hvfa_hnlt_temp_end - hvfa_hnlt_temp_begin) +
            1000 * 4.12 * (self.episode_df["HNLT.localState.fMidTemperature"].iloc[-1] -
                           self.episode_df["HNLT.localState.fMidTemperature"].iloc[0]) + heat_energy_stored_building
        ) / 3600

        # sign change -> positive value if storage is cooler than at begin
        cool_energy_stored = -(25000 * 4.12 * (hvfa_cn_temp_end - hvfa_cn_temp_begin) + 1000 * 4.12 * (
            self.episode_df["CN.localState.fMidTemperature"].iloc[-1] -
            self.episode_df["CN.localState.fMidTemperature"].iloc[0]) +
            cool_energy_stored_building
            ) / 3600
        cool_energy_demand = (
            ufh_cool +
            ict_cool +
            abs(np.mean(self.episode_df['CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatFlowRate']))
        ) * self.experiment_duration

        with np.errstate(divide='ignore'):
            use_energy = (
                heat_energy_demand +
                heat_energy_stored +
                cool_energy_demand +
                cool_energy_stored +
                np.sum(electricity_demand))
            # specific heating and cooling costs
            spec_thermal_energy_costs = round(total_costs / use_energy, 3)

            # coefficient of cooling and heating performance
            cochp = round(use_energy / (gas_demand + np.sum(electrical_diff)) * 100, 1)

            # electrical self generation ratio
            self_generation_elec = round(abs(np.sum(electricity_generation) / np.sum(electricity_demand)) * 100, 1)

        # standard deviation and minutes in which temperature limits were not kept
        std_temp_HNHT = round(self.episode_df['HNHT.localState.fMidTemperature'].std(), 2)
        std_temp_HNLT = round(self.episode_df['HNLT.localState.fMidTemperature'].std(), 2)
        std_temp_CN = round(self.episode_df['CN.localState.fMidTemperature'].std(), 2)
        std_temp_mean = round((std_temp_HNHT+std_temp_HNLT+std_temp_CN)/3, 2)
        time_temp_HNHT_fail = (
            (self.episode_df['HNHT.localState.fMidTemperature'] <
             self.config['environment_specific']['temperature_limits_HNHT'][0]) |
            (self.episode_df['HNHT.localState.fMidTemperature'] >
             self.config['environment_specific']['temperature_limits_HNHT'][1])
            ).sum()*self.sampling_time / 60  # in minutes
        time_temp_HNLT_fail = (
            (self.episode_df['HNLT.localState.fMidTemperature'] <
             self.config['environment_specific']['temperature_limits_HNLT'][0]) |
            (self.episode_df['HNLT.localState.fMidTemperature'] >
             self.config['environment_specific']['temperature_limits_HNLT'][1])
            ).sum() * self.sampling_time / 60  # in minutes
        time_temp_CN_fail = (
            (self.episode_df['CN.localState.fMidTemperature'] <
             self.config['environment_specific']['temperature_limits_CN'][0]) |
            (self.episode_df['CN.localState.fMidTemperature'] >
             self.config['environment_specific']['temperature_limits_CN'][1])
            ).sum() * self.sampling_time / 60  # in minutes

        # combination of all self.networks
        time_temp_total_fail = (
            ((self.episode_df['HNHT.localState.fMidTemperature'] <
              self.config['environment_specific']['temperature_limits_HNHT'][0]) | (
                self.episode_df['HNHT.localState.fMidTemperature'] >
                self.config['environment_specific']['temperature_limits_HNHT'][1])) | (
                (self.episode_df['HNLT.localState.fMidTemperature'] <
                 self.config['environment_specific']['temperature_limits_HNLT'][0]) | (
                    self.episode_df['HNLT.localState.fMidTemperature'] >
                    self.config['environment_specific']['temperature_limits_HNLT'][1])) | (
                (self.episode_df['CN.localState.fMidTemperature'] <
                 self.config['environment_specific']['temperature_limits_CN'][0]) | (
                    self.episode_df['CN.localState.fMidTemperature'] >
                    self.config['environment_specific']['temperature_limits_CN'][1]))
                    ).sum() * self.sampling_time / 60  # in minutes

        # write values into lists
        results_0 = [["Costs of use energy in €/kWh", "COCHP in %", "Self-generation level in %",
                      "Downtime in min", "Mean standard deviation in K"],
                     [spec_thermal_energy_costs, cochp, self_generation_elec, time_temp_total_fail, std_temp_mean]]
        results_1 = [["", "HNHT", "HNLT", "CN"],
                     ["Downtime in min", time_temp_HNHT_fail, time_temp_HNLT_fail, time_temp_CN_fail],
                     ["Standard deviation in K", std_temp_HNHT, std_temp_HNLT, std_temp_CN]]

        # create tables
        tables = [axes[0].table(cellText=[list(map(str, i)) for i in zip(*results_0)],
                                colLabels=None,
                                rowLabels=None,
                                colWidths=[1 / 2, 1 / 2],
                                loc='center',
                                cellLoc='right')]
        tables.append(axes[1].table(cellText=[list(map(str, i)) for i in zip(*results_1)],
                                    colLabels=None,
                                    rowLabels=None,
                                    colWidths=[0.1, 0.45, 0.45],
                                    loc='center',
                                    cellLoc='right'))

        for table in tables:
            table.scale(1, 1.5)

        for table in tables:
            table_props = table.properties()
            table_cells = table_props['children']
            for cell in table_cells:  # Loop through cells and adjust height
                cell.set_text_props(fontsize=9)

        # calculate mean storage temperatures based on upper and lower temperatures - buffer and active storages
        hnht_buffer_upper = np.insert(
            500*4.2*(self.episode_df["HNHT.localState.fUpperTemperature"][1:].to_numpy() -
                     self.episode_df["HNHT.localState.fUpperTemperature"][:-1].to_numpy()
                     )/self.sampling_time, 0, 0)
        hnht_buffer_lower = np.insert(
            500*4.2*(self.episode_df["HNHT.localState.fLowerTemperature"][1:].to_numpy() -
                     self.episode_df["HNHT.localState.fLowerTemperature"][:-1].to_numpy()
                     )/self.sampling_time, 0, 0)
        self.episode_df['hnht_buffer'] = hnht_buffer_upper + hnht_buffer_lower

        hnlt_buffer_upper = np.insert(
            500*4.2*(self.episode_df["HNLT.localState.fUpperTemperature"][1:].to_numpy() -
                     self.episode_df["HNLT.localState.fUpperTemperature"][:-1].to_numpy()
                     )/self.sampling_time, 0, 0)
        hnlt_buffer_lower = np.insert(
            500*4.2*(self.episode_df["HNLT.localState.fLowerTemperature"][1:].to_numpy() -
                     self.episode_df["HNLT.localState.fLowerTemperature"][:-1].to_numpy()
                     )/self.sampling_time, 0, 0)
        self.episode_df['hnlt_buffer'] = hnlt_buffer_upper + hnlt_buffer_lower

        cn_buffer_upper = np.insert(
            500*4.2*(self.episode_df["CN.localState.fUpperTemperature"][1:].to_numpy() -
                     self.episode_df["CN.localState.fUpperTemperature"][:-1].to_numpy()
                     )/self.sampling_time, 0, 0)
        cn_buffer_lower = np.insert(
            500*4.2*(self.episode_df["CN.localState.fLowerTemperature"][1:].to_numpy() -
                     self.episode_df["CN.localState.fLowerTemperature"][:-1].to_numpy()
                     )/self.sampling_time, 0, 0)
        self.episode_df['cn_buffer'] = cn_buffer_upper + cn_buffer_lower

        # define observation names for heat map
        observation_names = [
            'hnht_buffer',
            'HNHT.VSIStorageSystem.localState.fHeatFlowRate',
            'HNHT.CHP1System.WMZ32x.sensorState.fHeatFlowRate',
            'HNHT.CHP2System.WMZ32x.sensorState.fHeatFlowRate',
            'HNHT.CondensingBoilerSystem.WMZ331.sensorState.fHeatFlowRate',
            'HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatFlowRate',
            'HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatFlowRate',
            'HNHT_HNLT.HeatExchanger1System.WMZ215.sensorState.fHeatFlowRate',
            'HNHT_HNLT.HeatPump1System.WMZ342.sensorState.fHeatFlowRate',
            'hnlt_buffer',
            'HNLT.HVFASystem.WMZx05.sensorState.fHeatFlowRate',
            'HNLT.OuterCapillaryTubeMats.WMZ235.sensorState.fHeatFlowRate',
            'HNLT.CompressorSystem.WMZ251.sensorState.fHeatFlowRate',
            'HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatFlowRate',
            'HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatFlowRate',
            'HNLT_CN.HeatPump2System.WMZ246.sensorState.fHeatFlowRate',
            'cn_buffer',
            'CN.HVFASystem.WMZx05.sensorState.fHeatFlowRate',
            'CN.eChillerSystem.WMZ138.sensorState.fHeatFlowRate',
            'CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatFlowRate',
        ]

        # plot heat flow rates as heat map
        # fix sign of producers/consumers - producer positiv
        self.episode_df["HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatFlowRate"] = - \
            self.episode_df["HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatFlowRate"]
        self.episode_df["HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatFlowRate"] = - \
            self.episode_df["HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatFlowRate"]
        self.episode_df["HNLT.OuterCapillaryTubeMats.WMZ235.sensorState.fHeatFlowRate"] = - \
            self.episode_df["HNLT.OuterCapillaryTubeMats.WMZ235.sensorState.fHeatFlowRate"]
        self.episode_df["HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatFlowRate"] = - \
            self.episode_df["HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatFlowRate"]
        self.episode_df["HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatFlowRate"] = - \
            self.episode_df["HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatFlowRate"]
        self.episode_df["CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatFlowRate"] = - \
            self.episode_df["CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatFlowRate"]
        self.episode_df['hnht_buffer'] = -self.episode_df['hnht_buffer']
        self.episode_df['hnlt_buffer'] = -self.episode_df['hnlt_buffer']
        self.episode_df['cn_buffer'] = -self.episode_df['cn_buffer']
        self.episode_df['HNHT.VSIStorageSystem.localState.fHeatFlowRate'] = -self.episode_df[
            'HNHT.VSIStorageSystem.localState.fHeatFlowRate']
        self.episode_df['HNLT.HVFASystem.WMZx05.sensorState.fHeatFlowRate'] = -self.episode_df[
            'HNLT.HVFASystem.WMZx05.sensorState.fHeatFlowRate']
        self.episode_df['CN.HVFASystem.WMZx05.sensorState.fHeatFlowRate'] = -self.episode_df[
            'CN.HVFASystem.WMZx05.sensorState.fHeatFlowRate']

        # add colorbar
        im = axes[2].imshow(self.episode_df[observation_names].astype(int).transpose(), cmap="bwr",
                            vmin=-30, vmax=30, aspect="auto", interpolation="none")
        ax_pos = axes[2].get_position().get_points().flatten()
        ax_colorbar = fig.add_axes([0.93, ax_pos[1] + 0.05, 0.01, ax_pos[3] - ax_pos[1] - 0.1])
        fig_cbar = fig.colorbar(im, ax=axes[2], shrink=0.9, cax=ax_colorbar)
        text_x = ax_colorbar.get_position().x0 + ax_colorbar.get_position().width / 2
        text_y = ax_colorbar.get_position().y1 + 0.01
        fig_text = fig.text(text_x, text_y, 'kW', ha='center', va='bottom', rotation=90)

        # self.configuration of axes
        axes[2].set_yticks(np.arange(20))
        axes[2].set_yticklabels(list(self.system_names.values()))
        axes[2].tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, rotation=45)
        axes[2].tick_params(which="minor", bottom=False, left=False)
        axes[2].set_xticks(self.tickpos)
        axes[2].set_xticklabels(self.ticknames)
        axes[2].set_xlabel("Time (UTC+1)")
        axes[2].xaxis.set_label_position("bottom")
        axes[2].set_yticks(np.arange(16 + 1) - 0.5, minor=True)
        plt.setp(axes[2].get_yticklabels(), rotation=30, ha="right", va="center", rotation_mode="anchor")
        axes[2].grid(False)

        # set layout and fontsize
        for ax in axes[0:2]:
            ax.axis('tight')
            ax.axis('off')

        return fig, axes, fig_cbar, fig_text

    def physical_figure(self):
        """
        figure containing heat flow rates and temperatures

        Returns:
            fig: figure with heat flow rates and temperatures
        """

        # create figure and subplots
        fig = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi)
        axes = []
        axes.append(fig.add_subplot(2, 1, 1))  # 0 - network temperatures for all self.networks
        axes.append(fig.add_subplot(2, 1, 2))  # 1 - machine, room and ambient temperatures

        x = self.episode_df.index
        y = self.episode_df

        # adjust layout
        plt.tight_layout()
        fig.subplots_adjust(left=0.125, bottom=0.15, right=0.9, top=0.95, wspace=0.3, hspace=0.5)

        # (0) - plot storage temperatures in self.networks
        axes[0].plot(
            x,
            y["HNHT.localState.fMidTemperature"],
            color="black",
            linestyle=self.linestyles[0],
            label="BS_HNHT")
        vsi_temp = (y["HNHT.VSIStorageSystem.VSIStorage.localState.fUpperTemperature"] +
                    y["HNHT.VSIStorageSystem.VSIStorage.localState.fLowerTemperature"]) / 2
        axes[0].plot(x, vsi_temp, color="black", linestyle=self.linestyles[1], label="ST_HNHT")
        axes[0].plot(
            x,
            y["HNLT.localState.fMidTemperature"],
            color="black",
            linestyle=self.linestyles[2],
            label="BS_HNLT")
        hnlt_hvfa_temp = (y["HNLT.HVFASystem.HVFAStorage.localState.fUpperTemperature"] +
                          y["HNLT.HVFASystem.HVFAStorage.localState.fLowerTemperature"]) / 2
        axes[0].plot(x, hnlt_hvfa_temp, color="black", linestyle=self.linestyles[3], label="ST_HNLT")
        axes[0].plot(x, y["CN.localState.fMidTemperature"], color="black", linestyle=self.linestyles[4], label="BS_CN")
        cn_hvfa_temp = (y["CN.HVFASystem.HVFAStorage.localState.fUpperTemperature"] +
                        y["CN.HVFASystem.HVFAStorage.localState.fLowerTemperature"]) / 2
        axes[0].plot(x, cn_hvfa_temp, color="black", linestyle=self.linestyles[5], label="ST_CN")

        axes[0].set_title("Storage temperatures", loc="center", fontdict={'fontweight': 'bold'})
        axes[0].set_ylabel("Temperature in °C")
        axes[0].margins(x=0.0, y=0.1)
        axes[0].set_axisbelow(True)
        axes[0].tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
        axes[0].tick_params(axis='x', top=False, bottom=True, labeltop=False, labelbottom=True, rotation=45)
        axes[0].set_xticks(self.tickpos)
        axes[0].set_xticklabels(self.ticknames)
        axes[0].set_xlabel("Time (UTC+1)")
        axes[0].xaxis.set_label_position("bottom")
        axes[0].set_ylim([-5, 80])
        axes[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.15),
                       frameon=False, edgecolor='black', fancybox=False)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)

        # (1) - plot system specific temperatures
        axes[1].plot(x, y["HNHT.StaticHeatingSystem.ConsumerTemperature.Celsius"],
                     color="black", linestyle=self.linestyles[0], label="Heizung")
        axes[1].plot(x, y["HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fReturnTemperature"],
                     color="black", linestyle=self.linestyles[1], label="DLRA")
        axes[1].plot(x, y["HNLT.CompressorSystem.WMZ251.sensorState.fReturnTemperature"],
                     color="black", linestyle=self.linestyles[3], label="DLK")
        axes[1].plot(x, y["HNLT_CN.UnderfloorHeatingSystem.ConsumerTemperature.Celsius"],
                     color="black", linestyle=self.linestyles[4], label="FBH")
        axes[1].plot(x, y["HNLT_CN.InnerCapillaryTubeMats.setPointState.fOperatingPoint"],
                     color="black", linestyle=self.linestyles[5], label="IFA")
        axes[1].plot(x, y["CN.CentralMachineCoolingSystem.WMZ100.sensorState.fReturnTemperature"],
                     color="black", linestyle=self.linestyles[6], label="HD")

        axes[1].set_title("System temperatures", loc="center", fontdict={'fontweight': 'bold'})
        axes[1].set_ylabel("Temperature in °C")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].set_axisbelow(True)
        axes[1].tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
        axes[1].tick_params(axis='x', top=False, bottom=True, labeltop=False, labelbottom=True, rotation=45)
        axes[1].set_xticks(self.tickpos)
        axes[1].set_xticklabels(self.ticknames)
        axes[1].set_xlabel("Time (UTC+1)")
        axes[1].xaxis.set_label_position("bottom")
        axes[1].legend(loc="upper center", ncol=4, bbox_to_anchor=(
            0.5, -0.2), frameon=False, edgecolor='black', fancybox=False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)

        return fig, axes, None, None

    def single_systems_figure(self):
        """
        figure containing metrics for single systems

        Returns:
            fig: figure with single system characteristics
        """

        # create figure and subplots
        fig = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi)
        axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
        axes[0].set_title("HNHT | Single systems", loc="center", fontdict={'fontweight': 'bold'})
        axes[1].set_title("HNLT | Single systems", loc="center", fontdict={'fontweight': 'bold'})
        axes[2].set_title("CN | Single systems", loc="center", fontdict={'fontweight': 'bold'})

        # adjust layout
        plt.tight_layout()
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1)

        # get short and and longtime product by duration as string
        shorttime_product = str(self.config["environment_specific"]['products'][0][0])

        # no longtime product defined
        if len(self.config["environment_specific"]['products'][0]) == 1:
            longtime_product = shorttime_product
        else:
            longtime_product = str(self.config["environment_specific"]['products'][0][1])

        # iterate over self.networks - one network per subplot
        for idx, (network, systems) in enumerate(self.networks.items()):
            result_data = [["",
                            "ST | kWh",
                            "LT | kWh",
                            "BE | kWh",
                            "ST | €/kWh",
                            "LT | €/kWh",
                            "BE | €/kWh"]]
            # calculate mean prices and market allocation
            for idx_system, system in enumerate(systems):
                data = pd.read_excel(self.result_path, sheet_name=system)
                shorttime = round(sum(abs(data[shorttime_product + '_' + network])), 1)
                longtime = round(sum(abs(data[longtime_product + '_' + network])), 1)
                balancing_energy_demand = round(sum(abs(data['balancing_energy_' + network])), 1)

                idx_shorttime_unequal_zero = np.nonzero(round(data[shorttime_product + '_' + network], 2))[0]
                idx_shorttime_unequal_zero[idx_shorttime_unequal_zero == -0.0] = 0
                if len(idx_shorttime_unequal_zero) > 0:
                    shorttime_price = round(
                        np.mean(data["price_" + shorttime_product + '_' + network]
                                [idx_shorttime_unequal_zero]), 2)
                else:
                    shorttime_price = 0

                idx_longtime_unequal_zero = np.nonzero(round(data[longtime_product + '_' + network], 2))[0]
                idx_longtime_unequal_zero[idx_longtime_unequal_zero == -0.0] = 0
                if len(idx_longtime_unequal_zero) > 0:

                    longtime_price = round(np.mean(data["price_" + longtime_product + '_' + network]
                                                   [idx_longtime_unequal_zero]), 2)
                else:
                    longtime_price = 0

                # balancing energy costs in €/kWh within data set
                idx_balancing_energy_unequal_zero = np.nonzero(round(data['balancing_energy_' + network], 2))[0]
                idx_balancing_energy_unequal_zero[idx_balancing_energy_unequal_zero == -0.0] = 0
                if len(idx_balancing_energy_unequal_zero) > 0:
                    balancing_energy_price = round(
                        np.mean(data["cost_balancing_energy_" + network]
                                [idx_balancing_energy_unequal_zero]), 2)
                else:
                    balancing_energy_price = 0

                res = [
                    self.system_names[system],
                    shorttime,
                    longtime,
                    balancing_energy_demand,
                    shorttime_price,
                    longtime_price,
                    balancing_energy_price]
                result_data.append(res)

            # create table
            table = axes[idx].table(cellText=result_data,
                                    colLabels=None,
                                    rowLabels=None,
                                    colWidths=[1 / 7 for i in range(7)],
                                    loc='center',
                                    cellLoc='right')

            # scale table and adjust fontsize
            axes[idx].axis('tight')
            axes[idx].axis('off')
            table.scale(1, 1.5)
            table_props = table.properties()
            table_cells = table_props['children']
            for cell in table_cells:
                cell.set_text_props(fontsize=9)

        return fig, axes, None, None

    def price_figure(self):
        """
        figure containing clearing prices for different self.networks and products

        Returns:
            fig: figure with prices
        """

        # load prices and group them by product
        prices = {network: pd.read_excel(self.result_path, sheet_name=network) for network in self.networks}
        for system, price in prices.items():
            grouped = price.groupby('product')
            prices[system] = {group: data for group, data in grouped}

        # product list and self.linestyles
        product_list = [eval(key) for key in list(prices.values())[0].keys()]
        product_list = sorted(product_list, key=lambda x: x['product_type'])

        # create figure and subplots
        fig = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi)
        axes = [fig.add_subplot(len(product_list), 1, i + 1) for i in range(len(product_list))]

        # adjust layout
        plt.tight_layout()
        fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.95, wspace=0.3, hspace=0.7)

        # iterate over products - one product per subplot
        for idx_product, product in enumerate(product_list):
            product_label = str(product['product_type'] / 3600) + \
                ' h - Execution in: ' + str(product['lead_time'] / 3600) + ' h'

            # iterate over self.networks
            for system_index, (network, systems) in enumerate(self.networks.items()):

                # repeat values depending on sampling time and clearing points in time
                data = prices[network][str(product)]
                repeat_index = data.index.repeat(len(self.episode_df) / len(data))
                repeated_data = data.loc[repeat_index].reset_index(drop=True)

                # find indices of values around nan (no clearing)
                isnan = repeated_data['price'].isna().to_numpy()
                before_nan = np.where(isnan[:-1] & ~isnan[1:])[0] + 1
                after_nan = np.where(~isnan[:-1] & isnan[1:])[0]
                idx = np.unique(np.concatenate([before_nan, after_nan]))

                # plot prices
                axes[idx_product].plot(repeated_data.index, repeated_data['price'], color="black",
                                       linestyle=self.linestyles[system_index], label=network)

                # mark points with nan (no clearing)
                axes[idx_product].plot(repeated_data.index[idx], repeated_data['price'].iloc[idx],
                                       marker=self.markers[system_index], markersize=0.5,
                                       linestyle='None', color="black")

            # self.configuration of axes
            axes[idx_product].set_title("Market clearing price | " +
                                        product_label, loc="center", fontdict={'fontweight': 'bold'})
            axes[idx_product].set_ylabel("Price in €/kWh")
            axes[idx_product].margins(x=0.0, y=0.1)
            axes[idx_product].set_axisbelow(True)
            axes[idx_product].tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
            axes[idx_product].tick_params(
                axis='x',
                top=False,
                bottom=True,
                labeltop=False,
                labelbottom=True,
                rotation=45)
            axes[idx_product].set_xticks(self.tickpos)
            axes[idx_product].spines['right'].set_visible(False)
            axes[idx_product].spines['top'].set_visible(False)
            axes[idx_product].set_xticklabels(self.ticknames)
            axes[idx_product].set_xlabel("Time (UTC+1)")
            axes[idx_product].xaxis.set_label_position("bottom")

        # create legend for whole figure
        handles, labels = [], []
        ax = axes[0]
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=3,
            frameon=False,
            edgecolor='black',
            fancybox=False)
        return fig, axes, None, None


if __name__ == '__main__':
    # set paths and read config
    root_path = pathlib.Path(__file__).parents[1]
    # run_days = ["2023_07_12_1_day_"]
    # scenarios = ["benchmark","s0"]

    # run_days = ["2023_07_12_1_day_","2024_01_10_1_day_","2024_04_09_1_day_"]
    # scenarios = ["benchmark","s0","s1","s2","s3_1","s3_2","s3_3","s3_4","s3_4","s4","s5"]
    run_days = ["2024_09_14_"]
    scenarios = ["s7_live"]
    for run_day in run_days:
        for scenario in scenarios:

            result_path = os.path.join(root_path, "results", "eta_heating_systems_mas", run_day + scenario + ".xlsx")
            config_path = os.path.join(root_path, "config", run_day+scenario + ".json")
            fig_path = os.path.join(root_path, "results", "eta_heating_systems_mas",
                                    "results_" + run_day+scenario + ".pdf")

            plotter = Plotter(result_path=result_path, config_path=config_path, fig_size=(16.5 / 2.54, 23.5 / 2.54))

            # figure functions to be called
            config = read_config(config_path)
            if config["environment_specific"]["is_benchmark_scenario"]:
                figure_functions = [plotter.total_system_figure,
                                    plotter.physical_figure]
            else:
                figure_functions = [plotter.total_system_figure,
                                    plotter.physical_figure,
                                    plotter.single_systems_figure,
                                    plotter.price_figure]

            # call all functions and save figure to pdf and svg
            with PdfPages(fig_path) as pdf:
                for fig_func in figure_functions:
                    fig, _, _, _ = fig_func()
                    pdf.savefig(fig)
                    plt.close(fig)
