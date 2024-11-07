"""
plotting supplementary material
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "plotting supplementary material"

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from experiments.eta_heating_systems.environment.plotter import Plotter

project_id = 'diss_fbo_'
public_git_repo_url = "https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4220"

root_path = pathlib.Path(__file__).parents[0]

# global settings
font_path = 'C:\\Windows\\Fonts\\lmroman10-regular.otf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["font.size"] = 9
plt.rcParams['lines.linewidth'] = 0.9
linestyles = ["solid", "dashed", "dashdot", "dotted",
              (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
grey_colors = ['#f7f7f7', '#e5e5e5', '#cccccc', '#bdbdbd', '#969696', '#737373']
patterns = ['....', 'oooo', '////',  '----',  'xxxx', '||||']
markers = ['o', 's', '^']


def fig_1():
    fig_size = (16.5 / 2.54, 8.25 / 2.54)
    fig_dpi = 600

    # create figure and subplots
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    axes = [fig.add_subplot(1, 4, 1), fig.add_subplot(1, 4, 2), fig.add_subplot(1, 4, (3, 4))]

    # adjust layout
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, bottom=0.3, right=0.95, top=0.95, wspace=0.7)

    # figure 1a - industrial energy demands
    # Data for the plot
    labels = ['Wärme', 'Kälte', 'Mechanik', 'Sonstige']

    # https://ag-energiebilanzen.de/daten-und-fakten/anwendungsbilanzen/
    industrial_energy_demands = [73, 2, 22, 3]

    # calculate the percentages
    total = sum(industrial_energy_demands)
    percentages = [f'{(i / total) * 100:.0f} %' for i in industrial_energy_demands]

    # combine labels and percentages
    legend_labels = [f'{label} ({percentage})' for label, percentage in zip(labels, percentages)]

    wedges, texts = axes[0].pie(industrial_energy_demands,
                                startangle=140, colors=grey_colors,
                                textprops=dict(color="black"))

    # Add legend below the chart with percentages in brackets
    axes[0].legend(wedges, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.185), handlelength=2,
                   handleheight=1, ncol=1, frameon=False, edgecolor='black', fancybox=False)

    # figure 1b - industrial energy carriers
    labels = ['Gas', 'Kohlen', 'Heizöl', 'Fernwärme', 'Strom', 'Sonstige']

    # https://ag-energiebilanzen.de/daten-und-fakten/anwendungsbilanzen/
    industrial_energy_carriers = [46, 21, 5, 10, 7, 11]

    # calculate the percentages
    total = sum(industrial_energy_carriers)
    percentages = [f'{(i / total) * 100:.0f} %' for i in industrial_energy_carriers]

    # combine labels and percentages
    legend_labels = [f'{label} ({percentage})' for label, percentage in zip(labels, percentages)]

    wedges, texts = axes[1].pie(industrial_energy_carriers,
                                startangle=140, colors=grey_colors,
                                textprops=dict(color="black"))

    # Add legend below the chart with percentages in brackets
    axes[1].legend(wedges, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), handlelength=2,
                   handleheight=1, ncol=1, frameon=False, edgecolor='black', fancybox=False)

    # figure 1c
    time = np.arange(2005, 2024)
    x_ticks = [2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]
    y_ticks = [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]

    # indices to the base of 2015
    # https://www.destatis.de/DE/Themen/Wirtschaft/Preise/Publikationen/Energiepreise/statistischer-bericht-energiepreisentwicklung-5619001241035.html
    coal_index_2015 = np.array([79, 81.7, 84.5, 82.7, 87.1, 89.8, 98.7, 101.6, 102.5, 101.5, 100, 99.5, 97, 98.1, 103.4,
                                104.4, 106.9, 112.6,])
    oil_index_2015 = np.array([91.7, 102.8, 101.3, 133.3, 88.1, 112.9, 143.8, 157.8, 147, 133.9, 100, 83.2, 97.5, 119.1,
                               116, 78.2, 123.2, 288.8])
    gas_index_2015 = np.array([65.8, 84.8, 84.5, 99.5, 93.2, 86.8, 100, 113.3, 113.1, 108.3, 100, 84.2, 89.9, 98.2, 91,
                               73.5, 149.2, 392])
    electricity_index_2015 = np.array([65.1, 67.6, 69.7, 72.8, 76.5, 79.7, 85.9, 89, 99.5, 100.4, 100, 100.7, 102.5,
                                       102, 104.8, 110.5, 112, 130])

    # factor to the base of 2020
    factor_coal = 100/104.4
    factor_oil = 100/78.2
    factor_gas = 100/73.5
    factor_electricity = 100/110.5

    # indices to the base of 2020
    coal_index_2020 = coal_index_2015*factor_coal
    oil_index_2020 = oil_index_2015*factor_oil
    gas_index_2020 = gas_index_2015*factor_gas
    electricity_index_2020 = electricity_index_2015*factor_electricity

    # appending values for 2023,(2024)
    # https://www.destatis.de/DE/Themen/Wirtschaft/Preise/Publikationen/Energiepreise/statistischer-bericht-energiepreisentwicklung-5619001241035.html
    coal_index_2020 = np.log(np.append(coal_index_2020, 143.5))/np.log(100)
    oil_index_2020 = np.log(np.append(oil_index_2020, 146.4))/np.log(100)
    gas_index_2020 = np.log(np.append(gas_index_2020, 234.4))/np.log(100)
    electricity_index_2020 = np.log(np.append(electricity_index_2020, 134))/np.log(100)

    axes[2].plot(time, coal_index_2020, color="black", linestyle=linestyles[0], label="Kohlen")
    axes[2].plot(time, oil_index_2020, color="black", linestyle=linestyles[1], label="Heizöl")
    axes[2].plot(time, gas_index_2020, color="black", linestyle=linestyles[2], label="Erdgas")
    axes[2].plot(time, electricity_index_2020, color="black", linestyle=linestyles[3], label="Strom")
    axes[2].set_ylabel("Log. Erzeugerpreisindex")
    axes[2].margins(x=0.0, y=0.1)
    axes[2].set_axisbelow(True)
    axes[2].tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
    axes[2].tick_params(axis='x', top=False, bottom=True, labeltop=False, labelbottom=True, rotation=45)
    axes[2].set_yticks(y_ticks)
    axes[2].set_yticklabels(y_ticks)
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(x_ticks)
    axes[2].set_xlabel("Zeit")
    axes[2].xaxis.set_label_position("bottom")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), handlelength=1.5, columnspacing=0.5, ncol=4,
                   frameon=False, edgecolor='black', fancybox=False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)

    return fig


def fig_8_1():
    fig_size = (15 / 2.54, 8 / 2.54)
    fig_dpi = 600

    # create figure and subplots
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    axes = [fig.add_subplot(1, 1, 1)]

    # adjust layout
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.3, right=0.95, top=0.95)

    y_ticks = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    ticknames = []
    tickpos = []
    for step in range(24):
        tickdate = datetime(year=2024, month=1, day=1, hour=0, second=0) + timedelta(seconds=step * 3600)
        if tickdate.minute == 0 and tickdate.second == 0:
            tickpos.append(step)
            ticknames.append(tickdate.strftime("%H:%M"))
        elif tickdate.hour == 0 and tickdate.minute == 0 and tickdate.second == 0:
            tickpos.append(step)
            ticknames.append(tickdate.strftime("%d.%m"))

    # ambient temperatures for winter, sommer, spring
    # link
    time = np.arange(0, 24)
    ambient_temperature_winter = np.array([-6.05, -6.25, -6.7, -7, -7.1, -7.25, -7.55, -7.7, -7.55, -6.6, -5.3, -4.3,
                                           -3.2, -1.95, -1.1, -0.9, -1.5, -3.5, -4.8, -5.95, -6.6, -7.3, -7.9, -8.15])
    ambient_temperature_summer = np.array([20.7, 19.7, 19.0, 19.2, 19.6, 19.3, 19.7, 20.3, 20.7, 22.2, 23.9, 25.3, 27.1,
                                           27.5, 27.1, 27.2, 27.1, 26.7, 26.1, 25.2, 24.2, 22.5, 20.5, 19.1])
    ambient_temperature_spring = np.array([16.0, 18.8, 19.4, 16.2, 15.4, 17.9, 18.9, 17.1, 16.6, 16.1, 15.7, 12.7, 12.6,
                                           11.5, 10.5, 9.2,  10.9,  10.9,  10.8,  10.7,  10.4,  9.7,  9.4,  8.9])

    axes[0].plot(time, ambient_temperature_summer, color="black", linestyle=linestyles[1], label="Sommer (12.07.2023)")
    axes[0].plot(time, ambient_temperature_winter, color="black", linestyle=linestyles[0], label="Winter (10.01.2024)")
    axes[0].plot(time, ambient_temperature_spring, color="black", linestyle=linestyles[2],
                 label="Übergangszeit (09.04.2024)")
    axes[0].set_ylabel("Umgebungstemperatur in °C")
    axes[0].margins(x=0.0, y=0.1)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
    axes[0].tick_params(axis='x', top=False, bottom=True, labeltop=False, labelbottom=True, rotation=45)
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_ticks)
    axes[0].set_xticks(tickpos)
    axes[0].set_xticklabels(ticknames)
    axes[0].set_xlabel("Zeit in h")
    axes[0].xaxis.set_label_position("bottom")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), handlelength=1.5, columnspacing=0.5, ncol=4,
                   frameon=False, edgecolor='black', fancybox=False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)

    return fig


def _prepare_temperatures(run_name):
    '''
    prepare storage temperature plot for documentation
    '''
    # set paths and read config
    root_path = pathlib.Path(__file__).parents[1]
    run_name = run_name
    result_path = os.path.join(root_path, "experiments", "eta_heating_systems", "results",
                               "eta_heating_systems_mas", run_name + ".xlsx")
    config_path = os.path.join(root_path, "experiments", "eta_heating_systems", "config", run_name + ".json")
    plotter = Plotter(result_path=result_path, config_path=config_path, fig_size=(16.4 / 2.54, 8 / 2.54))
    fig, axes, _, _ = plotter.physical_figure()

    axes[0].set_title("")
    axes[1].remove()
    leg = axes[0].get_legend()
    leg.set_bbox_to_anchor((0.5, 0.18))
    axes[0].set_position([0.125, 0.02, 0.85, 0.95])  # [left, bottom, width, height]
    axes[0].set_xticklabels([])
    axes[0].set_xlabel('')
    axes[0].set_ylim([-5, 80])
    axes[0].set_ylabel('Temperatur in °C')

    return fig


def _prepare_utilization(run_name):
    '''
    prepare storage temperature plot for documentation
    '''
    # set paths and read config
    root_path = pathlib.Path(__file__).parents[1]
    run_name = run_name
    result_path = os.path.join(root_path, "experiments", "eta_heating_systems", "results",
                               "eta_heating_systems_mas", run_name + ".xlsx")
    config_path = os.path.join(root_path, "experiments", "eta_heating_systems", "config", run_name + ".json")
    plotter = Plotter(result_path=result_path, config_path=config_path, fig_size=(16.4 / 2.54, 11 / 2.54))
    fig, axes, fig_cbar, fig_text = plotter.total_system_figure()

    axes[2].set_title("")
    axes[0].remove()
    axes[1].remove()
    fig_cbar.remove()
    fig_text.remove()

    vmin, vmax = -30, 30
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('bwr')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.125, 0.1, 0.85, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks(np.linspace(vmin, vmax, 7))
    cbar.set_label('kW')

    axes[2].set_position([0.125, 0.28, 0.85, 0.73])  # [left, bottom, width, height]
    axes[2].set_xlabel('Zeit (UTC+1)')
    axes[2].set_ylabel([])
    return fig


def _prepare_clearing_price(run_name):
    '''
    prepare storage temperature plot for documentation
    '''
    # set paths and read config
    root_path = pathlib.Path(__file__).parents[1]
    run_name = run_name
    result_path = os.path.join(root_path, "experiments", "eta_heating_systems", "results",
                               "eta_heating_systems_mas", run_name + ".xlsx")
    config_path = os.path.join(root_path, "experiments", "eta_heating_systems", "config", run_name + ".json")
    plotter = Plotter(result_path=result_path, config_path=config_path, fig_size=(16.4 / 2.54, 20 / 2.54))
    fig, axes, fig_cbar, fig_text = plotter.price_figure()

    for ax in range(3):
        axes[ax].set_xticklabels([])
        axes[ax].set_xlabel('')

    for ax in axes:
        title = ax.get_title()
        title = title.replace("Market clearing price", "Markträumungspreis").replace("Execution", "Ausführung")
        ax.set_title(title)
        ax.set_ylabel("Preis in €/kWh")
    fig.subplots_adjust(left=0.125, bottom=0.12, right=0.975, top=0.95, wspace=0.3, hspace=0.3)
    axes[3].set_xlabel('Zeit (UTC+1)')
    return fig


def _prepare_clearing_price_long_short(run_name):
    # set paths and read config
    root_path = pathlib.Path(__file__).parents[1]
    run_name = run_name
    result_path = os.path.join(root_path, "experiments", "eta_heating_systems", "results",
                               "eta_heating_systems_mas", run_name + ".xlsx")
    config_path = os.path.join(root_path, "experiments", "eta_heating_systems", "config", run_name + ".json")
    plotter = Plotter(result_path=result_path, config_path=config_path, fig_size=(16.4 / 2.54, 10 / 2.54))
    fig, axes, fig_cbar, fig_text = plotter.price_figure()

    for ax in range(3):
        axes[ax].set_xticklabels([])
        axes[ax].set_xlabel('')

    for ax in axes:
        title = ax.get_title()
        title = title.replace("Market clearing price", "Markträumungspreis").replace("Execution", "Ausführung")
        ax.set_title(title)
        ax.set_ylabel("Preis in €/kWh")
    axes[3].set_xlabel('Zeit (UTC+1)')

    axes[1].remove()
    axes[2].remove()
    fig.subplots_adjust(left=0.125, bottom=0.25, right=0.95, top=0.95, hspace=-0.55)
    return fig


def _prepare_clearing_price_one_product(run_name):
    # set paths and read config
    root_path = pathlib.Path(__file__).parents[1]
    run_name = run_name
    result_path = os.path.join(root_path, "experiments", "eta_heating_systems", "results",
                               "eta_heating_systems_mas", run_name + ".xlsx")
    config_path = os.path.join(root_path, "experiments", "eta_heating_systems", "config", run_name + ".json")
    plotter = Plotter(result_path=result_path, config_path=config_path, fig_size=(16.4 / 2.54, 6 / 2.54))
    fig, axes, fig_cbar, fig_text = plotter.price_figure()

    title = axes[0].get_title()
    title = title.replace("Market clearing price", "Markträumungspreis").replace("Execution", "Ausführung")
    axes[0].set_title(title)
    axes[0].set_ylabel("Preis in €/kWh")
    axes[0].set_xlabel('Zeit (UTC+1)')

    fig.subplots_adjust(left=0.125, bottom=0.37, right=0.95, top=0.9, hspace=-0.55)
    return fig


# benchmark
def fig_240409_b_temp():
    run_name = "2024_04_09_1_day_benchmark"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_b_util():
    run_name = "2024_04_09_1_day_benchmark"
    fig = _prepare_utilization(run_name)
    return fig


# s0
def fig_240409_s0_temp():
    run_name = "2024_04_09_1_day_s0"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s0_util():
    run_name = "2024_04_09_1_day_s0"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s0_price():
    run_name = "2024_04_09_1_day_s0"
    fig = _prepare_clearing_price(run_name)
    return fig


# s1
def fig_240409_s1_temp():
    run_name = "2024_04_09_1_day_s1"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s1_util():
    run_name = "2024_04_09_1_day_s1"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s1_price():
    run_name = "2024_04_09_1_day_s1"
    fig = _prepare_clearing_price_long_short(run_name)
    return fig


def fig_240409_s1_price_a():
    run_name = "2024_04_09_1_day_s1"
    fig = _prepare_clearing_price(run_name)
    return fig


# s2
def fig_240409_s2_temp():
    run_name = "2024_04_09_1_day_s2"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s2_util():
    run_name = "2024_04_09_1_day_s2"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s2_price_a():
    run_name = "2024_04_09_1_day_s2"
    fig = _prepare_clearing_price(run_name)
    return fig


# s3
def fig_240409_s3_1_temp():
    run_name = "2024_04_09_1_day_s3_1"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s3_1_util():
    run_name = "2024_04_09_1_day_s3_1"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s3_1_price():
    run_name = "2024_04_09_1_day_s3_1"
    fig = _prepare_clearing_price(run_name)
    return fig


def fig_240409_s3_2_temp():
    run_name = "2024_04_09_1_day_s3_2"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s3_2_util():
    run_name = "2024_04_09_1_day_s3_2"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s3_2_price():
    run_name = "2024_04_09_1_day_s3_2"
    fig = _prepare_clearing_price_one_product(run_name)
    return fig


def fig_240409_s3_3_temp():
    run_name = "2024_04_09_1_day_s3_3"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s3_3_util():
    run_name = "2024_04_09_1_day_s3_3"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s3_3_price():
    run_name = "2024_04_09_1_day_s3_3"
    fig = _prepare_clearing_price_one_product(run_name)
    return fig


def fig_240409_s3_4_temp():
    run_name = "2024_04_09_1_day_s3_4"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s3_4_util():
    run_name = "2024_04_09_1_day_s3_4"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s3_4_price():
    run_name = "2024_04_09_1_day_s3_4"
    fig = _prepare_clearing_price_one_product(run_name)
    return fig


# s4
def fig_240409_s4_temp():
    run_name = "2024_04_09_1_day_s4"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s4_util():
    run_name = "2024_04_09_1_day_s4"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s4_price():
    run_name = "2024_04_09_1_day_s4"
    fig = _prepare_clearing_price(run_name)
    return fig


# s5
def fig_240409_s5_temp():
    run_name = "2024_04_09_1_day_s5"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240409_s5_util():
    run_name = "2024_04_09_1_day_s5"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240409_s5_price():
    run_name = "2024_04_09_1_day_s5"
    fig = _prepare_clearing_price(run_name)
    return fig


# benchmark live
def fig_240905_b_live_temp():
    run_name = "2024_09_05_s7_benchmark_live"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240905_b_live_util():
    run_name = "2024_09_05_s7_benchmark_live"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240915_b_live_temp():
    run_name = "2024_09_15_s7_benchmark_live"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240915_b_live_util():
    run_name = "2024_09_15_s7_benchmark_live"
    fig = _prepare_utilization(run_name)
    return fig


# s7 live
def fig_240830_s7_temp():
    run_name = "2024_08_30_s7_live"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240830_s7_util():
    run_name = "2024_08_30_s7_live"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240830_s7_price():
    run_name = "2024_08_30_s7_live"
    fig = _prepare_clearing_price(run_name)
    return fig


def fig_240914_s7_temp():
    run_name = "2024_09_14_s7_live"
    fig = _prepare_temperatures(run_name)
    return fig


def fig_240914_s7_util():
    run_name = "2024_09_14_s7_live"
    fig = _prepare_utilization(run_name)
    return fig


def fig_240914_s7_price():
    run_name = "2024_09_14_s7_live"
    fig = _prepare_clearing_price(run_name)
    return fig


if __name__ == '__main__':
    # PDF for plot development
    figure_functions = [fig_240914_s7_temp, fig_240914_s7_util, fig_240914_s7_price]
    root_path = pathlib.Path(__file__).parents[0]
    fig_path = os.path.join(root_path, "plotting.pdf")

    # call all functions and save figure to pdf and svg
    with PdfPages(fig_path) as pdf:
        for fig_func in figure_functions:
            fig = fig_func()
            pdf.savefig(fig)
            plt.close(fig)

    for fig_func in figure_functions:
        fig = fig_func()
        fig.savefig(os.path.join(root_path, fig_func.__name__+'.svg'), format='svg')
