"""
virtual environment
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "virtual environment (fmu)"

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pathlib
from datetime import datetime, timedelta
from typing import Any, Callable
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.envs import BaseEnvSim, StateConfig, StateVar
from eta_utility.type_hints import StepResult, TimeStep
from experiments.eta_heating_systems.environment.plotter import Plotter
from multi_agent_system.base.util import read_config


class EtaHeatingSystems(BaseEnvSim):
    """
    supply system for ETA Research Factory (fmu) based on BaseEnvSim.

    :param env_id: Identification for the environment, useful when creating multiple environments
    :param config_run: Configuration of the optimization run
    :param seed: Random seed to use for generating random numbers in this environment
        (default: None / create random seed)
    :param verbose: Verbosity to use for logging (default: 2)
    :param callback: Callback which should be called after each episode
    :param sampling_time: Length of a timestep in seconds
    :param episode_duration: Duration of one episode in seconds
    """

    # set info
    version = "v0.1"
    description = "(c) Fabian Borst"
    fmu_name = "eta_heating_systems"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,  # config_run contains run_name, general_settings, path_settings, env_info
        seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        sampling_time: float,  # from settings section of config file
        episode_duration: TimeStep | str,  # from settings section of config file
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        **kwargs: Any,
    ):
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            # seed=seed,
            # verbose=verbose,
            callback=callback,
            sampling_time=sampling_time,
            episode_duration=episode_duration,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            **kwargs,
        )

        # intialize dataframe for episode
        self.episode_df = pd.DataFrame()
        self.scenario_time = [self.scenario_time_begin]
        root_path = pathlib.Path(__file__).parents[1]
        from experiments.eta_heating_systems.config.global_ import series_name, run_name
        self.result_path = os.path.join(root_path, "results", series_name, run_name + ".xlsx")

        state_var_tuple = (
            # strategy
            StateVar(
                name="Strategy.localSetParameters.bProductionModeActivated",
                ext_id="Strategy.localSetParameters.bProductionModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="Strategy.localSetParameters.bHeatingModeActivated",
                ext_id="Strategy.localSetParameters.bHeatingModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="T_ambient",
                ext_id="T_ambient",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100,
                high_value=100,
            ),
            # ambient
            StateVar(
                name="Ambient.localState.fOutsideTemperature",
                ext_id="Ambient.localState.fOutsideTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # hnht state
            StateVar(
                name="HNHT.localState.fUpperTemperature",
                ext_id="HNHT.localState.fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT.localState.fMidTemperature",
                ext_id="HNHT.localState.fMidTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT.localState.fLowerTemperature",
                ext_id="HNHT.localState.fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # hnlt state
            StateVar(
                name="HNLT.localState.fUpperTemperature",
                ext_id="HNLT.localState.fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT.localState.fMidTemperature",
                ext_id="HNLT.localState.fMidTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT.localState.fLowerTemperature",
                ext_id="HNLT.localState.fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # cn state
            StateVar(
                name="CN.localState.fUpperTemperature",
                ext_id="CN.localState.fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN.localState.fMidTemperature",
                ext_id="CN.localState.fMidTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN.localState.fLowerTemperature",
                ext_id="CN.localState.fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # chp1 system
            StateVar(
                name="HNHT.CHP1System.control.bAlgorithmModeActivated",
                ext_id="HNHT.CHP1System.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CHP1System.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CHP1System.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CHP1System.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CHP1System.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CHP1System.RV32x.control.bAlgorithmModeActivated",
                ext_id="HNHT.CHP1System.RV32x.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT.CHP1System.RV32x.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CHP1System.RV32x.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CHP1System.RV32x.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CHP1System.RV32x.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CHP1System.RV32x.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT.CHP1System.RV32x.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT.CHP1System.WMZ32x.sensorState.fHeatEnergy",
                ext_id="HNHT.CHP1System.WMZ32x.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP1System.WMZ32x.sensorState.fHeatFlowRate",
                ext_id="HNHT.CHP1System.WMZ32x.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP1System.WMZ32x.sensorState.fReturnTemperature",
                ext_id="HNHT.CHP1System.WMZ32x.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP1System.P_el",  # use chp counter
                ext_id="HNHT.CHP1System.P_el",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP1System.P_gas",  # use chp counter
                ext_id="HNHT.CHP1System.P_gas",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # chp2 system
            StateVar(
                name="HNHT.CHP2System.control.bAlgorithmModeActivated",
                ext_id="HNHT.CHP2System.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CHP2System.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CHP2System.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CHP2System.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CHP2System.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CHP2System.RV32x.control.bAlgorithmModeActivated",
                ext_id="HNHT.CHP2System.RV32x.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT.CHP2System.RV32x.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CHP2System.RV32x.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CHP2System.RV32x.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CHP2System.RV32x.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CHP2System.RV32x.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT.CHP2System.RV32x.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT.CHP2System.WMZ32x.sensorState.fHeatEnergy",
                ext_id="HNHT.CHP2System.WMZ32x.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP2System.WMZ32x.sensorState.fHeatFlowRate",
                ext_id="HNHT.CHP2System.WMZ32x.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP2System.WMZ32x.sensorState.fReturnTemperature",
                ext_id="HNHT.CHP2System.WMZ32x.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP2System.P_el",  # use chp counter
                ext_id="HNHT.CHP2System.P_el",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CHP2System.P_gas",  # use chp counter
                ext_id="HNHT.CHP2System.P_gas",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # condensing boiler system
            StateVar(
                name="HNHT.CondensingBoilerSystem.control.bAlgorithmModeActivated",
                ext_id="HNHT.CondensingBoilerSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CondensingBoilerSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CondensingBoilerSystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.RV331.control.bAlgorithmModeActivated",
                ext_id="HNHT.CondensingBoilerSystem.RV331.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT.CondensingBoilerSystem.RV331.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CondensingBoilerSystem.RV331.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.RV331.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CondensingBoilerSystem.RV331.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.RV331.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT.CondensingBoilerSystem.RV331.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.WMZ331.sensorState.fHeatEnergy",
                ext_id="HNHT.CondensingBoilerSystem.WMZ331.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.WMZ331.sensorState.fHeatFlowRate",
                ext_id="HNHT.CondensingBoilerSystem.WMZ331.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.WMZ331.sensorState.fReturnTemperature",
                ext_id="HNHT.CondensingBoilerSystem.WMZ331.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CondensingBoilerSystem.P_gas",  # must be set manually within data logger file
                ext_id="HNHT.CondensingBoilerSystem.P_gas",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # vsi storage system
            StateVar(
                name="HNHT.VSIStorageSystem.control.bAlgorithmModeActivated",
                ext_id="HNHT.VSIStorageSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.VSIStorageSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.localSetParameters.bLoadingAlgorithm",
                ext_id="HNHT.VSIStorageSystem.localSetParameters.bLoadingAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.VSIStorageSystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.SV307.control.bAlgorithmModeActivated",
                ext_id="HNHT.VSIStorageSystem.SV307.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT.VSIStorageSystem.SV307.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.VSIStorageSystem.SV307.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.SV307.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.VSIStorageSystem.SV307.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.SV307.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT.VSIStorageSystem.SV307.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.localState.fHeatFlowRate",  # name like in live env
                ext_id="HNHT.VSIStorageSystem.WMZ305.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.VSIStorage.localState.fUpperTemperature",
                ext_id="HNHT.VSIStorageSystem.VSIStorage.localState.fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT.VSIStorageSystem.VSIStorage.localState.fLowerTemperature",
                ext_id="HNHT.VSIStorageSystem.VSIStorage.localState.fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # central machine heating system
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.control.bAlgorithmModeActivated",
                ext_id="HNHT.CentralMachineHeatingSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CentralMachineHeatingSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CentralMachineHeatingSystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.PU300.control.bAlgorithmModeActivated",
                ext_id="HNHT.CentralMachineHeatingSystem.PU300.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.PU300.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.CentralMachineHeatingSystem.PU300.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.PU300.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.CentralMachineHeatingSystem.PU300.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.PU300.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT.CentralMachineHeatingSystem.PU300.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=6,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatEnergy",
                ext_id="HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatFlowRate",
                ext_id="HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fReturnTemperature",  # machine temperature
                ext_id="HNHT.CentralMachineHeatingSystem.WMZ300.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            # static heating system
            StateVar(
                name="HNHT.StaticHeatingSystem.control.bAlgorithmModeActivated",
                ext_id="HNHT.StaticHeatingSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.StaticHeatingSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.SV.control.bAlgorithmModeActivated",
                ext_id="HNHT.StaticHeatingSystem.SV.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT.StaticHeatingSystem.SV.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.StaticHeatingSystem.SV.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.SV.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.StaticHeatingSystem.SV.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.SV.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT.StaticHeatingSystem.SV.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.RV350.control.bAlgorithmModeActivated",
                ext_id="HNHT.StaticHeatingSystem.RV350.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT.StaticHeatingSystem.RV350.control.bSetStatusOnAlgorithm",
                ext_id="HNHT.StaticHeatingSystem.RV350.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.RV350.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT.StaticHeatingSystem.RV350.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.RV350.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT.StaticHeatingSystem.RV350.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatEnergy",
                ext_id="HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatFlowRate",
                ext_id="HNHT.StaticHeatingSystem.WMZ350.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # static heating room temperature - use emonio
            StateVar(
                name="HNHT.StaticHeatingSystem.ConsumerTemperature.Celsius",
                ext_id="HNHT.StaticHeatingSystem.ConsumerTemperature.Celsius",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            # 100 - heating mode - hard code low = high value = 100
            StateVar(
                name="HNHT.StaticHeatingSystem.SV.setPointState.fOperatingPoint",
                ext_id="HNHT.StaticHeatingSystem.SV.setPointState.fOperatingPoint",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # heat pump system - hnht hnlt
            StateVar(
                name="HNHT_HNLT.HeatPump1System.control.bAlgorithmModeActivated",
                ext_id="HNHT_HNLT.HeatPump1System.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.control.bSetStatusOnAlgorithm",
                ext_id="HNHT_HNLT.HeatPump1System.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT_HNLT.HeatPump1System.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.RV342.control.bAlgorithmModeActivated",
                ext_id="HNHT_HNLT.HeatPump1System.RV342.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.RV342.control.bSetStatusOnAlgorithm",
                ext_id="HNHT_HNLT.HeatPump1System.RV342.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.RV342.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT_HNLT.HeatPump1System.RV342.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.RV342.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT_HNLT.HeatPump1System.RV342.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.WMZ342.sensorState.fHeatEnergy",
                ext_id="HNHT_HNLT.HeatPump1System.WMZ342.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.WMZ342.sensorState.fHeatFlowRate",
                ext_id="HNHT_HNLT.HeatPump1System.WMZ342.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.WMZ342.sensorState.fReturnTemperature",
                ext_id="HNHT_HNLT.HeatPump1System.WMZ342.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.fReturnTemperature_cold.Celsius",  # temperature sensor
                ext_id="HNHT_HNLT.HeatPump1System.fReturnTemperature_cold.Celsius",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT_HNLT.HeatPump1System.P_el",  # use emonio
                ext_id="HNHT_HNLT.HeatPump1System.P_el",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # heat exchanger 1
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.control.bAlgorithmModeActivated",
                ext_id="HNHT_HNLT.HeatExchanger1System.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.control.bSetStatusOnAlgorithm",
                ext_id="HNHT_HNLT.HeatExchanger1System.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT_HNLT.HeatExchanger1System.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.RV315.control.bAlgorithmModeActivated",
                ext_id="HNHT_HNLT.HeatExchanger1System.RV315.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.RV315.control.bSetStatusOnAlgorithm",
                ext_id="HNHT_HNLT.HeatExchanger1System.RV315.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.RV315.setSetPoint.fSetPointAlgorithm",
                ext_id="HNHT_HNLT.HeatExchanger1System.RV315.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.RV315.localSetParameters.nControlModeAlgorithm",
                ext_id="HNHT_HNLT.HeatExchanger1System.RV315.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.WMZ215.sensorState.fHeatEnergy",
                ext_id="HNHT_HNLT.HeatExchanger1System.WMZ215.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNHT_HNLT.HeatExchanger1System.WMZ215.sensorState.fHeatFlowRate",
                ext_id="HNHT_HNLT.HeatExchanger1System.WMZ215.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # inner capillary tube mats
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.control.bAlgorithmModeActivated",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.control.bSetStatusOnAlgorithm",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.SV.control.bAlgorithmModeActivated",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.SV.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.SV.control.bSetStatusOnAlgorithm",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.SV.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.SV.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.SV.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.SV.localSetParameters.nControlModeAlgorithm",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.SV.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.RV500.control.bAlgorithmModeActivated",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.RV500.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.RV500.control.bSetStatusOnAlgorithm",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.RV500.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.RV500.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.RV500.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.RV500.localSetParameters.nControlModeAlgorithm",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.RV500.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatEnergy",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatFlowRate",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.WMZ405.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT_CN.InnerCapillaryTubeMats.setPointState.fOperatingPoint",  # production hall temperature
                ext_id="HNLT_CN.InnerCapillaryTubeMats.setPointState.fOperatingPoint",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            StateVar(
                # 100 - heating mode - hard code low = high value = 100
                name="HNLT_CN.InnerCapillaryTubeMats.SVHNLT_CN.setPointState.fOperatingPoint",
                ext_id="HNLT_CN.InnerCapillaryTubeMats.SVHNLT_CN.setPointState.fOperatingPoint",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            # hnlt hvfa
            StateVar(
                name="HNLT.HVFASystem.control.bAlgorithmModeActivated",
                ext_id="HNLT.HVFASystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.HVFASystem.control.bSetStatusOnAlgorithm",
                ext_id="HNLT.HVFASystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.HVFASystem.localSetParameters.bLoadingAlgorithm",
                ext_id="HNLT.HVFASystem.localSetParameters.bLoadingAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.HVFASystem.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT.HVFASystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT.HVFASystem.RVx05.control.bAlgorithmModeActivated",
                ext_id="HNLT.HVFASystem.RVx05.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNLT.HVFASystem.RVx05.control.bSetStatusOnAlgorithm",
                ext_id="HNLT.HVFASystem.RVx05.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.HVFASystem.RVx05.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT.HVFASystem.RVx05.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="HNLT.HVFASystem.RVx05.localSetParameters.nControlModeAlgorithm",
                ext_id="HNLT.HVFASystem.RVx05.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNLT.HVFASystem.HVFAStorage.localState.fUpperTemperature",
                ext_id="HNLT.HVFASystem.HVFAStorage.localState.fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT.HVFASystem.WMZx05.sensorState.fHeatFlowRate",
                ext_id="HNLT.HVFASystem.WMZx05.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT.HVFASystem.HVFAStorage.localState.fLowerTemperature",
                ext_id="HNLT.HVFASystem.HVFAStorage.localState.fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # compressor
            StateVar(
                name="HNLT.CompressorSystem.control.bAlgorithmModeActivated",
                ext_id="HNLT.CompressorSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.CompressorSystem.control.bSetStatusOnAlgorithm",
                ext_id="HNLT.CompressorSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.CompressorSystem.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT.CompressorSystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT.CompressorSystem.RV251.control.bAlgorithmModeActivated",
                ext_id="HNLT.CompressorSystem.RV251.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNLT.CompressorSystem.RV251.control.bSetStatusOnAlgorithm",
                ext_id="HNLT.CompressorSystem.RV251.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.CompressorSystem.RV251.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT.CompressorSystem.RV251.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNLT.CompressorSystem.RV251.localSetParameters.nControlModeAlgorithm",
                ext_id="HNLT.CompressorSystem.RV251.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNLT.CompressorSystem.WMZ251.sensorState.fHeatEnergy",
                ext_id="HNLT.CompressorSystem.WMZ251.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT.CompressorSystem.WMZ251.sensorState.fHeatFlowRate",
                ext_id="HNLT.CompressorSystem.WMZ251.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT.CompressorSystem.WMZ251.sensorState.fReturnTemperature",  # compressor temperature
                ext_id="HNLT.CompressorSystem.WMZ251.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            # underfloor heating
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.control.bAlgorithmModeActivated",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.control.bSetStatusOnAlgorithm",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.SV.control.bAlgorithmModeActivated",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.SV.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.SV.control.bSetStatusOnAlgorithm",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.SV.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.SV.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.SV.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.SV.localSetParameters.nControlModeAlgorithm",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.SV.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.RV425.control.bAlgorithmModeActivated",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.RV425.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.RV425.control.bSetStatusOnAlgorithm",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.RV425.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.RV425.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.RV425.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.RV425.localSetParameters.nControlModeAlgorithm",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.RV425.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatEnergy",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatFlowRate",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.WMZ425.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT_CN.UnderfloorHeatingSystem.ConsumerTemperature.Celsius",  # use emonio
                ext_id="HNLT_CN.UnderfloorHeatingSystem.ConsumerTemperature.Celsius",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            StateVar(
                # 100 - heating mode - hard code low = high value = 100
                name="HNLT_CN.UnderfloorHeatingSystem.SV423.setPointState.fOperatingPoint",
                ext_id="HNLT_CN.UnderfloorHeatingSystem.SV423.setPointState.fOperatingPoint",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            # heat exchanger 6 (outer capillary tube mats)
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.control.bAlgorithmModeActivated",
                ext_id="HNLT.OuterCapillaryTubeMats.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.control.bSetStatusOnAlgorithm",
                ext_id="HNLT.OuterCapillaryTubeMats.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT.OuterCapillaryTubeMats.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.RV600.control.bAlgorithmModeActivated",
                ext_id="HNLT.OuterCapillaryTubeMats.RV600.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.RV600.control.bSetStatusOnAlgorithm",
                ext_id="HNLT.OuterCapillaryTubeMats.RV600.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.RV600.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT.OuterCapillaryTubeMats.RV600.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.RV600.localSetParameters.nControlModeAlgorithm",
                ext_id="HNLT.OuterCapillaryTubeMats.RV600.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.WMZ235.sensorState.fHeatEnergy",
                ext_id="HNLT.OuterCapillaryTubeMats.WMZ235.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.WMZ235.sensorState.fHeatFlowRate",
                ext_id="HNLT.OuterCapillaryTubeMats.WMZ235.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT.OuterCapillaryTubeMats.ConsumerTemperature.Celsius",
                # use temperature sensor for outer capillary tube mats return temperature
                ext_id="HNLT.OuterCapillaryTubeMats.ConsumerTemperature.Celsius",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100),
            # heat pump system - hnlt cn
            StateVar(
                name="HNLT_CN.HeatPump2System.control.bAlgorithmModeActivated",
                ext_id="HNLT_CN.HeatPump2System.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.HeatPump2System.control.bSetStatusOnAlgorithm",
                ext_id="HNLT_CN.HeatPump2System.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="HNLT_CN.HeatPump2System.setSetPoint.fSetPointAlgorithm",
                ext_id="HNLT_CN.HeatPump2System.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT_CN.HeatPump2System.WMZ246.sensorState.fHeatEnergy",
                ext_id="HNLT_CN.HeatPump2System.WMZ246.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT_CN.HeatPump2System.WMZ246.sensorState.fHeatFlowRate",
                ext_id="HNLT_CN.HeatPump2System.WMZ246.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="HNLT_CN.HeatPump2System.P_el",  # use emonio
                ext_id="HNLT_CN.HeatPump2System.P_el",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # eChiller
            StateVar(
                name="CN.eChillerSystem.control.bAlgorithmModeActivated",
                ext_id="CN.eChillerSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.eChillerSystem.control.bSetStatusOnAlgorithm",
                ext_id="CN.eChillerSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.eChillerSystem.setSetPoint.fSetPointAlgorithm",
                ext_id="CN.eChillerSystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN.eChillerSystem.SV138.control.bAlgorithmModeActivated",
                ext_id="CN.eChillerSystem.SV138.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="CN.eChillerSystem.SV138.control.bSetStatusOnAlgorithm",
                ext_id="CN.eChillerSystem.SV138.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.eChillerSystem.SV138.setSetPoint.fSetPointAlgorithm",
                ext_id="CN.eChillerSystem.SV138.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="CN.eChillerSystem.SV138.localSetParameters.nControlModeAlgorithm",
                ext_id="CN.eChillerSystem.SV138.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="CN.eChillerSystem.WMZ138.sensorState.fHeatEnergy",
                ext_id="CN.eChillerSystem.WMZ138.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="CN.eChillerSystem.WMZ138.sensorState.fHeatFlowRate",
                ext_id="CN.eChillerSystem.WMZ138.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="CN.eChillerSystem.WMZ138.sensorState.fReturnTemperature",
                ext_id="CN.eChillerSystem.WMZ138.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN.eChillerSystem.P_el",  # use emonio
                ext_id="CN.eChillerSystem.P_el",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            # cn hvfa
            StateVar(
                name="CN.HVFASystem.control.bAlgorithmModeActivated",
                ext_id="CN.HVFASystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.HVFASystem.control.bSetStatusOnAlgorithm",
                ext_id="CN.HVFASystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.HVFASystem.localSetParameters.bLoadingAlgorithm",
                ext_id="CN.HVFASystem.localSetParameters.bLoadingAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.HVFASystem.setSetPoint.fSetPointAlgorithm",
                ext_id="CN.HVFASystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN.HVFASystem.RVx05.control.bAlgorithmModeActivated",
                ext_id="CN.HVFASystem.RVx05.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="CN.HVFASystem.RVx05.control.bSetStatusOnAlgorithm",
                ext_id="CN.HVFASystem.RVx05.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.HVFASystem.RVx05.setSetPoint.fSetPointAlgorithm",
                ext_id="CN.HVFASystem.RVx05.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=-100e3,
                high_value=100e3,
            ),
            StateVar(
                name="CN.HVFASystem.RVx05.localSetParameters.nControlModeAlgorithm",
                ext_id="CN.HVFASystem.RVx05.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=3,
            ),
            StateVar(
                name="CN.HVFASystem.WMZx05.sensorState.fHeatFlowRate",
                ext_id="CN.HVFASystem.WMZx05.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="CN.HVFASystem.HVFAStorage.localState.fUpperTemperature",
                ext_id="CN.HVFASystem.HVFAStorage.localState.fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN.HVFASystem.HVFAStorage.localState.fLowerTemperature",
                ext_id="CN.HVFASystem.HVFAStorage.localState.fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            # central machine cooling system
            StateVar(
                name="CN.CentralMachineCoolingSystem.control.bAlgorithmModeActivated",
                ext_id="CN.CentralMachineCoolingSystem.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.control.bSetStatusOnAlgorithm",
                ext_id="CN.CentralMachineCoolingSystem.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.setSetPoint.fSetPointAlgorithm",
                ext_id="CN.CentralMachineCoolingSystem.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.PU100.control.bAlgorithmModeActivated",
                ext_id="CN.CentralMachineCoolingSystem.PU100.control.bAlgorithmModeActivated",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1),
            StateVar(
                name="CN.CentralMachineCoolingSystem.PU100.control.bSetStatusOnAlgorithm",
                ext_id="CN.CentralMachineCoolingSystem.PU100.control.bSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.PU100.setSetPoint.fSetPointAlgorithm",
                ext_id="CN.CentralMachineCoolingSystem.PU100.setSetPoint.fSetPointAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=100e3,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.PU100.localSetParameters.nControlModeAlgorithm",
                ext_id="CN.CentralMachineCoolingSystem.PU100.localSetParameters.nControlModeAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0,
                high_value=6,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatEnergy",
                ext_id="CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatEnergy",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatFlowRate",
                ext_id="CN.CentralMachineCoolingSystem.WMZ100.sensorState.fHeatFlowRate",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-100,
                high_value=100,
            ),
            StateVar(
                name="CN.CentralMachineCoolingSystem.WMZ100.sensorState.fReturnTemperature",  # machine temperature
                ext_id="CN.CentralMachineCoolingSystem.WMZ100.sensorState.fReturnTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100)
        )

        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)
        self.action_space, self.observation_space = self.state_config.continuous_spaces()
        self.progress_last_step = 0

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """

        # progress counter for simulation process
        progress = math.floor((self.simulator.time / (self.simulator.stop_time - self.simulator.start_time)) * 100)

        if progress != self.progress_last_step and (progress % 1) == 0:
            self.progress_last_step = progress
            print('[INFO] Simulation progress:', progress, ' % ...')

        # return observation
        observations = super().step(action)

        # proceed in time
        current_time = self.scenario_time[-1] + timedelta(seconds=self.sampling_time)
        self.scenario_time.append(current_time)

        if current_time == self.scenario_time_end:
            self.episode_df = pd.DataFrame(self.state_log)
            self.episode_df.fillna(method="bfill", inplace=True)
            self.episode_df['scenario_time'] = self.scenario_time
            with pd.ExcelWriter(self.result_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                self.episode_df.to_excel(writer, sheet_name='StateVars', index=False)
            # call render (static) method
            from experiments.eta_heating_systems.config.global_ import series_name, run_name
            self.render(series_name, run_name)

        return observations

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """

        # receive observations from simulation
        observations = super().reset(seed=seed)
        return observations

    @staticmethod
    def render(series_name, run_name):
        # set paths and read config
        root_path = pathlib.Path(__file__).parents[1]
        result_path = os.path.join(root_path, "results", series_name, run_name + ".xlsx")
        config_path = os.path.join(root_path, "config", run_name + ".json")
        fig_path = os.path.join(root_path,
                                "results",
                                series_name,
                                "results_" + run_name + ".pdf")
        config = read_config(config_path)
        plotter = Plotter(result_path=result_path, config_path=config_path, fig_size=(16.5 / 2.54, 23.5 / 2.54))

        # figure functions to be called
        if config["environment_specific"]["is_benchmark_scenario"]:
            figure_functions = [plotter.total_system_figure,
                                plotter.physical_figure]
        else:
            figure_functions = [plotter.total_system_figure,
                                plotter.physical_figure,
                                plotter.single_systems_figure,
                                plotter.price_figure]

        # call all functions and save figure to pdf
        with PdfPages(fig_path) as pdf:
            for fig_func in figure_functions:
                fig, _, _, _ = fig_func()
                pdf.savefig(fig)
                plt.close(fig)

        return


if __name__ == '__main__':
    EtaHeatingSystems.render(series_name="eta_heating_systems_mas", run_name="2024_04_09_1_day_s1")
