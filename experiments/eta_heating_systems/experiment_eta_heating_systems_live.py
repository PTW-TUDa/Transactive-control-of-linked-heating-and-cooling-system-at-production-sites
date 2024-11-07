"""
experiment file (main)
"""

__author__ = "Fabian Borst"
__maintainer__ = "Fabian Borst"
__email__ = "f.borst@ptw.tu-darmstadt.de"
__project__ = "Transactive control of linked heating and cooling system at production sites"
__subject__ = "main file to execute experiments"

import pathlib
import experiments.eta_heating_systems.config.global_ as global_
from eta_utility.eta_x import ETAx
import warnings
warnings.filterwarnings("error")


def main() -> None:
    root_path = pathlib.Path(__file__).parent

    global_.series_name = "eta_heating_systems_mas"
    global_.run_name = "2024_09_15_s7_benchmark_live"
    experiment_s7_live = ETAx(root_path, global_.run_name, relpath_config="config")
    experiment_s7_live.play(global_.series_name, global_.run_name)


if __name__ == "__main__":
    main()
