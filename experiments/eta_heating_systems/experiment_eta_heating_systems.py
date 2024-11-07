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

    run_days = ["2023_07_12"]

    for run_day in run_days:

        # benchmark run
        global_.run_name = run_day+"_1_day_benchmark"
        experiment_b = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_b.play(global_.series_name, global_.run_name)

        # s0 (base mas) - 900/3600 products, power controlled, using balacing energy,
        # consumer modeled, considering product allocation
        global_.run_name = run_day+"_1_day_s0"
        experiment_s0 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s0.play(global_.series_name, global_.run_name)

        # s1 - neglecting product allocation of converters, not using inherent storage capacity of consumers,
        # reduced product allocation of active storages
        global_.run_name = run_day+"_1_day_s1"
        experiment_s1 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s1.play(global_.series_name, global_.run_name)

        # s2 - no consumer modeling
        global_.run_name = run_day+"_1_day_s2"
        experiment_s2 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s2.play(global_.series_name, global_.run_name)

        # s3 - product variation 1800/3600 - 3600/0 - 900/0 - 1800/0
        global_.run_name = run_day+"_1_day_s3_1"
        experiment_s3_1 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s3_1.play(global_.series_name, global_.run_name)

        global_.run_name = run_day+"_1_day_s3_2"
        experiment_s3_2 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s3_2.play(global_.series_name, global_.run_name)

        global_.run_name = run_day+"_1_day_s3_3"
        experiment_s3_3 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s3_3.play(global_.series_name, global_.run_name)

        global_.run_name = run_day+"_1_day_s3_4"
        experiment_s3_4 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s3_4.play(global_.series_name, global_.run_name)

        # s4 - no balancing energy in hnlt and cn
        global_.run_name = run_day+"_1_day_s4"
        experiment_s4 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s4.play(global_.series_name, global_.run_name)

        # s5 - energy controlled
        global_.run_name = run_day+"_1_day_s5"
        experiment_s5 = ETAx(root_path, global_.run_name, relpath_config="config")
        experiment_s5.play(global_.series_name, global_.run_name)


if __name__ == "__main__":
    main()
