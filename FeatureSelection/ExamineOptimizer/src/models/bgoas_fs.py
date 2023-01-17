import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\FeatureSelection')
sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\ErrorDetector')

from FeatureSelection.BinaryGOA.specialized_optimizer.BGOA_S import OriginalBGOAS
from ErrorDetector.StationarityTest.adf_test import StationaryTester
from FeatureSelection.ExamineOptimizer.src.feature_selection_config import FeatureSelectionConfig as Config
from FeatureSelection.ExamineOptimizer.src.utils.metric_util import DynamicEvaluator
from ErrorDetector.preprocessing.data_preprocessing import (
    split_dataset,
    get_normalized_dataset,
    load_data)

CUR_DIR = Path.cwd()
DATA_REL_PATH = '../../data/Engine_Timing_sim_data_without_time_12_01_22_12_2022.xlsx'
DATA_ABS_PATH = CUR_DIR / DATA_REL_PATH


class FeatureSelection:
    """
    A class formulates and modifies the problem to be solved as a feature selection problem.
    This class houses the below methods:
    1. create_problem: This method defines the problem dictionary for the optimizer.
    2. fitness_function: This method defines the fitness function for the problem.
    3. amend_position: This callback function is used to amend the position of the solution based on requirements.
    4. decode_solution: This method decodes the solution to get the names of the selected features.
    """

    def __init__(self, norm_dataset: tuple[pd.DataFrame], epoch, pop_size):
        self.norm_train_df, self.norm_val_df, self.norm_test_df = norm_dataset[0], norm_dataset[1], norm_dataset[2]
        self.model = None
        self.n_features = Config.NUM_FEATURES
        self.epoch = epoch
        self.pop_size = pop_size
        self.problem = None

    def create_problem(self):
        """
        Define problem dictionary for the optimizer.
        """
        # Define problem dictionary
        n_features = Config.NUM_FEATURES
        lower_bound = [0, ] * n_features
        upper_bound = [1.99, ] * n_features
        self.problem = {
            "fit_func": self.fitness_function,
            "lb": lower_bound,
            "ub": upper_bound,
            "minmax": Config.MIN_MAX_PROBLEM,
            "obj_weights": Config.OBJ_WEIGHTS,
            "amend_position": self.amend_position,
        }

    def fitness_function(self, solution) -> list[float]:
        """
        Fitness function for the problem. This is the objective function to be optimized.

        Args:
            solution: list of index of features

        Returns:
            performance_metrics_values(list[float]): list of performance metrics values in the order
            of Config.OBJ_WEIGHTS
        """
        evaluator = DynamicEvaluator(norm_train_df=self.norm_train_df,
                                     norm_val_df=self.norm_val_df,
                                     norm_test_df=self.norm_test_df,
                                     solution=solution)
        performance_metrics = evaluator.get_metrics()
        if Config.PRINT_ALL_METRICS:
            print(performance_metrics)
        performance_metrics_values = list(performance_metrics.values())
        return performance_metrics_values

    @staticmethod
    def amend_position(position, lower, upper) -> list:
        """
        This callback function is used to amend the position of the solution based on requirements.
        This is the main function to be modified to change the behavior of the optimizer.
        This function helps in formulating the problem as a feature selection problem.

        Args:
            position(list): list of values specifying the position of the solution in the search space
            lower(list): list of lower bounds of the search space
            upper(list): list of upper bounds of the search space

        Returns:
            new_pos(list): list of modified value according to desired behaviour of the optimizer

        Note:
            Try to keep the values of the position in the range of [0, 1.99] to keep the problem feature selection
            problem
        """
        # bounded_pos = np.clip(position, lower, upper)
        new_pos = [0 if np.random.rand() >= x else 1 for x in position]
        if np.all((new_pos == 0)):
            new_pos[np.random.randint(0, len(new_pos))] = 1
        return np.array(new_pos)

    @staticmethod
    def decode_solution(solution: np.ndarray[int], feature_names: np.ndarray[int]) -> list[str]:
        """
        Decode the solution to get the names of the selected features

        Args:
            solution(np.ndarray[int]): an array of 0s and 1s
            feature_names(np.ndarray[int]): an array of feature names

        Returns:
            list[str]: a list of names of selected features
        """
        return list(feature_names[np.array(solution) == 1])


def main():
    print(f"Data Preprocessing Started...")
    engine_timing_data_frame = load_data(DATA_ABS_PATH)
    if engine_timing_data_frame is None:
        print(f"Data loading Failed...")
        raise Exception("Data loading Failed")
        return
    print(f"Data loaded from {DATA_ABS_PATH}...")
    # Stationarity test
    stat_tester = StationaryTester()
    for col_name in engine_timing_data_frame.columns:
        print('=' * 15 + f'Stationary test for {col_name}' + '=' * 15 + '\n')
        stat_tester.test(engine_timing_data_frame[col_name].to_numpy(), has_trends=True)
    print('Stationary test finished!!!')
    # Splitting the data
    train_test_val_split = (0.8, 0.1, 0.1)
    train_df, test_df, val_df = split_dataset(engine_timing_data_frame, train_test_val_split)
    # ## Normalization of data
    norm_train_df, norm_val_df, norm_test_df = get_normalized_dataset(train_df, val_df, test_df)
    print(f"Data Preprocessing Finished!!!")
    print(f"Feature Selection using Binary GOA algorithm starting...")
    # 2. Define algorithm and trial
    c_min = 0.00004
    c_max = 1.0
    epoch = 1
    pop_size = 10
    feature_selector = FeatureSelection((norm_train_df, norm_val_df, norm_test_df), epoch=epoch, pop_size=pop_size)
    feature_selector.create_problem()

    model = OriginalBGOAS(epoch, pop_size, c_min, c_max)
    best_position, best_fitness = model.solve(feature_selector.problem)
    print("=" * 20 + f"Results" + "=" * 20)
    print(f"Best features: {feature_selector.decode_solution(best_position, np.array(norm_train_df.columns))}\n"
          f"Best performance result of selected metric: {best_fitness}")


if __name__ == "__main__":
    main()
