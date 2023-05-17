import numpy as np
import sys
import logging
from pathlib import Path
from tabulate import tabulate

from ErrorDetector.StationarityTest.adf_test import StationaryTester
from ErrorDetector.preprocessing.data_preprocessing import (
    split_dataset,
    get_normalized_dataset,
    load_data)
from FeatureSelection.BinaryGOA.specialized_optimizer.BGOA_S import OriginalBGOAS
from FeatureSelection.ExamineOptimizer.src.models.APSA import FeatureSelection
from config import APSAConfig as Config


CUR_DIR = Path.cwd()
DATA_REL_PATH = r'data/Engine_timing_with_dynamic_setpoints_230320231707.xlsx'
DATA_ABS_PATH = CUR_DIR / DATA_REL_PATH


logging_level = logging.INFO
logger = logging.getLogger('FeatureSelection')
logger.setLevel(logging_level)
sh = logging.StreamHandler()

logger_format = '%(asctime)s - %(filename)s::%(funcName)s::(line no.:%(lineno)d) - [%(levelname)s] - %(message)s'
formatter = logging.Formatter(logger_format)
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.debug("Starting the program in debug mode")

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def main():
    print(f"Data Preprocessing Started...")
    engine_timing_data_frame = load_data(DATA_ABS_PATH)
    if engine_timing_data_frame is None:
        print(f"Data loading Failed...")
        raise Exception("Data loading Failed")

    print(f"Data loaded from {DATA_ABS_PATH}")
    # Stationarity test
    print(f"Stationarity test Started...")
    stat_tester = StationaryTester()
    stat_test_result_dict = {
        'signal_name': [],
        'stationarity': []
    }
    for col_name in engine_timing_data_frame.columns:
        stat_test_result_dict['signal_name'].append(col_name)
        # print('=' * 15 + f'Stationary test for {col_name}' + '=' * 15 + '\n')
        stat_test_result = stat_tester.test(engine_timing_data_frame[col_name].to_numpy(),
                                            has_trends=True, verbose=False)
        stat_test_result_dict['stationarity'].append(stat_test_result)
    print(tabulate(stat_test_result_dict, headers='keys', tablefmt='outline'))
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
    epoch = 16
    pop_size = 10
    objective_metrics = np.array(['mse', 'mae', 'rmse', 'mape'])
    selected_metrics = objective_metrics[np.flatnonzero(Config.OBJ_WEIGHTS)]
    parameters_dict = [
        ["c_min", c_min],
        ["c_max", c_max],
        ["Number of Epoch", epoch],
        ["Population Size", pop_size],
        ["Number of features", Config.NUM_FEATURES],
        ["Feature Names", '\n'.join(list(norm_train_df.columns.to_numpy()))],
        ["Selected objective", selected_metrics],
    ]
    headers = ["Parameter Name", "Value"]
    print(tabulate(parameters_dict, headers, tablefmt='grid'))
    feature_selector = FeatureSelection((norm_train_df, norm_val_df, norm_test_df), epoch=epoch, pop_size=pop_size)
    feature_selector.create_problem()

    model = OriginalBGOAS(epoch, pop_size, c_min, c_max)
    best_position, best_fitness = model.solve(feature_selector.problem)
    print("=" * 20 + f"Results" + "=" * 20)
    print(f"Best features: {feature_selector.decode_solution(best_position, np.array(norm_train_df.columns))}\n"
          f"Best performance result of selected metric: {best_fitness}")


if __name__ == "__main__":
    main()
