import sys

import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\FeatureSelection')
sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\ErrorDetector')

import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.metrics import accuracy_score
from FeatureSelection.BinaryGOA.specialized_optimizer.BGOA_S import OriginalBGOAS
from ErrorDetector.StationarityTest.adf_test import StationaryTester
from FeatureSelection.ExamineOptimizer.src.feature_selection_config import FeatureSelectionConfig as Config
# from sklearn.metrics import precision_score, recall_score, f1_score, plot_confusion_matrix, accuracy_score
from FeatureSelection.ExamineOptimizer.src.utils.metric_util import DynamicEvaluator
from ErrorDetector.preprocessing.data_preprocessing import (
    split_dataset,
    get_normalized_dataset,
    load_data)
from pathlib import Path
import os

CSV_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
# CUR_DIR = Path(__file__).parent.absolute()
# DATA_EXCEL_FILE_NAME = '../data/Engine_Timing_sim_data_12_01_22_12_2022.xlsx'
# DATA_EXCEL_FILE_PATH = os.path.join(CUR_DIR, DATA_EXCEL_FILE_NAME)

CUR_DIR = Path.cwd()
DATA_REL_PATH = '../../data/Engine_Timing_sim_data_without_time_12_01_22_12_2022.xlsx'
DATA_ABS_PATH = CUR_DIR / DATA_REL_PATH


def amend_position(position, lower, upper):
    pos = np.clip(position, lower, upper).astype(int)
    if np.all((pos == 0)):
        pos[np.random.randint(0, len(pos))] = 1
    return pos


class FeatureSelection:

    def __init__(self, norm_dataset: tuple[pd.DataFrame], epoch, pop_size):
        self.norm_train_df, self.norm_val_df, self.norm_test_df = norm_dataset[0], norm_dataset[1], norm_dataset[2]
        # self.n_hidden_nodes = n_hidden_nodes

        #
        # self.n_inputs = self.X_train.shape[1]
        # self.model, self.problem_size, self.n_dims, self.problem = None, None, None, None
        # self.optimizer, self.solution, self.best_fit = None, None, None
        self.model = None
        self.n_features = Config.NUM_FEATURES
        # self.dataset = dataset
        self.epoch = epoch
        self.pop_size = pop_size
        self.problem = None

    # def get_forecaster(self):
    #     # create model
    #     forecaster = AbstractForecaster(num_features=self.n_features)
    #     forecast_model_history, model = forecaster.train_model(self.dataset)
    #     self.model = model
    #     return forecaster

    def create_problem(self):
        ## 1. Define problem dictionary
        n_features = Config.NUM_FEATURES
        lower_bound = [0, ] * n_features
        upper_bound = [1.99, ] * n_features
        self.problem = {
            "fit_func": self.fitness_function,
            "lb": lower_bound,
            "ub": upper_bound,
            "minmax": Config.MIN_MAX_PROBLEM,
            "obj_weights": Config.OBJ_WEIGHTS,
            "amend_position": amend_position,
        }

    # def encode_column_names(self):
    #     self.col_names = self.dataset.input_colummns

    def fitness_function(self, solution) -> list:
        evaluator = DynamicEvaluator(norm_train_df=self.norm_train_df,
                                     norm_val_df=self.norm_val_df,
                                     norm_test_df=self.norm_test_df,
                                     solution=solution)
        performance_metrics = evaluator.get_metrics()
        if Config.PRINT_ALL_METRICS:
            print(performance_metrics)
        performance_metrics_values = list(performance_metrics.values())
        return performance_metrics_values

    # def amend_position(position, lower, upper):
    #     # bounded_pos = np.clip(position, lower, upper)
    #     new_pos = [x if np.random.rand() >= x else np.logical_not(x) for x in position]
    #     if np.all((new_pos == 0)):
    #         new_pos[np.random.randint(0, len(new_pos))] = 1
    #     return np.array(new_pos)

    # def decode_solution(self, solution):
    #     # solution: is a vector.
    #     # solution = [w11, w21, w31, w12, w22, w32, b1, b2, wh11, wh21, wh12, wh22, wh13, wh23, bo1, bo2, bo3 ]
    #     # number of weights = n_inputs * n_hidden_nodes + n_hidden_nodes + n_hidden_nodes * n_outputs + n_outputs
    #     # we decode the solution into the neural network weights
    #     # we return the model with the new weight (weight from solution)
    #     weight_sizes = [(w.shape, np.size(w)) for w in self.model.get_weights()]
    #     # ( (3, 5),  15 )
    #     weights = []
    #     cut_point = 0
    #     for ws in weight_sizes:
    #         temp = np.reshape(solution[cut_point: cut_point + ws[1]], ws[0])
    #         # [0: 15], (3, 5),
    #         weights.append(temp)
    #         cut_point += ws[1]
    #     self.model.set_weights(weights)
    #
    # def prediction(self, solution, x_data):
    #     self.decode_solution(solution)
    #     return self.model.predict(x_data)

    # def training(self):
    #     self.create_network()
    #     self.create_problem()
    #     self.optimizer = OriginalBGOAV(self.epoch, self.pop_size)
    #     # self.optimizer = FPA.OriginalFPA(self.problem, self.epoch, self.pop_size)
    #     self.solution, self.best_fit = self.optimizer.solve(self.problem)

    # def fitness_function(self, solution):  # Used in training process
    #     # Assumption that we have 3 layer , 1 input layer, 1 hidden layer and 1 output layer
    #     # number of nodes are 3, 2, 3
    #     # solution = [w11, w21, w31, w12, w22, w32, b1, b2, wh11, wh21, wh12, wh22, wh13, wh23, bo1, bo2, bo3 ]
    #     self.decode_solution(solution)
    #     yhat = self.model.predict(self.X_train)
    #     yhat = np.argmax(yhat, axis=-1).astype('int')
    #     acc = accuracy_score(self.y_train, yhat)
    #     return acc


def main():
    print(f"Data Preprocessing Started...")
    engine_Timing_data_frame = load_data(DATA_ABS_PATH)
    # Stationarity test
    stat_tester = StationaryTester()
    for col_name in engine_Timing_data_frame.columns:
        print('=' * 15 + f'Stationary test for {col_name}' + '=' * 15 + '\n')
        stat_tester.test(engine_Timing_data_frame[col_name].to_numpy(), has_trends=True)
    # ## Splitting the data
    train_test_val_split = (0.8, 0.1, 0.1)
    train_df, test_df, val_df = split_dataset(engine_Timing_data_frame, train_test_val_split)
    # ## Normalization of data
    norm_train_df, norm_val_df, norm_test_df = get_normalized_dataset(train_df, val_df, test_df)
    print(f"Data Preprocessing Finished!!!")

    ## 2. Define algorithm and trial
    c_min = 0.00004
    c_max = 1.0
    epoch = 20
    pop_size = 10
    feature_selector = FeatureSelection((norm_train_df, norm_val_df, norm_test_df), epoch=epoch, pop_size=pop_size)
    feature_selector.create_problem()

    model = OriginalBGOAS(epoch, pop_size, c_min, c_max)
    best_position, best_fitness = model.solve(feature_selector.problem)
    print(f"Best position length: {len(best_position)}")
    print(f"Best features: {best_position}, Best accuracy: {best_fitness}")


if __name__ == "__main__":
    main()
