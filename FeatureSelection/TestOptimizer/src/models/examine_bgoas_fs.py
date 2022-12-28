
import sys
sys.path.append('D:\STUDY MATERIAL\Masters Study Material\WS2022\Thesis\CodeBase\Work_Space')

from BinaryGOA.specialized_optimizer.BGOA_S import OriginalBGOAS


from sklearn.model_selection import train_test_split
import numpy as np
sys.path.append('D:\STUDY MATERIAL\Masters Study Material\WS2022\Thesis\CodeBase\Work_Space\TestOptimizer\src')
sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Work_Space\\TestOptimizer\\src\\utils')
# from src.config import Config
from config import Config
from data_util import get_dataset
from metric_util import Evaluator


def amend_position(position, lower, upper):
    # bounded_pos = np.clip(position, lower, upper)
    new_pos = [0 if np.random.rand() >= x else 1 for x in position]

    return np.array(new_pos)


def fitness_function(solution):
    evaluator = Evaluator(train_X, test_X, train_Y, test_Y, solution, Config.CLASSIFIER, Config.DRAW_CONFUSION_MATRIX, Config.AVERAGE_METRIC)
    metrics = evaluator.get_metrics()
    if Config.PRINT_ALL:
        print(metrics)
    return list(metrics.values())      # Metrics return: [accuracy, precision, recall, f1]


data = get_dataset(Config.DATASET_NAME)
train_X, test_X, train_Y, test_Y = train_test_split(data.features, data.labels,
            stratify=data.labels, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
print(f"y_label: {test_Y}")

## 1. Define problem dictionary
n_features = data.features.shape[1]
n_labels = len(np.unique(data.labels))
LOWER_BOUND = [0, ] * n_features
UPPER_BOUND = [1.99, ] * n_features

problem = {
    "fit_func": fitness_function,
    "lb": LOWER_BOUND,
    "ub": UPPER_BOUND,
    "minmax": Config.MIN_MAX_PROBLEM,
    "obj_weights": Config.OBJ_WEIGHTS,
    "amend_position": amend_position,
}

## 2. Define algorithm and trial
c_min = 0.00004
c_max = 1.0
epoch = 20
pop_size = 20

model = OriginalBGOAS(epoch, pop_size, c_min, c_max)
best_position, best_fitness = model.solve(problem)
print(f"Best features: {best_position}, Best accuracy: {best_fitness}")


