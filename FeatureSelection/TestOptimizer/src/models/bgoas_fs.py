import sys

from matplotlib import pyplot as plt

sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\FeatureSelection')
sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\ErrorDetector')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from FeatureSelection.BinaryGOA.specialized_optimizer.BGOA_V import OriginalBGOAV
from ErrorDetector.Classifier.AbstractForecaster import AbstractForecaster
from FeatureSelection.ExamineOptimizer.src.feature_selection_config import FeatureSelectionConfig as Config
from sklearn.metrics import precision_score, recall_score, f1_score, plot_confusion_matrix, accuracy_score

CSV_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'



class FeatureSelection:

    def __init__(self, dataset, epoch, pop_size):
        # self.X_train, self.y_train, self.X_test, self.y_test = dataset[0], dataset[1], dataset[2], dataset[3]
        # self.n_hidden_nodes = n_hidden_nodes

        #
        # self.n_inputs = self.X_train.shape[1]
        # self.model, self.problem_size, self.n_dims, self.problem = None, None, None, None
        # self.optimizer, self.solution, self.best_fit = None, None, None
        self.n_features = Config.NUM_FEATURES
        self.dataset = dataset
        self.epoch = epoch
        self.pop_size = pop_size
        self.problem = None

    def create_network(self):
        # create model
        forecaster = AbstractForecaster(num_features=self.n_features)
        forecast_model_history, model = forecaster.train_model(self.dataset)
        self.model = model
        self.problem_size = self.n_dims = np.sum([np.size(w) for w in self.model.get_weights()])

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
            "amend_position": self.amend_position,
        }

    def get_metrics(self):
        ## Train on training set
        self.clf.fit(self.X_train, self.y_train)

        ## Test and get accuracy on testing set
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=self.average, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=self.average, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=self.average)

        ## Save confusion matrix
        if self.draw_confusion_matrix:
            plot_confusion_matrix(self.clf, self.X_test, self.y_test)
            plt.savefig('confusion_matrix.png')
            plt.title('Confusion Matrix')
            plt.show()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    @staticmethod
    def fitness_function(solution):
        evaluator = Evaluator(train_X, test_X,
                              train_Y, test_Y,
                              solution, Config.CLASSIFIER,
                              Config.DRAW_CONFUSION_MATRIX, Config.AVERAGE_METRIC)
        metrics = FeatureSelection.get_metrics()
        if Config.PRINT_ALL:
            print(metrics)
        return list(metrics.values())

    def amend_position(position, lower, upper):
        # bounded_pos = np.clip(position, lower, upper)
        new_pos = [x if np.random.rand() >= x else np.logical_not(x) for x in position]
        if np.all((new_pos == 0)):
            new_pos[np.random.randint(0, len(new_pos))] = 1
        return np.array(new_pos)

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
    # with requests.Session() as s:
    #     download = s.get(CSV_URL)
    #     decoded_content = download.content.decode('utf-8')
    #     csv_data = open('pima-indians-diabetes.csv', 'wb')
    #     csv_data.write(decoded_content.encode('utf-8'))
    #     csv_data.close()
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    cut = int(0.6 * len(X))  # 60% for training, 40% for testing
    X_train, X_test = X[0: cut], X[cut:]
    y_train, y_test = Y[0: cut], Y[cut:]
    dataset = [X_train, y_train, X_test, y_test]

    n_hidden_nodes = [10, 5]
    epoch = 50
    pop_size = 50

    # define model
    model = FeatureSelection(dataset, n_hidden_nodes, epoch, pop_size)
    # fit model
    model.training()

    # access to the best model - best set of weights - the final weights of neural network
    # model.solution

    # evaluate on test set
    yhat = model.prediction(solution=model.solution, x_data=X_test)
    yhat = np.argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    print('Accuracy: %.3f' % acc)


if __name__ == "__main__":
    main()
