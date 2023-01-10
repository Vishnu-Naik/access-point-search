
from ErrorDetector.Classifier.AbstractForecaster import AbstractForecaster
from FeatureSelection.ExamineOptimizer.src.feature_selection_config import FeatureSelectionConfig as Config
from ErrorDetector.preprocessing.data_preprocessing import WindowGenerator
import numpy as np
import pandas as pd


class DynamicEvaluator:
    """
    This class is used to dynamically generate a neural network based on the solution provided by the optimizer.
    The solution is a binary array of size equal to the number of features in the dataset.
    The solution is used to select the features that will be used to train the neural network. That is:
    - if the solution[i] = 1, then the feature at index i will be used to train the neural network
    - if the solution[i] = 0, then the feature at index i will not be used to train the neural network

    This class also evaluates and provides performance metrics for the selected features using method `get_metrics()`

    This class houses the following methods:

    Public method:
        - `get_metrics()` - Returns the performance metrics for the selected features
    Abstract methods:
        - `_get_selected_feature_list(data_frame: pd.DataFrame)` - Returns a list of names of selected features
        - `_get_trained_forecaster()` - Returns a trained forecaster
    """
    def __init__(self, norm_train_df, norm_test_df, norm_val_df, solution=None):
        self.forecaster = None
        self.solution = solution
        if self.solution is None:
            self.solution = np.ones(len(norm_train_df.columns))

        # store the train and test features and labels
        self.selected_features_indexes = np.flatnonzero(self.solution)
        self.n_selected_features = len(self.selected_features_indexes)
        self.selected_features_names = self._get_selected_feature_list(norm_train_df)

        self.dataset_window = WindowGenerator(
            input_width=Config.INPUT_WIDTH, label_width=Config.LABEL_WIDTH, shift=Config.SHIFT,
            train_df=norm_train_df, val_df=norm_val_df, test_df=norm_test_df,
            label_columns=self.selected_features_names,
            input_columns=self.selected_features_names)

    def _get_selected_feature_list(self, data_frame: pd.DataFrame):
        """
        Returns a list of names of selected features
        Args:
            data_frame: A pandas data frame containing all the features

        Returns:
            a list of names of selected features
        """
        # get the selected features
        selected_features = data_frame.columns[self.selected_features_indexes]
        return list(selected_features)

    def _get_trained_forecaster(self):
        """
        This method builds a forecaster dynamically using AbstractForecaster based of number of selected features.
        Returns a trained forecaster

        Returns:
            A trained forecaster
        """
        # create model
        forecaster = AbstractForecaster(num_features=self.n_selected_features)
        _, _ = forecaster.train_model(self.dataset_window)
        return forecaster

    def get_metrics(self):
        """
        This method evaluates the performance of the selected features using the trained forecaster.
        Returns a list of performance metrics for the selected features.

        Returns:
            A list of performance metrics
        """
        # Train on training set
        forecaster = self._get_trained_forecaster()
        y_true, y_pred = forecaster.get_true_and_predicted_values(self.dataset_window.test)
        performance_metrics = forecaster.get_model_performance_metrics(y_true, y_pred)
        return performance_metrics
