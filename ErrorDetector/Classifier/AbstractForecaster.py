
import tensorflow as tf
import sys
from permetrics.regression import RegressionMetric
import time
import numpy as np

sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\AccessPointSearch')
from ErrorDetector.Classifier.lstm_sequential_model import LSTMSequentialModel
from ErrorDetector.StationarityTest.adf_test import StationaryTester
from ErrorDetector.preprocessing.data_preprocessing import (
    WindowGenerator,
    split_dataset,
    get_normalized_dataset,
    load_data)
from matplotlib import pyplot as plt


class AbstractForecaster:
    """
    Abstract class for time forecasting. This class contains the following functions:

        - `predict(model, data)` - Returns a prediction results.
        - `plot_model_metrics(history)` - Plots training metrics.
        - `train_model(self, window, plot)` - Returns trained model and history.
        - `evaluate_model(self, window)` - Returns evaluation results.
        - `get_model_performance_metrics(y_true, y_pred, required_metrics: list = ['mse', 'mae', 'rmse', 'mape'])` - Returns model performance metrics.
        - `get_true_and_predicted_values(self, test_dataset)` - Returns true and predicted values
    
    """

    def __init__(self, n_epochs=32, callback_patience=3, model_type='lstm_sequential_model', num_features=1,
                 time_steps_per_batch=24):
        self.train_val_accuracy = None
        self.train_accuracy = None
        if model_type == 'lstm_sequential_model':
            lstm_model_generator = LSTMSequentialModel(num_features, n_epochs, callback_patience)
            self.model = lstm_model_generator.build_and_compile_model()
        elif model_type == 'lstm_functional_model':
            self.model = LSTMSequentialModel.build_and_compile_model()
        else:
            self.model = None
        self.n_epochs = n_epochs
        self.callback_patience = callback_patience
        self.num_features = num_features
        self.time_steps_per_batch = time_steps_per_batch

    def __repr__(self):
        return f"Model: {self.model.name}\n" \
                f"Number of epochs: {self.n_epochs}\n" \
                f"Callback patience: {self.callback_patience}\n" \
                f"Number of features: {self.num_features}\n" \
                f"Time steps per batch: {self.time_steps_per_batch}\n"

    @staticmethod
    def predict(model, data):
        """Forecast for the given data.

        :param data: Current time steps.
        :param model: loaded model from disk.
        :returns: A list of forecast time steps.
        """
        return model.predict(data)

    @staticmethod
    def plot_model_metrics(history):
        plt.figure()
        plt.title('model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(history.history['accuracy'],
                 label='Train accuracy')
        plt.plot(history.history['val_accuracy'],
                 label='Val accuracy')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def train_model(self, window: WindowGenerator, plot=False):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=self.callback_patience,
                                                          mode='min')

        history = self.model.fit(window.train, epochs=self.n_epochs,
                                 validation_data=window.val,
                                 callbacks=[early_stopping])
        if plot:
            self.plot_model_metrics(history)

        self.train_accuracy = history.history['accuracy'][-1]
        self.train_val_accuracy = history.history['val_accuracy'][-1]
        return history, self.model

    def evaluate_model(self, window):
        loss, mae, accuracy = self.model.evaluate(window, verbose=0)
        return accuracy

    @staticmethod
    def get_model_performance_metrics(y_true, y_pred, required_metrics: list = ['mse', 'mae', 'rmse', 'mape']) -> dict:
        """Get the performance metrics of the model."""
        obj_metric = RegressionMetric(y_true.flatten(), y_pred.flatten())
        # metrics_dict = obj_metric.get_metrics_by_dict({
        #     "RMSE": {"decimal": 4},
        #     "MAE": {"decimal": 4},
        #     "MSE": {"decimal": 4},
        #     # "R2": {"decimal": 4},
        #     # "MAPE": {"decimal": 4},
        #     # "SMAPE": {"decimal": 4},
        #     # "MASE": {"decimal": 4},
        #     # "NRMSE": {"decimal": 4},
        # })
        # metrics_dict = obj_metric.get_metrics_by_list_names(required_metrics)
        # metrics_dict = {k: round(v, 4) for k, v in metrics_dict.items()}
        performance_metrics_dict = obj_metric.get_metrics_by_dict({key: {'decimal': 4}
                                                                   for key in required_metrics})

        return performance_metrics_dict

    def get_true_and_predicted_values(self, test_dataset):
        y_pred = self.model.predict(test_dataset)
        y_true_iter = test_dataset.map(lambda x, y: y)
        y_true = np.zeros((self.n_epochs, self.time_steps_per_batch, self.num_features))
        for i in y_true_iter.as_numpy_iterator():
            y_true = np.concatenate((y_true, i), axis=0)
        y_true = y_true[self.n_epochs:, :, :]
        return y_true, y_pred

def main():
    ENGINE_TIMING_DATA_FILE_PATH = '../data/Engine_Timing_sim_data_12_01_22_12_2022.xlsx'
    engine_Timing_data_frame = load_data(ENGINE_TIMING_DATA_FILE_PATH, plot=True)
    engine_Timing_data_frame.drop(columns=['time'], inplace=True)
    stat_tester = StationaryTester()
    for col_name in engine_Timing_data_frame.columns:
        print('=' * 15 + f'Stationary test for {col_name}' + '=' * 15 + '\n')
        stat_tester.test(engine_Timing_data_frame[col_name].to_numpy(), has_trends=True)
    # ## Splitting the data
    train_test_val_split = (0.8, 0.1, 0.1)
    train_df, test_df, val_df = split_dataset(engine_Timing_data_frame, train_test_val_split)
    # ## Normalization of data
    train_df, val_df, test_df = get_normalized_dataset(train_df, val_df, test_df)
    engine_timing_data_set = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
                                             input_width=24, label_width=24, shift=24)

    forecaster = AbstractForecaster(num_features=len(engine_timing_data_set.train_df.columns))
    # forecast_model_history, model = forecaster.train_model(engine_timing_data_set)
    # forecaster.model.save('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\'
    #                       'ErrorDetector\\saved_models\\lstm_sequential_model_'+time.strftime('%Y_%m_%d-%H_%M')+'.h5')
    forecaster.model = tf.keras.models.load_model('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git'
                                                  '\\ErrorDetector\\saved_models\\lstm_sequential_model_2022_12_26-11_03.h5')
    y_true, y_pred = forecaster.get_true_and_predicted_values(engine_timing_data_set.test)
    performance_metrics = forecaster.get_model_performance_metrics(y_true, y_pred)
    print(performance_metrics)


if __name__ == '__main__':
    main()
