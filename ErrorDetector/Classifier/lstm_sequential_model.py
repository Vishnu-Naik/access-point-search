#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import sys

sys.path.append('D:\\STUDY MATERIAL\\Masters Study Material\\WS2022\\Thesis\\CodeBase\\Git\\ErrorDetector')

from StationarityTest.adf_test import StationaryTester
from preprocessing.data_preprocessing import (
    WindowGenerator,
    split_dataset,
    get_normalized_dataset,
    load_data)
from matplotlib import pyplot as plt


class LSTMSequentialModel:
    """
    A class which build a multi-input multi-output LSTM model in sequential style.

    This class contains the following functions:

    1. `build_and_compile_model(self)` - Returns a compiled model.

    For Example:

    >>> lstm_model_generator = LSTMSequentialModel(num_features=1, n_epochs=32, callback_patience=3)
    >>> lstm_model_generator
    LSTMSequentialModel(num_features=1, n_epochs=32, callback_patience=3)

    >>> lstm_model = lstm_model_generator.build_and_compile_model()

    """
    def __init__(self, num_features, n_epochs=32, callback_patience=3):
        # self.model = self.build_and_compile_model(num_features)
        self.num_features = num_features
        self.n_epochs = n_epochs
        self.callback_patience = callback_patience

    def __repr__(self):
        return f'LSTMSequentialModel(num_features={self.num_features}, n_epochs={self.n_epochs}, callback_patience={self.callback_patience})'

    def build_and_compile_model(self):
        """Build and compile the model.

        Returns:
            model: Compiled model.
        """
        model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=self.num_features)
        ], name='LSTM_Sequential_Model')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), 'accuracy'])

        return model

    # @staticmethod
    # def plot_model_metrics(history):
    #     plt.figure()
    #     plt.title('model accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.plot(history.history['accuracy'],
    #              label='Train accuracy')
    #     plt.plot(history.history['val_accuracy'],
    #              label='Val accuracy')
    #     plt.legend(['train', 'val'], loc='upper left')
    #     plt.show()

    # def train_model(self, window, plot=False):
    #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                       patience=self.callback_patience,
    #                                                       mode='min')
    #
    #     history = self.model.fit(window.train, epochs=self.n_epochs,
    #                              validation_data=window.val,
    #                              callbacks=[early_stopping])
    #     if plot:
    #         self.plot_model_metrics(history)
    #
    #     return history


def main():
    # ## Engine Timing model simulation data forecasting
    # Here we are considering the simulation data obtained from the above said model. In total we have 6 time series.
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

    forecast_model = LSTMSequentialModel(num_features=len(engine_timing_data_set.train_df.columns))
    forecast_model_history = forecast_model.train_model(engine_timing_data_set)
    print(forecast_model_history.history)


if __name__ == '__main__':
    main()
