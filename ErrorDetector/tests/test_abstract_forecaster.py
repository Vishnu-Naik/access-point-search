import ErrorDetector.Classifier.AbstractForecaster as AbstractForecaster
import pytest
from pathlib import Path
import tensorflow as tf
import os

CUR_DIR = Path(__file__).parent.absolute()
SAVED_MODEL_FILE_REL_PATH = 'data/test_lstm_sequential_model_2022_12_26-11_03.h5'
SAVED_MODEL_FILE_ABS_PATH = os.path.join(CUR_DIR, SAVED_MODEL_FILE_REL_PATH)


def test_abstract_forecaster(get_window_data):
    window = get_window_data
    forecaster = AbstractForecaster.AbstractForecaster(
        num_features=len(window.train_df.columns),
        model_type='lstm_sequential_model')
    assert forecaster.__repr__() == \
           f"Model: LSTM_Sequential_Model\n" \
           f"Number of epochs: 32\n" \
           f"Callback patience: 3\n" \
           f"Number of features: 6\n" \
           f"Time steps per batch: 24\n"

    forecaster.model = tf.keras.models.load_model(SAVED_MODEL_FILE_ABS_PATH)
    y_true, y_pred = forecaster.get_true_and_predicted_values(window.test)

    num_sample = 0
    for i in window.test.as_numpy_iterator():
        num_sample = num_sample + i[1].shape[0]
        num_time_step = i[1].shape[1]
        num_features = i[1].shape[2]

    assert (num_sample, num_time_step, num_features) == (y_true.shape[0], y_true.shape[1], y_true.shape[2])
    assert (num_sample, num_time_step, num_features) == (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2])

    performance_metrics = forecaster.get_model_performance_metrics(
        y_true, y_pred, required_metrics=['mse', 'mae', 'rmse', 'mape'])
    assert performance_metrics[0] is not None
    assert performance_metrics[1] is not None
    assert performance_metrics[2] is not None
    assert performance_metrics[3] is not None
