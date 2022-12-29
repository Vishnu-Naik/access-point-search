import pytest

from ErrorDetector.preprocessing.data_preprocessing import load_data
import tensorflow as tf
import pandas as pd
import os
from pathlib import Path

CUR_DIR = Path(__file__).parent.absolute()


@pytest.mark.data_related
@pytest.mark.parametrize("example_excel_file_path", ["../data/Engine_Timing_sim_data_12_01_22_12_2022.xlsx",
                                                     "wrong_file_name.xlsx"])
def test_load_data(example_excel_file_path):
    """Test load data."""
    data_excel_file_name = example_excel_file_path
    data_excel_file_path = os.path.join(CUR_DIR, data_excel_file_name)
    if data_excel_file_name == "wrong_file_name.xlsx":
        with pytest.raises(FileNotFoundError):
            load_data(data_excel_file_path)
    else:
        data_frame = load_data(data_excel_file_path, plot=True)
        assert isinstance(data_frame, pd.DataFrame)
        assert data_frame.columns.tolist() == ['time', 'Compression:1', 'Engine Dynamics:1', 'Sum:1',
                                               'Intake Manifold:1', 'Intake Manifold:2', 'Throttle:1']


def test_window_generator(get_window_data):
    """Test window generator."""
    window = get_window_data
    assert window.__repr__() == \
           'Total window size: 25\n' \
           'Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]\n' \
           'Label indices: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]\n' \
           "Label column name(s): {'Compression:1': 0, 'Engine Dynamics:1': " \
           "1, 'Sum:1': 2, 'Intake Manifold:1': 3, 'Intake Manifold:2': 4, 'Throttle:1': " \
           '5}\n' \
           "Selected input column name(s): ['Engine Dynamics:1', 'Compression:1', 'Sum:1', 'Intake Manifold:1', " \
           "'Intake Manifold:2', 'Throttle:1']\n" \
           "Selected label column name(s): ['Engine Dynamics:1', 'Compression:1', 'Sum:1', 'Intake Manifold:1', " \
           "'Intake Manifold:2', 'Throttle:1']"
    assert window.train.element_spec == (tf.TensorSpec(shape=(None, 24, 6), dtype=tf.float32, name=None)
                                         , tf.TensorSpec(shape=(None, 24, 6), dtype=tf.float32, name=None))
    assert window.val.element_spec == (tf.TensorSpec(shape=(None, 24, 6), dtype=tf.float32, name=None)
                                       , tf.TensorSpec(shape=(None, 24, 6), dtype=tf.float32, name=None))
    assert window.test.element_spec == (tf.TensorSpec(shape=(None, 24, 6), dtype=tf.float32, name=None)
                                        , tf.TensorSpec(shape=(None, 24, 6), dtype=tf.float32, name=None))
    for sample_inputs, sample_labels in window.train.take(1):
        assert f'Inputs shape (batch, time, features): {sample_inputs.shape}' == \
               'Inputs shape (batch, time, features): (32, 24, 6)'
        assert f'Labels shape (batch, time, features): {sample_labels.shape}' == \
               'Labels shape (batch, time, features): (32, 24, 6)'
    assert window.example[0].shape == (32, 24, 6)
    assert window.example[1].shape == (32, 24, 6)
