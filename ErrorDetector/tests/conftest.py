import pytest
import requests
from pathlib import Path
import os
from ErrorDetector.preprocessing.data_preprocessing import (
    WindowGenerator,
    split_dataset,
    get_normalized_dataset,
    load_data)

CUR_DIR = Path(__file__).parent.absolute()
TEST_DATA_REL_PATH = 'data/Engine_Timing_sim_data_12_01_22_12_2022.xlsx'
TEST_DATA_ABS_PATH = os.path.join(CUR_DIR, TEST_DATA_REL_PATH)

# This file is a global fixture for all tests

@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    """Disable requests.get network calls."""

    def stunted_get():
        raise RuntimeError("Network access not allowed during testing!")

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: stunted_get())


@pytest.fixture
def get_window_data():
    """Set the data state to the default state."""

    data_frame = load_data(TEST_DATA_ABS_PATH, plot=True)
    data_frame.drop(columns=['time'], inplace=True)
    train_test_val_split = [0.8, 0.1, 0.1]
    train_df, val_df, test_df = split_dataset(data_frame, train_test_val_split)
    train_df, val_df, test_df = get_normalized_dataset(train_df, val_df, test_df)
    kwargs = {'input_width': 24, 'label_width': 24, 'shift': 1,
              'label_columns': ['Engine Dynamics:1', 'Compression:1', 'Sum:1',
                                'Intake Manifold:1', 'Intake Manifold:2',
                                'Throttle:1'],
              'input_columns': ['Engine Dynamics:1', 'Compression:1', 'Sum:1',
                                'Intake Manifold:1', 'Intake Manifold:2',
                                'Throttle:1'],
              'train_df': train_df, 'val_df': val_df, 'test_df': test_df}
    window = WindowGenerator(**kwargs)
    return window
