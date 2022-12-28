#!/usr/bin/env python
import logging
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# CUR_DIR = Path(__file__).parent.parent.resolve()
CUR_DIR = Path(__file__).parent.absolute()
DATA_EXCEL_FILE_NAME = '../data/Engine_Timing_sim_data_12_01_22_12_2022.xlsx'
DATA_EXCEL_FILE_PATH = os.path.join(CUR_DIR, DATA_EXCEL_FILE_NAME)


# ### Windowing signal signal dataset
def windowed_dataset(series, feature_length, label_length, skip_step=1):
    """Here tensorflow's batch pipeline is used to window the dataset."""
    # feature_length = 10
    # label_length = 3

    features = series.batch(feature_length, drop_remainder=True)
    labels = series.batch(feature_length).skip(skip_step) \
                .map(lambda labels: labels[:label_length])

    predicted_steps = tf.data.Dataset.zip((features, labels))

    for features, label in predicted_steps.take(2):
        print(f'{features.numpy()}, " => ", {label.numpy()}')


class WindowGenerator:
    """
    A Generic window generator class. This class is used to create a windowed dataset.
    
    To generate a windowed dataset, we need to pass the following parameters:
    
        `input_width`: No. of timesteps needed as input feature for each batch
        `label_width`: No. of timesteps needed as target vector for each batch
        `shift`: No. of timesteps to shift the window of target vector for each batch
        `input_columns(Optional)`: List of features needed (in our case it is signals)
        `label_columns(Optional)`: List of labels needed (in our case it is signals)
        `train_df`: Training dataset
        `test_df`: Testing dataset
        `val_df`: Validation dataset
        `batch_size`: Batch size for each batch
    
    For Example:
        >>> train_df = pd.DataFrame(np.ones(100))
        >>> train_df
              0
        0   1.0
        1   1.0
        2   1.0
        3   1.0
        4   1.0
        ..  ...
        95  1.0
        96  1.0
        97  1.0
        98  1.0
        99  1.0
        <BLANKLINE>
        [100 rows x 1 columns]
        >>> val_df = pd.DataFrame(np.ones(100))
        >>> test_df = pd.DataFrame(np.ones(100))
        >>> window = WindowGenerator(input_width=24, label_width=24, shift=1, \
            train_df=train_df, val_df=val_df, test_df=test_df)
        >>> window
        Total window size: 25
        Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
        Label indices: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        Label column name(s): {0: 0}
        Selected input column name(s): None
        Selected label column name(s): None
        >>> window.train.element_spec
        (TensorSpec(shape=(None, 24, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None, 24, 1), dtype=tf.float32, name=None))
        >>> window.test.element_spec
        (TensorSpec(shape=(None, 24, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None, 24, 1), dtype=tf.float32, name=None))
        >>> for sample_inputs, sample_labels in window.train.take(1):
        ...   print(f'Inputs shape (batch, time, features): {sample_inputs.shape}')
        ...   print(f'Labels shape (batch, time, features): {sample_labels.shape}')
        Inputs shape (batch, time, features): (32, 24, 1)
        Labels shape (batch, time, features): (32, 24, 1)

    Directly pass the `train`, `val` and `test` property of `TimeSeriesUtilities` to the `model.fit` method. As shown below 
        
    model.fit(wide_window.train, epochs=MAX_EPOCHS,
                           validation_data=wide_window.val)"""

    def __init__(self, input_width: int, label_width: int, shift: int,
                 train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                 input_columns: list = None, label_columns: list = None, batch_size: int = 32):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = label_columns
        self.input_columns = input_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in
                                          enumerate(input_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

        self._example = None

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.column_indices}',
            f'Selected input column name(s): {self.input_columns}',
            f'Selected label column name(s): {self.label_columns}'
        ])

    def split_window(self, features):
        """Splitting window helper method
        Typically, data in TensorFlow is packed into arrays where the outermost 
        index is across examples (the "batch" dimension).
        The middle indices are the "time" or "space" (width, height) dimension(s).
        The innermost indices are the features.
        
        The inputs and labels are in the shape of (batch, time, features).
        
        __Note:__ The below parameter of TimeSeriesUtilities class controls the window specs:
        1. label_columns - No. of features are needed (in our case it is signals)
        2. input_width - No. of timesteps needed as input feature for each batch
        3. label_width - No. of timesteps needed as target vector for each batch"""

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # Selection of particular columns
        if self.input_columns is not None:
            inputs = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.input_columns],
                axis=-1)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        logging.info(f'Inputs shape: {inputs.shape}')
        logging.info(f'Labels shape: {labels.shape}')

        return inputs, labels
    
    def plot(self, model=None, plot_col=None, max_subplots=3):
        """
        A plotter method to the WindowGenerator, so that we can visualize it better.
        This method will plot the example sample which is a property of this class
        (can be set any sample using example property of this class)

        E.g., WindowGenerator.example = example_inputs, example_labels"""

        if plot_col is None:
            plot_col = list(self.column_indices.keys())[0]

        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        
    def make_dataset(self, data):
        """Generates dataset with batches of (input_window, label_window) pairs"""

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)

        return ds
    
    # Lets put train, test, validation dataset in one place.
    # The most convenient is to attach as property to TimeSeriesUtilities class
    @property
    def train(self):
        """Tensorflow Map dataset containing training dataset"""
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """Tensorflow Map dataset containing validation dataset"""
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """Tensorflow Map dataset containing testing dataset"""
        return self.make_dataset(self.test_df)
    
    # We add one batch to example property that we described earlier
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def get_normalized_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) \
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalizing the data
    Here we are normalizing the data using mean and standard deviation of the data.
    This is done to make the data in range of 0 to 1
    
    Args:
        train_df: train pandas dataframe
        val_df: validation pandas dataframe
        test_df: test pandas dataframe

    Returns:
        train_df, val_df, test_df: normalized train, test, validation pandas dataframe
    """

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    return train_df, val_df, test_df


def load_and_create_dummy_data(file_path: str) -> pd.DataFrame:
    """Loading the data from a csv file which has single column named 'signal1' and replicating it 3 times
    and converting it into pandas dataframe.
    
    Args:
        file_path: path of the file

    Returns:
        data_frame: pandas dataframe
    
    For example:
    >>> load_and_create_dummy_data('data.csv')
    Traceback (most recent call last):
    ...
    FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'
    """

    data_frame = pd.DataFrame()
    file_extension = file_path.split('.')[-1]
    try:
        if file_extension == 'csv':
            data_frame = pd.read_csv(file_path, names=['signal1'])
        elif file_extension == 'xlsx':
            data_frame = pd.read_excel(file_path, names=['signal1'])
    except FileNotFoundError:
        logging.error(f'File not found at path: {file_path}\n')
        raise FileNotFoundError

    # data_frame.head()
    signal2 = data_frame.to_numpy().flatten()
    signal3 = data_frame.to_numpy().flatten()
    data_frame['signal2'] = signal2
    data_frame['signal3'] = signal3
    data_frame.describe()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=data_frame)

    return data_frame


def load_data(file_path: str, plot: bool = False) -> pd.DataFrame:
    """Loading the data from CSV or Excel file and plotting it and converting it into pandas dataframe

    Args:
        file_path: path of the file
        plot: if True, plot the data

    Returns:
        data_frame: pandas dataframe

    For example:
    >>> load_data('data.csv')
    Traceback (most recent call last):
    FileNotFoundError
    """

    file_extension = file_path.split('.')[-1]
    pd_data_frame = pd.DataFrame()

    try:
        if file_extension == 'csv':
            pd_data_frame = pd.read_csv(file_path)
        elif file_extension == 'xlsx':
            pd_data_frame = pd.read_excel(file_path)
    except FileNotFoundError:
        logging.error(f'File not found at path: {file_path}\n')
        raise FileNotFoundError

    # if plot:
    #     plt.figure(figsize=(12, 8))
    #     n_max = pd_data_frame.shape[1]
    #     for n in pd_data_frame.columns:
    #         plt.plot(pd_data_frame[n], label=n)
    #         plt.legend()
    #     plt.xlabel('Time')
    #     plt.show()

    if plot:
        max_n = pd_data_frame.shape[1]
        for n, col_name in enumerate(pd_data_frame.columns):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{col_name}')
            plt.plot(pd_data_frame[col_name],
                     label=col_name, marker='.', zorder=-10)

        plt.xlabel('Time')
        plt.show()

    return pd_data_frame


def split_dataset(data_frame: pd.DataFrame, train_test_val_split: tuple)\
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splitting the data. Here we are splitting the data into train, test and validation set
    
    Args:
        data_frame: a pandas dataframe
        train_test_val_split: a tuple of train, test and validation set ratio

    Returns:
        train_df: train pandas dataframe
        val_df: validation pandas dataframe
        test_df: test pandas dataframe    
    """

    train_df = data_frame[:int(len(data_frame)*train_test_val_split[0])]
    test_df = data_frame[int(len(data_frame)*train_test_val_split[0]):int(
        len(data_frame)*(train_test_val_split[0]+train_test_val_split[1]))]
    val_df = data_frame[int(
        len(data_frame)*(train_test_val_split[0]+train_test_val_split[1])):]

    return train_df, test_df, val_df


def main():
    # ## Loading the data
    data_frame = load_data(DATA_EXCEL_FILE_PATH, plot=True)

    # ## Splitting the data
    train_test_val_split = (0.8, 0.1, 0.1)
    train_df, test_df, val_df = split_dataset(data_frame, train_test_val_split)

    # ## Normalization of data
    #
    # [TODO] The model shouldn't have access to future values in the training set when training,
    # [TODO] and hence normalization should be done using moving averages.
    #
    # But now we are considering simple average and standard deviation for normalization
    train_df, val_df, test_df = get_normalized_dataset(train_df, val_df, test_df)

    w1 = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
                         input_width=24, label_width=24, shift=24,
                         label_columns=['signal1'])

    w2 = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
                         input_width=6, label_width=1, shift=1,
                         input_columns=['signal1', 'signal2', ],
                         label_columns=['signal1', 'signal2', ])

    # train_df[:w2.total_window_size]

    # Stack three slices, the length of the total window. 
    # This is done so that we can see the plot of a feature and taget vector.
    # The plot function of Window Generator only works on example batch of data.
    # If we dont provide the example batch, it will take the one batch of the train dataset.
    example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                               np.array(train_df[100:100+w2.total_window_size]),
                               np.array(train_df[200:200+w2.total_window_size])])

    example_inputs, example_labels = w2.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'Labels shape: {example_labels.shape}')

    # w2.example = example_inputs, example_labels
    # Here we can see that there are 6 time steps input for each batch (each row = one batch)
    # and there is one timestep target value
    w2.plot()
    # We can select the feature to plot by passing the column name
    w2.plot(plot_col='signal2')

    # ## Dataset
    #

    # Each element is an (inputs, label) pair.
    w2.train.element_spec

    for sample_inputs, sample_labels in w2.train.take(1):
        print(f'Inputs shape (batch, time, features): {sample_inputs.shape}')
        print(f'Labels shape (batch, time, features): {sample_labels.shape}')
        # print(f'elements: {sample_inputs}')

    w2.train

    # ## Example dataset creating
    # In this section we mention some example on how dataset can be generated using TimeSeriesUtilities class
    # we created in previous section

    # ### Single step window
    # one timestep input and target

    single_step_window = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
                                         input_width=1, label_width=1, shift=1,
                                         label_columns=['signal1'])

    # ### Multistep step window
    # Multi timestep input and target

    wide_window = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
                                  input_width=24, label_width=24, shift=1,
                                  label_columns=['signal1'])


if __name__ == '__main__':
    main()
