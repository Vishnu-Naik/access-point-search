# Error Detector

This folder contains all the code related to the Error Detector module. This module is responsible for detecting the anomalies in the system. The following are the folder structure and the description of each file:

## Folder Structure

```
.
├── Classifier
│   ├── AbstractForecaster.py
│   └── lstm_sequnetial_model.py
├── preprocessing
│   └── data_preprocessing.py
├── StationarityTest
│   ├── adf_test.py
└── tests # This folder contains the unit tests for the code
    ├── data    # This folder contains the data required for testing
    ├── conftest.py
    ├── test_abstact_forecaster.py
    └── test_data_processing.py

```

## File Description

1. `AbstractForecaster.py` - This file contains the abstract class for the forecaster model. In this project we are using LSTM based forecaster model. The forecaster model is responsible for predicting the next value in the time-series.

2. `lstm_sequnetial_model.py` - This file contains the implementation of the forecaster model using sequential model. But any forecaster model can be used as long as it appopriately imported in the `AbstractForecaster` class.

3. `data_preprocessing.py` - This file contains the implementation of the data preprocessor. The data preprocessor is responsible for reading the data and converting it into the format that is required by the forecaster model.

    A custom window generator is designed to generate the input and output data for the forecaster model. The window generator is responsible for generating the input and output data for the forecaster model. This window generator is very powerful as it can be used to generate any length of input and output data for the forecaster model.

4. `adf_test.py` - This file contains the implementation of the Augmented Dickey-Fuller test. This test is used to check whether the time-series is stationary or not.


