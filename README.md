# Access Point Search

This repository contains the programatic implementation of Access Point Search Algorithm (APSA) which automates the task of finding the optimal set of Access Points that can aid Deep Learning based Anomaly Detector models in detecting anomalies in a particular system.

## Table Of Contents
1. Getting Started
    - Prerequisites
    - Running the code
3. Configuring APSA

## Getting Started
### Prerequisites
To run the code, it is recommended to create a virtual environment and install the required packages. The required packages are listed in the requirements.txt file. To install the required packages, run the following command in the terminal:
```bash
pip install -r requirements.txt
```

### Execution of APSA
To start APSA on imported data, execute the following command in the terminal:
```bash
python main.py
```

### Steps to use APSA

The following steps are to be followed to use APSA:
1. Place the dataset in the `data` folder.
2. Import the dataset in the `main.py` file using variable `DATA_REL_PATH`.
3. Configure the APSA using the `config.py` file is required.
4. Run the `main.py` file.

The program will run APSA and will provide the optimal set of Access Points that can be used to detect anomalies in the system.

## Configuring APSA

The APSA can be configured using ´config.py´ file. The following parameters can be configured:
1. OBJ_WEIGHTS - This array can be used to represent which performance metrics APSA has to optimize. Each element in the array represents the weightage of the corresponding performance metric. The performance metrics are:
    - Mean Square Error (MSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - Root Mean Square Error (RMSE)

    _Eg: If the user wants to optimize MSE, the user can set the OBJ_WEIGHTS as [1, 0, 0, 0]_

2. MIN_MAX_PROBLEM - This parameter can be used to represent whether the problem is a minimization problem or a maximization problem. If the problem is a minimization problem, set this parameter to `min`. If the problem is a maximization problem, set this parameter to `max`.

3. INPUT_WIDTH - This parameter should be used to specify the number of time-steps that forecaster model gets as input.
4. LABEL_WIDTH - This parameter should be used to specify the number of time-steps that forecaster model has to predict.
5. SHIFT - This parameter should be used to specify the number of time-steps that forecaster model has to shift the input to get the next input.
6. PRINT_ALL_METRICS - This parameter should be used to specify whether the performance metrics has to be shown after every epoch of APSA.
7. NUM_FEATURES - This parameter should be used to specify the number of Access Points that are available for selection.

More technical details about the structure of the code can be found in the [docs](docs) folder.