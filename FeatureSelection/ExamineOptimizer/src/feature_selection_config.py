class FeatureSelectionConfig:
    RANDOM_STATE = 42

    DATASET_NAME = "Engine_Timing"
    NUM_FEATURES = 18
    CLASSIFIER = "lstm_sequential_model"  # 'lstm_sequential_model', 'lstm_functional_model'
    MIN_MAX_PROBLEM = "min"
    OBJ_WEIGHTS = [1, 0, 0, 0]  # Metrics return: ['mse', 'mae', 'rmse', 'mape']
    INPUT_WIDTH = 24
    LABEL_WIDTH = 24
    SHIFT = 24
    PRINT_ALL_METRICS = True
