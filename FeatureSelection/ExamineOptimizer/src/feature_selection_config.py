class FeatureSelectionConfig:
    RANDOM_STATE = 42

    DATASET_NAME = "Engine_Timing"
    NUM_FEATURES = 6
    CLASSIFIER = "lstm_sequential_model"  # 'lstm_sequential_model', 'lstm_functional_model'
    MIN_MAX_PROBLEM = "max"
    OBJ_WEIGHTS = [1, 0, 0, 0]  # Metrics return: [accuracy, precision, recall, f1]
    PRINT_ALL = False
    INPUT_WIDTH = 24
    LABEL_WIDTH = 24
    SHIFT = 24
