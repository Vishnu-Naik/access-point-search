# Feature Selection

This folder contains the heart of APSA as this module implements the APSA. As APSA is inspired by Feature Selection technique, hence the folder is named appropriately. The following are the folder structure and the description of each file:

## Folder Structure

```
.
├── BinaryGOA
│   ├── specialized_optimizer
│   │   ├── __init__.py
│   │   ├── BGOA_M.py
│   │   ├── BGOA_S.py
│   │   ├── BGOA_V.py
│   │   └── GOA.py
│   └── __init__.py
├── ExamineOptimizer
│       ├── src
│       │   ├── models
│       │   │   ├── __init__.py
│       │   │   └── APSA.py
│       │   ├── tests
│       │   │   ├── __init__.py
│       │   │   └── test_bgoa_fs.py
│       │   ├── utils
│       │   │   ├── __init__.py
│       │   │   └── metric_util.py
│       │   └── __init__.py
│       └── __init__.py
└── init__.py

```

## File Description

1. `BGOA_M.py` - This file contains the implementation of the Mution type Binary variant of Grasshopper Optimizer.

2. `BGOA_S.py` - This file contains the implementation of the Binary variant of Grasshopper Optimizer with Sigmoid transfer function.

3. `BGOA_V.py` - This file contains the implementation of the Binary variant of Grasshopper Optimizer with tanh transfer function.

4. `GOA.py` - This file contains the implementation of the Grasshopper Optimizer Algorithm.

5. `APSA.py` - This file is the most important file in this project as it contains the implementation of the APSA. This file contains the implementation of the APSA algorithm. This file is responsible for selecting the Access Point from the given set of Access Points.

6. `test_bgoa_fs.py` - This file contains the unit tests for the APSA algorithm.

7. `metric_util.py` - This file contains the implementation of the metrics that are used to evaluate the performance of the APSA algorithm.


_Note: In this project we are using the Binary variant of Grasshopper Optimizer with sigmoid transfer function. Hence other varints can be ignored or can be extended in future_
