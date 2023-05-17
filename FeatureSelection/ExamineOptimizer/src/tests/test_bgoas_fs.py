import sys
import numpy as np

import FeatureSelection.ExamineOptimizer.src.models.APSA as APSA

def test_amend_position():
    pos = [0.4391244, 0.3797554, 0.56115975, 0.55154934, 0.50944481]
    new_pos = APSA.FeatureSelection.amend_position(pos, [0, 0, 0, 0, 0], [1.99, 1.99, 1.99, 1.99, 1.99])
    print(new_pos)
    assert np.any((new_pos == 1))

