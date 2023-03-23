import sys
import numpy as np

sys.path.append('D:\STUDY MATERIAL\Masters Study Material\WS2022\Thesis\CodeBase\AccessPointSearch\FeatureSelection')

import FeatureSelection.ExamineOptimizer.src.models.bgoas_fs as bgoas_fs

def test_amend_position():
    pos = [0.4391244, 0.3797554, 0.56115975, 0.55154934, 0.50944481]
    new_pos = bgoas_fs.FeatureSelection.amend_position(pos, [0, 0, 0, 0, 0], [1.99, 1.99, 1.99, 1.99, 1.99])
    print(new_pos)
    assert np.any((new_pos == 1))

