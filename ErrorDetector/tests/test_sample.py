import unittest


# example for unittest style of unit testing starts
class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 5, "Should be 6")


if __name__ == '__main__':
    unittest.main()
# example for unittest style of unit testing ends


# example of pytest style of unit testing starts
def test_sub():
    assert 2 == 2
# example of pytest style of unit testing ends
