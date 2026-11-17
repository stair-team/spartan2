import unittest

import numpy as np
from scipy.sparse import csr_matrix

from spartan.model.fraudar.greedy import fastGreedyDecreasing


class TestFastGreedyDecreasing(unittest.TestCase):
    def test_keeps_full_matrix_when_it_is_best(self):
        # A 2x2 all-ones block scores 4/4 = 1.0.  Deleting one row gives
        # 2/3 ~= 0.667, so the full block must be returned.
        result = fastGreedyDecreasing(
            csr_matrix(np.ones((2, 2))), [1.0, 1.0], maxsize=-1)
        rows, cols, score = result
        self.assertEqual((len(rows), len(cols)), (2, 2))
        self.assertAlmostEqual(score, 1.0)

    def test_keeps_larger_full_matrix_when_it_is_best(self):
        result = fastGreedyDecreasing(
            csr_matrix(np.ones((3, 3))), [1.0] * 3, maxsize=-1)
        rows, cols, score = result
        self.assertEqual((len(rows), len(cols)), (3, 3))
        self.assertAlmostEqual(score, 1.5)

    def test_integer_maxsize_is_still_enforced(self):
        result = fastGreedyDecreasing(
            csr_matrix(np.ones((2, 2))), [1.0, 1.0], maxsize=3)
        rows, cols, score = result
        self.assertLessEqual(len(rows) + len(cols), 3)
        self.assertAlmostEqual(score, 2.0 / 3.0)

    def test_pair_maxsize_is_still_enforced(self):
        result = fastGreedyDecreasing(
            csr_matrix(np.ones((2, 2))), [1.0, 1.0], maxsize=(1, 2))
        rows, cols, score = result
        self.assertLessEqual(len(rows), 1)
        self.assertLessEqual(len(cols), 2)
        self.assertAlmostEqual(score, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
