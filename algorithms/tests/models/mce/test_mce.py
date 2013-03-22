# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy.testing import assert_allclose

from molusce.algorithms.models.mce.mce import MCE



class TestMlpManager (unittest.TestCase):
    def test_LR(self):
        data = [
            [1.0,     4.0, 6.0, 7.0],
            [1.0/4,   1.0, 3.0, 4.0],
            [1.0/6, 1.0/3, 1.0, 2.0],
            [1.0/7, 1.0/4, 1.0/2, 1]
        ]
        mce = MCE(data)
        w = mce.getWeights()
        answer = [0.61682294, 0.22382863, 0.09723423, 0.06211421]
        assert_allclose(w, answer)



if __name__ == "__main__":
    unittest.main()
