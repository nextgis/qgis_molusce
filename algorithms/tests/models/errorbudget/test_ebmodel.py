# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy import ma as ma

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.errorbudget.ebmodel import EBudget


class TestModel (unittest.TestCase):
    def setUp(self):
        self.reference = Raster('../../examples/data.tif')
            #~ [1,1,1,1],
            #~ [1,1,2,2],
            #~ [2,2,2,2],
            #~ [3,3,3,3]

        self.simulated = Raster('../../examples/data1.tif')
            #~ [1,1,2,3],
            #~ [3,1,2,3],
            #~ [3,3,3,3],
            #~ [1,1,3,2]

    def test_Init(self):
        Rj = {1: np.ma.array(data =
                 [[1.0, 1.0, 1.0, 1.0],
                 [1.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0]],
                             mask =
                 [[False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False]],
                ),
            2: np.ma.array(data =
                 [[0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 1.0],
                 [0.0, 0.0, 0.0, 0.0]],
                             mask =
                 [[False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False]],
                ),
            3: np.ma.array(data =
                 [[0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [1.0, 1.0, 1.0, 1.0]],
                             mask =
                [[False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False],
                 [False, False, False, False]],
                )
        }

        eb = EBudget(self.reference, self.simulated)
        np.testing.assert_equal(eb.Rj, Rj)

    def test_NoNo(self):
        eb = EBudget(self.reference, self.simulated)

        noNo = eb.NoNo()
        answer = 1.0/3
        np.testing.assert_almost_equal(noNo, answer)





if __name__ == "__main__":
    unittest.main()
