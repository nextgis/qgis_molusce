# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy import ma as ma

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.errorbudget.ebmodel import EBudget, weightedSum


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

    def test_weightedSum(self):
        eb = EBudget(self.reference, self.simulated)
        W = eb.W

        S1 = weightedSum(eb.Sj[1], W)
        S2 = weightedSum(eb.Sj[2], W)
        S3 = weightedSum(eb.Sj[3], W)
        self.assertEqual(S1, 5.0/16)
        self.assertEqual(S2, 3.0/16)
        self.assertEqual(S3, 8.0/16)



    def test_NoNo(self):
        eb = EBudget(self.reference, self.simulated)

        noNo = eb.NoNo()
        answer = 1.0/3
        np.testing.assert_almost_equal(noNo, answer)

    def test_NoMed(self):
        eb = EBudget(self.reference, self.simulated)

        noM = eb.NoMed()
        answer = (6.0*5/16 + 6.0*3/16 + 4.0*8/16)/16
        np.testing.assert_almost_equal(noM, answer)

    def test_MedMed(self):
        eb = EBudget(self.reference, self.simulated)
        medM= eb.MedMed()
        answer = 5.0/16
        np.testing.assert_almost_equal(medM, answer)

    def test_MedPer(self):
        eb = EBudget(self.reference, self.simulated)
        medP= eb.MedPer()
        answer = min(6.0/16, 5.0/16) + min(6.0/16, 3.0/16) + min(4.0/16, 8.0/16)
        np.testing.assert_almost_equal(medP, answer)

    def test_Mask(self):
        reference = Raster('../../examples/data.tif')
        simulated = Raster('../../examples/data1.tif')
        reference.resetMask([2])
        simulated.resetMask([2])

        eb = EBudget(reference, simulated)
        W = eb.W

        S1 = weightedSum(eb.Sj[1], W)
        S3 = weightedSum(eb.Sj[3], W)
        np.testing.assert_almost_equal(S1, 5.0/8)
        np.testing.assert_almost_equal(S3, 3.0/8)


        noNo = eb.NoNo()
        np.testing.assert_almost_equal(noNo, 0.5)

        noM = eb.NoMed()
        np.testing.assert_almost_equal(noM, (5.0*5/8 + 3.0*3/8)/8)

        medM= eb.MedMed()
        np.testing.assert_almost_equal(medM, 4.0/8)

        medP= eb.MedPer()
        np.testing.assert_almost_equal(medP, 1.0)



if __name__ == "__main__":
    unittest.main()
