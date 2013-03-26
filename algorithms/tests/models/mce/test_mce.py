# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy.testing import assert_allclose

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst
from molusce.algorithms.models.mce.mce import MCE


class TestMCE(unittest.TestCase):
    def setUp(self):
        self.factor = Raster('../../examples/multifact.tif')
            #~ [1,1,3]
            #~ [3,2,1]
            #~ [0,3,1]

        self.state  = Raster('../../examples/sites.tif')
        self.state.resetMask(maskVals= [0])
    def test_MCE(self):

        aa = AreaAnalyst(self.state, self.state)
        data = [
            [1.0,     4.0, 6.0, 7.0],
            [1.0/4,   1.0, 3.0, 4.0],
            [1.0/6, 1.0/3, 1.0, 2.0],
            [1.0/7, 1.0/4, 1.0/2, 1]
        ]
        mce = MCE([self.factor, self.factor, self.factor, self.factor], data, aa, 1, 2)
        w = mce.getWeights()
        answer = [0.61682294, 0.22382863, 0.09723423, 0.06211421]
        assert_allclose(w, answer)



if __name__ == "__main__":
    unittest.main()
