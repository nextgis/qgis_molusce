# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

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
            #~ [1,2,1],
            #~ [1,2,1],
            #~ [0,1,2]
    def test_MCE(self):

        data = [
            [1.0,     4.0, 6.0, 7.0],
            [1.0/4,   1.0, 3.0, 4.0],
            [1.0/6, 1.0/3, 1.0, 2.0],
            [1.0/7, 1.0/4, 1.0/2, 1]
        ]
        # Multiband
        factor = Raster('../../examples/two_band.tif')
        mce = MCE([self.factor, factor, self.factor], data, 1, 2)
        w = mce.getWeights()
        answer = [0.61682294, 0.22382863, 0.09723423, 0.06211421]
        assert_almost_equal(w, answer)

        # One-band
        mce = MCE([self.factor, self.factor, self.factor, self.factor], data, 1, 2)
        w = mce.getWeights()
        answer = [0.61682294, 0.22382863, 0.09723423, 0.06211421]
        assert_almost_equal(w, answer)

        mask = [
            [False, False, False],
            [False, False, False],
            [False, False, True]
        ]
        p = mce.getPrediction(self.state).getBand(1)
        answer = [      # The locations where the big numbers are stored must be masked (see mask and self.state)
            [2, 2, 2],
            [2, 2, 2],
            [100, 2, 100]
        ]
        answer = np.ma.array(data = answer, mask = mask)
        assert_almost_equal(p, answer)
        c = mce.getConfidence().getBand(1)
        answer = [      # The locations where the big numbers are stored must be masked (see mask and self.state)
            [1.0/3,  0,  1  ],
            [1,     0,  1.0/3],
            [1000,  1,  1000]
        ]
        answer = np.ma.array(data = answer, mask = mask)
        assert_almost_equal(c, answer)

if __name__ == "__main__":
    unittest.main()
