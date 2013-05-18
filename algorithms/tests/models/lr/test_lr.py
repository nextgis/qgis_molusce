# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.lr.lr import LR



class TestLRManager (unittest.TestCase):
    def setUp(self):
        self.output  = Raster('../../examples/multifact.tif')
            #~ [1,1,3]
            #~ [3,2,1]
            #~ [0,3,1]
        self.output.resetMask([0])
        self.state   = self.output
        self.factors = [Raster('../../examples/sites.tif'), Raster('../../examples/sites.tif')]
            #~ [1,2,1],
            #~ [1,2,1],
            #~ [0,1,2]


        self.output1  = Raster('../../examples/data.tif')
        self.state1   = self.output1
        self.factors1 = [Raster('../../examples/fact16.tif')]

    def test_LR(self):
        #~ data = [
            #~ [3.0, 1.0, 3.0],
            #~ [3.0, 1.0, 3.0],
            #~ [0,   3.0, 1.0]
        #~ ]
        #~ result = np.ma.array(data = data, mask = (data==0))

        lr = LR(ns=0)   # 3-class problem
        lr.setTrainingData(self.state, self.factors, self.output)
        lr.train()
        predict = lr.getPrediction(self.state, self.factors)
        predict = predict.getBand(1)
        assert_array_equal(predict, self.output.getBand(1))

        lr = LR(ns=1) # Two-class problem (it's because of boundary effect)
        lr.setTrainingData(self.state1, self.factors1, self.output1)
        lr.train()
        predict = lr.getPrediction(self.state1, self.factors1)
        predict = predict.getBand(1)
        data = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        result = np.ma.array(data = data, mask = (data==0))
        assert_array_equal(predict, result)

if __name__ == "__main__":
    unittest.main()
