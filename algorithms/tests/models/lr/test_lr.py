# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.lr.lr import LR



class TestMlpManager (unittest.TestCase):
    def setUp(self):
        self.output  = Raster('../../examples/multifact.tif')
        self.output.setMask([0])
        self.state   = self.output
        self.factors = [Raster('../../examples/sites.tif'), Raster('../../examples/sites.tif')]
        
        self.output1  = Raster('../../examples/data.tif')
        self.state1   = self.output1
        self.factors1 = [Raster('../../examples/fact16.tif')]
        
    def test_LR(self):
        data = [
            [1.0, 1.0, 3.0],
            [3.0, 1.0, 1.0],
            [0,   3.0, 1.0]
        ]
        result = np.ma.array(data = data, mask = (data==0))
        
        lr = LR(ns=0)
        lr.setTrainingData(self.state, self.factors, self.output)
        lr.train()
        predict = lr.getPrediction(self.state, self.factors)
        predict = predict.getBand(1)
        assert_array_equal(predict, result)

        lr = LR(ns=1)
        lr.setTrainingData(self.state1, self.factors1, self.output1)
        lr.train()
        
        predict = lr.getPrediction(self.state1, self.factors1)
        predict = predict.getBand(1)

    
if __name__ == "__main__":
    unittest.main()
