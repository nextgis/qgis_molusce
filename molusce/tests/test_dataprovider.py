# encoding: utf-8

import sys
sys.path.insert(0, '../../')

import unittest
from numpy.testing import assert_array_equal
import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster

class TestRaster (unittest.TestCase):
    def setUp(self):
        self.r1 = Raster('../models/woe/examples/multifact.tif')
        self.r2 = Raster('../models/woe/examples/sites.tif')
        
    def test_Raster(self):
        self.assertEqual(self.r1.getBandsCount(), 1)
        b1 = self.r1.getBand(1)
        shape = b1.shape
        self.assertEqual(shape, (3, 3))
        
        self.assertEqual(self.r2.getBandsCount(), 1)
        b1 = self.r2.getBand(1)
        
        data = np.array(
            [
                [1,2,1],
                [1,2,1],
                [0,1,2]
            ])
        mask = [
            [False, False, False],
            [False, False, False],
            [False, False, False]
        ]
        data = ma.array(data=data, mask=mask)
        assert_array_equal(b1, data)
        

    
if __name__ == "__main__":
    unittest.main()


