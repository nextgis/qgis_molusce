# encoding: utf-8

import sys
sys.path.insert(0, '../../')

import unittest
from numpy.testing import assert_array_equal
import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster, ProviderError

class TestRaster (unittest.TestCase):
    def setUp(self):
        self.r1 = Raster('examples/multifact.tif')
        self.r2 = Raster('examples/sites.tif')
        
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
        self.data = ma.array(data=data, mask=mask)
        
    def test_RasterInit(self):
        self.assertEqual(self.r1.getBandsCount(), 1)
        band = self.r1.getBand(1)
        shape = band.shape
        self.assertEqual(shape, (3, 3))
        
        self.assertEqual(self.r2.getBandsCount(), 1)
        band = self.r2.getBand(1)
        assert_array_equal(band, self.data)
        
    def test_getNeighbours(self):
        neighbours = self.r2.getNeighbours(row=1,col=0, size=0)
        self.assertEqual(neighbours, [[1]])
        
        neighbours = self.r2.getNeighbours(row=1,col=1, size=1)
        assert_array_equal(neighbours, [self.data])
        
        # Check pixel on the raster bound and nonzero neighbour size
        self.assertRaises(ProviderError, self.r2.getNeighbours, col=1, row=0, size=1)
        self.assertRaises(ProviderError, self.r2.getNeighbours, col=1, row=1, size=2)

    
if __name__ == "__main__":
    unittest.main()


