# encoding: utf-8

import sys
import os
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
        self.r3 = Raster('examples/two_band.tif')
        
        # r1
        data1 = np.array(
            [
                [1,1,3],
                [3,2,1],
                [0,3,1]
            ])
        # r2
        data2 = np.array(
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
        self.data1 = ma.array(data=data1, mask=mask)
        self.data2 = ma.array(data=data2, mask=mask)
        
    def test_RasterInit(self):
        self.assertEqual(self.r1.getBandsCount(), 1)
        band = self.r1.getBand(1)
        shape = band.shape
        x = self.r1.getXSize()
        y = self.r1.getYSize()
        self.assertEqual(shape, (x,y))
        
        self.assertEqual(self.r2.getBandsCount(), 1)
        band = self.r2.getBand(1)
        assert_array_equal(band, self.data2)
        
        self.assertTrue(self.r1.geoDataMatch(self.r2))
        
    def test_getNeighbours(self):
        neighbours = self.r2.getNeighbours(row=1,col=0, size=0)
        self.assertEqual(neighbours, [[1]])
        
        neighbours = self.r2.getNeighbours(row=1,col=1, size=1)
        assert_array_equal(neighbours, [self.data2])
        
        neighbours = self.r3.getNeighbours(row=1,col=1, size=1)
        assert_array_equal(neighbours, [self.data2, self.data1])
        
        # Check pixel on the raster bound and nonzero neighbour size
        self.assertRaises(ProviderError, self.r2.getNeighbours, col=1, row=0, size=1)
        self.assertRaises(ProviderError, self.r2.getNeighbours, col=1, row=1, size=2)
        
    def test_geodata(self):
        geodata = self.r1.getGeodata()
        self.r1.setGeoData(geodata)
        geodata['xSize'] = geodata['xSize'] + 10
        self.assertRaises(ProviderError, self.r1.setGeoData, geodata=geodata)
        
        
    def test_save(self):
        try:
            filename = 'temp.tiff'
            self.r1.save(filename)
            r2 = Raster(filename)
            self.assertEqual(r2.get_dtype(), self.r1.get_dtype())
            self.assertEqual(r2.getBandsCount(), self.r1.getBandsCount())
            for i in range(r2.getBandsCount()):
                assert_array_equal(r2.getBand(i+1), self.r1.getBand(i+1))
        finally:
            os.remove(filename)

    
    #~ def test_normalize(self):
        #~ band = self.data2
        #~ band = (band - np.mean(band))/np.std(band)
        #~ self.r2.normalize()
        #~ assert_array_equal([band], self.r2.bands)

    
if __name__ == "__main__":
    unittest.main()


