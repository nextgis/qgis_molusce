# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster
from molusce.models.correlation.manager import CoeffManager
from molusce.models.correlation.model import correlation, cramer, jiu

class TestWoEManager (unittest.TestCase):
    def setUp(self):
        self.X = Raster('../../examples/multifact.tif')
        self.Y = Raster('../../examples/sites.tif')
        
    def test_CoeffManager(self):
        coeff = CoeffManager(self.X, self.Y)
        name = coeff.getName()
        coeff = coeff.get_correlation()
        true_name = self.Y.getFileName()
        self.assertEqual(name, true_name)
        self.X = self.X.getBand(1)
        for i in range(1, self.Y.getBandsCount()+1):
            band_y = self.Y.getBand(i)
            coeff = coeff[i-1]
            self.assertEqual(all(coeff), all([correlation(self.X, band_y),cramer(self.X, band_y),jiu(self.X, band_y)]))
   
if __name__ == "__main__":
    unittest.main()
