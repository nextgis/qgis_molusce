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
        true_name = self.Y.getFileName()
        self.Y = self.Y.getBand(1)
        self.X = self.X.getBand(1)
        self.assertEqual(name, true_name)        
        self.assertEqual(coeff.getCorr(), correlation(self.X, self.Y))
        self.assertEqual(coeff.getCramer(), cramer(self.X, self.Y))
        self.assertEqual(coeff.getJIU(), jiu(self.X, self.Y))

    
if __name__ == "__main__":
    unittest.main()
