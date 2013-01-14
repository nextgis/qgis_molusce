# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster
from molusce.models.correlation.manager import CoeffManager
from molusce.models.correlation.model import 

class TestWoEManager (unittest.TestCase):
    def setUp(self):
        self.X = Raster('../../examples/multifact.tif')
        self.Y = Raster('../../examples/sites.tif')
        
    def test_CoeffManager(self):
        coeff = CoeffManager(self.X, self.Y).getCoeff()
        self.assertEqual(len(coeff), 4)
        true_name = self.Y.getFileName()
        self.Y = self.Y.getBand(1)
        self.X = self.X.getBand(1)
        name  = coeff[0]
        true_coeff = []
        true_coeff.append(correlation(self.X, self.Y))
        true_coeff.append(cramer(self.X, self.Y))
        true_coeff.append(jiu(self.X, self.Y))
        self.assertEqual(name, true_name)        
        self.assertEqual(true_coeff, coeff[1:])

    
if __name__ == "__main__":
    unittest.main()
