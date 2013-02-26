# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest
import math

import numpy as np
from numpy import ma as ma

from molusce.algorithms.models.crosstabs.manager  import CrossTableManager
from molusce.algorithms.dataprovider import Raster


class TestCrossTableManager(unittest.TestCase):
    def setUp(self):
        self.factor = Raster('../../examples/multifact.tif')
        self.sites  = Raster('../../examples/sites.tif')

    
    def test_getTransitionMatrix(self):
        table = CrossTableManager(self.factor, self.sites)
        m = table.getTransitionMatrix()
        for i in range(2):
            self.assertAlmostEqual(sum(m[i,:]), 1.0)
    
    
if __name__ == "__main__":
    unittest.main()
