# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest
from numpy.testing import assert_array_equal

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster
from molusce.models.area_analysis.manager import AreaAnalyst, AreaAnalizerError

class TestAreaAnalysisManager (unittest.TestCase):
    def setUp(self):
        self.r1 = Raster('../../examples/multifact.tif')
        # r1 -> r1 transition
        self.r1r1 = [
            [5,  5,  15,],
            [15, 10, 5, ],
            [0,  15, 5, ]
        ]
        
        self.r2  = Raster('../../examples/multifact.tif')
        self.r2.setMask([0])
        self.r2r2 = [
            [0,   0, 8,],
            [8,   4, 0,],
            [100, 8, 0,]
        ]
        
        self.r3 = Raster('../../examples/multifact.tif')
        self.r3.setMask([2])
        
    def test_AreaAnalyst(self):
        aa = AreaAnalyst(self.r1, self.r1)
        raster = aa.makeChangeMap()
        band = raster.getBand(1)
        assert_array_equal(band, self.r1r1)
        
        # Masked raster
        aa = AreaAnalyst(self.r2, self.r2)
        raster = aa.makeChangeMap()
        band = raster.getBand(1)
        assert_array_equal(band, self.r2r2)
        
        # Gaps in the class numeration
        self.assertRaises(AreaAnalizerError, AreaAnalyst, self.r3, self.r3)
        
if __name__ == "__main__":
    unittest.main()
