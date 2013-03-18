# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest
from numpy.testing import assert_array_equal

import numpy as np
from numpy import ma as ma

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.woe.manager import WoeManager
from molusce.algorithms.models.woe.model import woe

class TestWoEManager (unittest.TestCase):
    def setUp(self):
        self.factor = Raster('../../examples/multifact.tif')
        self.sites  = Raster('../../examples/sites.tif')
        self.sites.binaryzation([2], 1)
        
        mask = [
            [False, False, False,],
            [False, False, False,],
            [False, False, False,]
        ]
        fact = [
            [1, 1, 3,],
            [3, 2, 1,],
            [0, 3, 1,]
        ]
        site = [
            [False, True,  False,],
            [False, True,  False,],
            [False, False, True,]
        ]
        self.factraster  = ma.array(data = fact, mask=mask, dtype=np.int)
        self.sitesraster = ma.array(data = site, mask=mask, dtype=np.bool)
        
    def test_WoeManager(self):
        w1 = WoeManager([self.factor], self.sites).getWoe()
        w2 = WoeManager([self.factor, self.factor], self.sites).getWoe()
        print w2
        print w1


    
if __name__ == "__main__":
    unittest.main()
