# encoding: utf-8

import sys
sys.path.insert(0, '../../')

import unittest

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster
from molusce.models.woe.manager import WoeManager
from molusce.models.woe.model import woe

class TestWoEManager (unittest.TestCase):
    def setUp(self):
        self.factor = Raster('../models/woe/examples/multifact.tif')
        self.sites  = Raster('../models/woe/examples/sites.tif')
        self.sites.binaryzation([2], 0)
        
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
        w = WoeManager([self.factor], self.sites).getWoe()
        self.assertEqual(len(w), 1)
        
        w = w[0]
        name = w['name']
        weights = w['band1']
        true_weights = woe(self.factraster, self.sitesraster)

        self.assertEqual(name, self.factor.getFileName())        
        self.assertEqual(true_weights, weights)

    
if __name__ == "__main__":
    unittest.main()
