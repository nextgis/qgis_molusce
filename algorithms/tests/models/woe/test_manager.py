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
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst


class TestWoEManager (unittest.TestCase):
    def setUp(self):
        self.factor = Raster('../../examples/multifact.tif')

        self.sites  = Raster('../../examples/sites.tif')
        self.sites.resetMask(maskVals= [0])
        
        mask = [
            [False, False, False,],
            [False, False, False,],
            [True,  False, False,]
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
        aa = AreaAnalyst(self.sites, self.sites)
        w1 = WoeManager([self.factor], aa)
        p = w1.getPrediction(self.sites).getBand(1)
        assert_array_equal(p, self.sites.getBand(1))

        initState = Raster('../../examples/data.tif')
        finalState = Raster('../../examples/data1.tif')
        aa = AreaAnalyst(initState, finalState)
        w = WoeManager([initState], aa)
        p = w.getPrediction(initState).getBand(1)

        # Calculate by hands:
        #1->1 transition raster:
        r11 = [
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        #1->2 raster:
        r12 = [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        #1->3 raster:
        r13 = [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        # 2->1
        r21 = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        # 2->2
        r22 = [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        # 2->3
        r23 = [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ]
        # 3->1
        r31 = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0]
        ]
        # 3->2
        r32 = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]
        # 3->3
        r33 = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ]
        geodata = initState.getGeodata()
        sites = {'11': r11, '12': r12, '13': r13, '21': r21, '22': r22, '23': r23, '31': r31, '32': r32, '33': r33}
        woeDict = {}    # WoE of transitions 
        for k in sites.keys(): #
            if k !='21' : # !!! r21 is zero
                x = Raster()
                x.create([np.ma.array(data=sites[k])], geodata)
                sites[k] = x
                woeDict[k] = woe(initState.getBand(1), x.getBand(1))
        #w1max = np.maximum(woeDict['11'], woeDict['12'], woeDict['13'])
        #w2max = np.maximum(woeDict['22'], woeDict['23'])
        #w3max = np.maximum(woeDict['31'], woeDict['32'], woeDict['33'])
        # Answer is index of finalClass that maximizes weights of transiotion initClass -> finalClass 
        answer = [
            [1, 1, 1, 1],
            [1, 1, 3, 3],
            [3, 3, 3, 3],
            [1, 1, 1, 1]
        ]
        assert_array_equal(p, answer)

        w = WoeManager([initState], aa, bins = {0: [[2], ],})
        p = w.getPrediction(initState).getBand(1)
    
if __name__ == "__main__":
    unittest.main()
