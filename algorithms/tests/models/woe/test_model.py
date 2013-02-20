# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy import ma as ma


from molusce.algorithms.models.woe.model import WoeError, binary_woe, woe, EPSILON


class TestModel (unittest.TestCase):
    
    def setUp(self):
        
        fact = [
            [True,  True,  False,],
            [False, False, True, ],
            [None,  False, True, ]
        ]
        fact1 = [
            [True,  True,  False,],
            [False, False, True, ],
            [None,  False, True, ]
        ]
        site = [
            [False, True,  False,],
            [False, True,  False,],
            [False, False, True, ]
        ]
        site1 = [
            [1, 2, 1,],
            [1, 2, 1,],
            [0, 1, 2,]
        ]
        zero = [
            [False, False, False,],
            [False, False, False,],
            [None,  False, False,]
        ]
        
        mask = [
            [False, False, False,],
            [False, False, False,],
            [True,  False, False,]
        ]
        mask1 = [
            [False, False, False,],
            [False, False, False,],
            [False,  False, False,]
        ]
        multifact = [
            [1, 1, 3,],
            [3, 2, 1,],
            [0, 3, 1,]
        ]
        
        bigfact = [
            [True,  True,  False, True,  True,  False,],
            [False, False, True,  False, False, True, ],
            [None,  False, True,  None,  False, True, ],
            [True,  False, True,  None,  False, True, ]
        ]
        bigsite = [
            [False, True,  False, False, True,  False,],
            [False, True,  False, False, False, True, ],
            [None,  False, False, None,  True,  True, ],
            [False, False, True,  None,  False, False,]
        ]
        bigmask = [
            [False, False, False, False, False, False,],
            [False, False, False, False, False, False,],
            [True,  False, False, True,  False, False, ],
            [False, False, False, True,  False, False,]
        ]
        
        self.factor     = ma.array(data = fact,      mask=mask,     dtype=np.bool)
        self.fact1      = ma.array(data = fact,      mask=mask1,    dtype=np.bool)
        self.multifact  = ma.array(data = multifact, mask=mask,     dtype=np.int)
        self.sites      = ma.array(data = site,      mask=mask,     dtype=np.bool)
        self.sites1     = ma.array(data = site1,     mask=mask1,    dtype=np.int)
        self.sites2     = ma.array(data = site1,     mask=mask,     dtype=np.int)
        self.zero       = ma.array(data = zero,      mask=mask,     dtype=np.bool)
        self.bigfactor  = ma.array(data = bigfact,   mask=bigmask,  dtype=np.bool)
        self.bigsite    = ma.array(data = bigsite,   mask=bigmask,  dtype=np.bool)
    
    def test_binary_woe(self):
        wPlus  = np.math.log ( (2.0/3 + EPSILON)/(2.0/5 + EPSILON) ) 
        wMinus = np.math.log ( (1.0/3 + EPSILON)/(3.0/5 + EPSILON) )        
        self.assertEqual(binary_woe(self.factor, self.sites), (wPlus, wMinus))
        
        wPlus  = np.math.log ( (5.0/7 + EPSILON)/(0.5/3.5 + EPSILON) ) 
        wMinus = np.math.log ( (2.0/7 + EPSILON)/(3.0/3.5 + EPSILON) ) 
        self.assertEqual(binary_woe(self.bigfactor, self.bigsite, unitcell=2), (wPlus, wMinus))
        
        # if Sites=Factor:
        wPlus  = np.math.log ( (1 + EPSILON)/EPSILON )
        wMinus = np.math.log ( EPSILON/(1 + EPSILON)  )
        self.assertEqual(binary_woe(self.factor, self.factor), (wPlus, wMinus))
        
        # Check areas size
        self.assertRaises(WoeError, binary_woe, self.factor, self.zero)
        self.assertRaises(WoeError, binary_woe, self.zero,   self.sites)
        self.assertRaises(WoeError, binary_woe, self.bigfactor, self.bigsite, 3)
        
        # Non-binary sites
        self.assertRaises(WoeError, woe, self.fact1, self.sites1)
        self.assertRaises(WoeError, woe, self.multifact, self.sites2)
        
    def test_woe(self):
        wPlus1  = np.math.log ( (2.0/3 + EPSILON)/(2.0/5 + EPSILON) ) 
        wMinus1 = np.math.log ( (1.0/3 + EPSILON)/(3.0/5 + EPSILON) ) 
        
        wPlus2  = np.math.log ( (1.0/3 + EPSILON)/(EPSILON) ) 
        wMinus2 = np.math.log ( (2.0/3 + EPSILON)/(1.0 + EPSILON) ) 
        
        wPlus3  = np.math.log ( (EPSILON)/(3.0/5 + EPSILON) ) 
        wMinus3 = np.math.log ( (1.0 + EPSILON)/(2.0/5 + EPSILON) )
        
        # Binary classes
        self.assertEqual(woe(self.factor, self.sites), [(wPlus1, wMinus1)])
        
        # Multiclass
        weights = woe(self.multifact, self.sites)        
        self.assertEqual(weights, [(wPlus1, wMinus1), (wPlus2, wMinus2), (wPlus3, wMinus3)])
        
        
        
    
if __name__ == "__main__":
    unittest.main()
