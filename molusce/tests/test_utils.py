# encoding: utf-8

import sys
sys.path.insert(0, '../../')

import unittest

import numpy as np

from molusce.utils import masks_identity

class TestRaster (unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1, 2, 1,],
            [1, 2, 1,],
            [0, 1, 2,]
        ])

        self.Y = np.array([
            [1, 1, 3,],
            [3, 2, 1,],
            [0, 3, 1,]
        ])
        self.X = np.ma.array(self.X, mask=(self.X == 0))
        self.Y = np.ma.array(self.Y, mask=(self.Y == 0))
        self.combo_mask = np.array([
            [False, False, False,],
            [False, False, False,],
            [True , False, False,]
        ])
        
    def test_masks_identity(self): 
        self.X, self.Y = masks_identity(self.X, self.Y)
        mask_x = np.matrix.flatten(self.X.mask)
        self.combo_mask = np.matrix.flatten(self.combo_mask)
        k = all(np.equal(mask_x, self.combo_mask))
        self.assertEqual(k, True, 'masks_identify failed')

    
if __name__ == "__main__":
    unittest.main()


