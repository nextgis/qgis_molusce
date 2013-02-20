# encoding: utf-8

import sys
sys.path.insert(0, '../../../')

import unittest

import numpy as np

from molusce.algorithms.utils import masks_identity, sizes_equal

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
        self.X2 = np.array([
            [1, 2, 1,],
            [1, 2, 1,],
            [0, 1, 2,],
            [0, 1, 2,]
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
    
    def test_size_equals(self):
        self.assertEqual(sizes_equal(self.X, self.Y), True, 'incorrent size')
        
    def test_Size_no_equals(self):
        self.assertEqual(sizes_equal(self.X2, self.Y), False, 'sizes are equal')
    
if __name__ == "__main__":
    unittest.main()


