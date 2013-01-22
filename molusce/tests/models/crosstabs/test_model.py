# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest
import math

import numpy as np
from numpy import ma as ma

from molusce.models.crosstabs.model  import CrossTable


class TestModel (unittest.TestCase):
    
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
        
        self.T = np.array([
            [2, 0, 3],
            [2, 1, 0],
        ])
        self.sum_r = [5, 3]
        self.sum_s = [4, 1, 3]
        self.total = 8
        self.r = 2 
        self.s = 3
        
        self.X = np.ma.array(self.X, mask=(self.X == 0))
        self.Y = np.ma.array(self.Y, mask=(self.Y == 0))

    def test_compute_table(self):
        mess = 'compute table failed'
        self.table = CrossTable(self.X, self.Y)
        for i in range(self.r):
            self.assertEqual(all(self.table.T[i]), all(self.T[i]), mess)
        self.assertEqual(all(self.table.compute_sum_rows()), all(self.sum_r), mess)   
        self.assertEqual(all(self.table.compute_sum_cols()), all(self.sum_s), mess) 
        self.assertEqual(self.table.n, self.total, mess)  
        r,s = self.table.shape 
        self.assertEqual(r, self.r, mess)
        self.assertEqual(s, self.s, mess)
    
    
if __name__ == "__main__":
    unittest.main()
