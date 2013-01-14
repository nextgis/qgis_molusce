# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest
import math

import numpy as np
from numpy import ma as ma


from molusce.models.correlation.model  import *


class TestModel (unittest.TestCase):
    
    def setUp(self):
        
        self.X = [
            [1, 2, 1,],
            [1, 2, 1,],
            [0, 1, 2,]
        ]

        self.Y = [
            [1, 1, 3,],
            [3, 2, 1,],
            [0, 3, 1,]
        ]
        self.X2 = [
            [1, 2, 1,],
            [1, 2, 1,],
            [0, 1, 2,],
            [0, 1, 2,]
        ]
        X = np.ma.array(self.X, mask=(self.X == 0))
        Y = np.ma.array(self.Y, mask=(self.Y == 0))
        self.T = [
            [1, 0, 0, 0],
            [0, 2, 0, 3],
            [0, 2, 1, 0],
        ]
        self.sum_r = [1, 5, 3]
        self.sum_s = [1, 4, 1, 3]
        self.total = 9
        self.r = 3 
        self.s = 4
        
        self.T_cramer_expect = [
            [1.0/9, 4.0/9 , 1.0/9, 1.0/3 ],
            [5.0/9, 20.0/9, 5.0/9, 15.0/9],
            [1.0/3, 4.0/3 , 1.0/3, 1.0   ]
        ]
    
    def test_size_equals(self):
        self.assertEqual(size_equals(self.X, self.Y), True, 'incorrent size')
        
    def test_Size_no_equals(self):
        self.assertEqual(size_equals(self.X2, self.Y), False, 'sizes equals')
    
    def test_resize(self): 
        self.assertEqual(np.shape(resize(self.X)),(9,) ,'reshape failed')
        self.assertEqual(np.shape(resize(self.X2)),(12,) ,'reshape failed')
        
    def test_correlation(self):
        n = np.shape(self.X)
        lenght = n[0]*n[1]
        mean_x = np.ma.mean(self.X)
        mean_y = np.ma.mean(self.Y)
        self.cov = np.ma.sum(np.multiply(np.subtract(self.X, mean_x), np.subtract(self.Y, mean_y)))/lenght
        self.S_x = np.std(self.X)
        self.S_y = np.std(self.Y)
        self.R=self.cov / (self.S_x * self.S_y)
        
        self.assertEqual(correlation(self.X,self.Y), self.R,'correlation failed')
        
    def test_compute_table(self):
        mess = 'compute table failed'
        self.items = compute_table(self.X, self.Y) 
        for i in range(self.r):
            self.assertEqual(all(self.items[0][i]), all(self.T[i]), mess)
        self.assertEqual(all(self.items[1]), all(self.sum_r), mess)   
        self.assertEqual(all(self.items[2]), all(self.sum_s), mess) 
        self.assertEqual(self.items[3], self.total, mess)   
        self.assertEqual(self.items[4], self.r, mess)
        self.assertEqual(self.items[5], self.s, mess)
          
    def test_cramer(self):
        self.T_cramer = np.subtract(self.T, self.T_cramer_expect)
        self.T_cramer = np.square(self.T_cramer)
        self.x2 = np.sum(np.divide(self.T_cramer, self.T_cramer_expect))
        self.cramer = math.sqrt(self.x2 / (self.total * min(self.r-1,self.s-1)))
        
        self.assertEqual(cramer(self.X, self.Y), self.cramer, 'cramer coeff failed')
        
    def test_jiu(self):
        self.assertAlmostEqual(jiu(self.X, self.Y), 0.584468198124, 9,'joint coeff failed')
        
    
if __name__ == "__main__":
    unittest.main()
