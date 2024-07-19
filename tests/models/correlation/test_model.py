import math
import unittest

import numpy as np
from numpy import ma as ma

from molusce.algorithms.models.correlation.model import DependenceCoef


class TestModel (unittest.TestCase):

    def setUp(self):

        self.X = np.array([
            [1, 2, 1],
            [1, 2, 1],
            [0, 1, 2]
        ])

        self.Y = np.array([
            [1, 1, 3],
            [3, 2, 1],
            [0, 3, 1]
        ])
        self.Y1 = np.array([
            [2, 1, 1],
            [1, 2, 2],
            [0, 3, 3]
        ])
        self.X2 = np.array([
            [1, 2, 1],
            [1, 2, 1],
            [0, 1, 2],
            [0, 1, 2]
        ])
        """
        self.T = np.array([
            [1, 0, 0, 0],
            [0, 2, 0, 3],
            [0, 2, 1, 0],
        ])"""
        self.T = np.array([
            [2, 0, 3],
            [2, 1, 0],
        ])
        self.sum_r = [5, 3]
        self.sum_s = [4, 1, 3]
        self.total = 8
        self.r = 2
        self.s = 3

        self.T_cramer_expect = np.array([
            [20.0/8, 5.0/8, 15.0/8],
            [12.0/8, 3.0/8,  9.0/8]
        ])
        self.X = np.ma.array(self.X, mask=(self.X == 0))
        self.Y = np.ma.array(self.Y, mask=(self.Y == 0))
        self.Y1 = np.ma.array(self.Y1, mask=(self.Y1 == 0))
        self.combo_mask = np.array([
            [False, False, False],
            [False, False, False],
            [True , False, False]
        ])

    def test_correlation(self):
        dc = DependenceCoef(self.X, self.Y)
        n = len(np.ma.compressed(self.X))
        mean_x = np.ma.mean(self.X)
        mean_y = np.ma.mean(self.Y)
        self.cov = np.ma.sum(np.multiply(np.subtract(self.X, mean_x), np.subtract(self.Y, mean_y)))/n
        self.S_x = np.std(self.X)
        self.S_y = np.std(self.Y)
        self.R = self.cov / (self.S_x * self.S_y)
        self.assertTrue(np.isclose(dc.correlation(), self.R),"correlation failed")

        dc = DependenceCoef(self.X, self.X)
        self.assertTrue(np.isclose(dc.correlation(), 1.0),"correlation failed")

    def test_correctness(self):
        dc = DependenceCoef(self.X, self.X)
        self.assertTrue(np.isclose(dc.correctness(), 100.0), "correctness % failed")

    def test_cramer(self):
        dc = DependenceCoef(self.X, self.Y)
        self.T_cramer = np.subtract(self.T, self.T_cramer_expect)
        self.T_cramer = np.square(self.T_cramer)
        self.x2 = np.sum(np.divide(self.T_cramer, self.T_cramer_expect))
        self.cramer = math.sqrt(self.x2 / (self.total * min(self.r-1,self.s-1)))
        self.assertEqual(dc.cramer(), self.cramer, "cramer coeff failed")

        dc = DependenceCoef(self.X, self.X)
        self.assertEqual(dc.cramer(), 1.0, "cramer coeff failed")

    def test_jiu(self):
        dc = DependenceCoef(self.X, self.Y)
        self.assertAlmostEqual(dc.jiu(), 0.385101639127, 9, "joint coeff failed")
        dc = DependenceCoef(self.X, self.X)
        self.assertEqual(dc.jiu(), 1.0, "joint coeff failed")

    def test_kappa(self):
        dc = DependenceCoef(self.Y, self.Y1)
        #~ table =  np.array([
            #~ [1, 2, 1],
            #~ [0, 1, 0],
            #~ [2, 0, 1],
        #~ ])
        Pa = 3.0/8
        Pe = 21.0/64
        Pmax = 6.0/8

        answer = (Pa - Pe)/(1 - Pe)
        self.assertEqual(dc.kappa(), answer)

        answer = (Pa - Pe)/(Pmax - Pe)
        self.assertEqual(dc.kappa(mode="loc"), answer)

        answer = (Pmax - Pe)/(1 - Pe)
        self.assertEqual(dc.kappa(mode="histo"), answer)


if __name__ == "__main__":
    unittest.main()
