import unittest

import numpy as np
from numpy import ma as ma

from molusce.algorithms.models.crosstabs.model import CrossTable


class TestCrossTable (unittest.TestCase):

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

    def test_init(self):
        mess = 'compute table failed'
        table = CrossTable(self.X, self.Y)
        crTab = table.getCrosstable()
        for i in range(self.r):
            self.assertEqual(all(crTab[i]), all(self.T[i]), mess)
        self.assertEqual(all(table.getSumRows()), all(self.sum_r), mess)
        self.assertEqual(all(table.getSumCols()), all(self.sum_s), mess)
        self.assertEqual(table.n, self.total, mess)
        r,s = table.shape
        self.assertEqual(r, self.r, mess)
        self.assertEqual(s, self.s, mess)

        x = np.ma.array([1,2,3,4,5,5])
        y = np.ma.array([1,2,3,3,5,6])

        table = CrossTable(x,y, expand=True)
        for i in [1,2,3,4,5,6]:
            self.assertIn(i, table.graduation_x)
            self.assertIn(i, table.graduation_y)

    def test_getTransition(self):
        self.table = CrossTable(self.X, self.Y)

        fromClass, toClass = 1, 2
        self.assertEqual(self.table.getTransition(fromClass, toClass), 0)
        fromClass, toClass = 1, 3
        self.assertEqual(self.table.getTransition(fromClass, toClass), 3)

    def test_expectedTable(self):
        # CrossTable:
        # [2, 0, 3],
        # [2, 1, 0],

        table = CrossTable(self.X, self.Y)
        tab = table.getExpectedTable()
        answer = [
            [20/8.0, 5/8.0, 15/8.0],
            [12/8.0, 3/8.0,  9/8.0]
        ]
        np.testing.assert_array_equal(answer, tab)

        tab = table.getExpectedProbtable()
        answer = [
            [20/64.0, 5/64.0, 15/64.0],
            [12/64.0, 3/64.0,  9/64.0]
        ]
        np.testing.assert_array_equal(answer, tab)
        np.testing.assert_array_equal(table.getProbCols(), [4.0/8, 1.0/8, 3.0/8])
        np.testing.assert_array_equal(table.getProbRows(), [5.0/8, 3.0/8])


if __name__ == "__main__":
    unittest.main()
