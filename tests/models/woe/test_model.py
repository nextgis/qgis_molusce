# QGIS MOLUSCE Plugin
# Copyright (C) 2026  NextGIS
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <https://www.gnu.org/licenses/>.

import math
import unittest

import numpy as np
from numpy import ma as ma

from molusce.algorithms.models.woe.model import (
    EPSILON,
    WoeError,
    _binary_woe,
    woe,
)


class TestModel(unittest.TestCase):
    def setUp(self):
        fact = [[True, True, False], [False, False, True], [None, False, True]]
        site = [
            [False, True, False],
            [False, True, False],
            [False, False, True],
        ]
        site1 = [[1, 2, 1], [1, 2, 1], [0, 1, 2]]
        zero = [
            [False, False, False],
            [False, False, False],
            [None, False, False],
        ]

        self.mask = [
            [False, False, False],
            [False, False, False],
            [True, False, False],
        ]
        self.mask1 = [
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]
        multifact = [[1, 1, 3], [3, 2, 1], [0, 3, 1]]

        bigfact = [
            [True, True, False, True, True, False],
            [False, False, True, False, False, True],
            [None, False, True, None, False, True],
            [True, False, True, None, False, True],
        ]
        bigsite = [
            [False, True, False, False, True, False],
            [False, True, False, False, False, True],
            [None, False, False, None, True, True],
            [False, False, True, None, False, False],
        ]
        self.bigmask = [
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [True, False, False, True, False, False],
            [False, False, False, True, False, False],
        ]

        self.factor = ma.array(data=fact, mask=self.mask, dtype=bool)
        self.fact1 = ma.array(data=fact, mask=self.mask1, dtype=bool)
        self.multifact = ma.array(data=multifact, mask=self.mask, dtype=int)
        self.sites = ma.array(data=site, mask=self.mask, dtype=bool)
        self.sites1 = ma.array(data=site1, mask=self.mask1, dtype=int)
        self.sites2 = ma.array(data=site1, mask=self.mask, dtype=int)
        self.zero = ma.array(data=zero, mask=self.mask, dtype=bool)
        self.bigfactor = ma.array(data=bigfact, mask=self.bigmask, dtype=bool)
        self.bigsite = ma.array(data=bigsite, mask=self.bigmask, dtype=bool)

    def test_binary_woe(self):
        wPlus = math.log((2.0 / 3 + EPSILON) / (2.0 / 5 + EPSILON))
        wMinus = math.log((1.0 / 3 + EPSILON) / (3.0 / 5 + EPSILON))
        self.assertEqual(_binary_woe(self.factor, self.sites), (wPlus, wMinus))

        wPlus = math.log((5.0 / 7 + EPSILON) / (0.5 / 3.5 + EPSILON))
        wMinus = math.log((2.0 / 7 + EPSILON) / (3.0 / 3.5 + EPSILON))
        self.assertEqual(
            _binary_woe(self.bigfactor, self.bigsite, unitcell=2),
            (wPlus, wMinus),
        )

        # if Sites=Factor:
        wPlus = math.log((1 + EPSILON) / EPSILON)
        wMinus = math.log(EPSILON / (1 + EPSILON))
        self.assertEqual(
            _binary_woe(self.factor, self.factor), (wPlus, wMinus)
        )

        # Check areas size
        self.assertRaises(WoeError, _binary_woe, self.factor, self.zero)
        self.assertRaises(WoeError, _binary_woe, self.zero, self.sites)
        self.assertRaises(
            WoeError, _binary_woe, self.bigfactor, self.bigsite, 3
        )

        # Non-binary sites
        self.assertRaises(WoeError, woe, self.fact1, self.sites1)
        # Assert does not raises
        woe(self.multifact, self.sites2)

    def test_woe(self):
        wPlus1 = math.log((2.0 / 3 + EPSILON) / (2.0 / 5 + EPSILON))
        wMinus1 = math.log((1.0 / 3 + EPSILON) / (3.0 / 5 + EPSILON))

        wPlus2 = math.log((1.0 / 3 + EPSILON) / (EPSILON))
        wMinus2 = math.log((2.0 / 3 + EPSILON) / (1.0 + EPSILON))

        wPlus3 = math.log((EPSILON) / (3.0 / 5 + EPSILON))
        wMinus3 = math.log((1.0 + EPSILON) / (2.0 / 5 + EPSILON))

        # Binary categories
        ans = [
            [wPlus1, wPlus1, wMinus1],
            [wMinus1, wMinus1, wPlus1],
            [None, wMinus1, wPlus1],
        ]
        ans = ma.array(data=ans, mask=self.mask)
        w = woe(self.factor, self.sites)
        np.testing.assert_equal(w["map"], ans)

        # Multiclass
        w1, w2, w3 = (
            (wPlus1 + wMinus2 + wMinus3),
            (wPlus2 + wMinus1 + wMinus3),
            (wPlus3 + wMinus1 + wMinus2),
        )
        ans = [[w1, w1, w3], [w3, w2, w1], [0, w3, w1]]
        ans = ma.array(data=ans, mask=self.mask)
        weights = woe(self.multifact, self.sites)

        np.testing.assert_equal(ans, weights["map"])


if __name__ == "__main__":
    unittest.main()
