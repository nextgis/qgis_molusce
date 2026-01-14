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

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.algorithms.utils import (
    binaryzation,
    masks_identity,
    reclass,
    sizes_equal,
)


class TestRaster(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2, 1], [1, 2, 1], [0, 1, 2]])

        self.Y = np.array([[1, 1, 3], [3, 2, 1], [0, 3, 1]])
        self.X2 = np.array([[1, 2, 1], [1, 3, 1], [0, 1, 4], [0, 1, 7]])
        self.X = np.ma.array(self.X, mask=(self.X == 0))
        self.Y = np.ma.array(self.Y, mask=(self.Y == 0))
        self.combo_mask = np.array(
            [
                [False, False, False],
                [False, False, False],
                [True, False, False],
            ]
        )

    def test_binarization(self):
        t = binaryzation(self.X, [0, 2])
        answer1 = np.array(
            [[False, True, False], [False, True, False], [True, False, True]]
        )
        answer2 = np.array(
            [[False, True, False], [False, True, False], [True, False, True]]
        )
        assert_array_equal(t, answer1)
        assert_array_equal(t, answer2)

        mask = [
            [False, False, False],
            [False, False, False],
            [True, False, False],
        ]
        data = [
            [False, True, False],
            [False, True, False],
            [False, False, True],
        ]
        assert_array_equal(
            binaryzation(np.array(data), [True]),
            np.ma.array(data=data, mask=mask),
        )

        data = np.ma.array(data=data, mask=mask)
        assert_array_equal(
            binaryzation(data, [True]), np.ma.array(data=data, mask=mask)
        )

    def test_masks_identity(self):
        self.X, self.Y = masks_identity(self.X, self.Y)
        mask_x = np.matrix.flatten(self.X.mask)
        self.combo_mask = np.matrix.flatten(self.combo_mask)
        k = all(np.equal(mask_x, self.combo_mask))
        self.assertEqual(k, True, "masks_identify failed")

    def test_size_equals(self):
        self.assertEqual(sizes_equal(self.X, self.Y), True, "incorrent size")

    def test_Size_no_equals(self):
        self.assertEqual(
            sizes_equal(self.X2, self.Y), False, "sizes are equal"
        )

    def test_reclass(self):
        X = reclass(self.X2, [1.1, 3.1, 4])
        answer = np.array([[1, 2, 1], [1, 2, 1], [1, 1, 4], [1, 1, 4]])
        assert_array_equal(X, answer)


if __name__ == "__main__":
    unittest.main()
