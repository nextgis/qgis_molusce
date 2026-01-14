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
from pathlib import Path

import numpy as np
from numpy import ma as ma

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.errorbudget.ebmodel import EBudget, weightedSum


class TestModel(unittest.TestCase):
    def setUp(self):
        self.examples_path = Path(__file__).parents[2] / "examples"
        self.reference = Raster(self.examples_path / "data.tif")
        # ~ [1,1,1,1],
        # ~ [1,1,2,2],
        # ~ [2,2,2,2],
        # ~ [3,3,3,3]

        self.simulated = Raster(self.examples_path / "data1.tif")
        # ~ [1,1,2,3],
        # ~ [3,1,2,3],
        # ~ [3,3,3,3],
        # ~ [1,1,3,2]

    def test_Init(self):
        Rj = {
            1: np.ma.array(
                data=[
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                mask=[
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ),
            2: np.ma.array(
                data=[
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                mask=[
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ),
            3: np.ma.array(
                data=[
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                mask=[
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ),
        }

        eb = EBudget(self.reference, self.simulated)
        np.testing.assert_equal(eb.Rj, Rj)

    def test_weightedSum(self):
        eb = EBudget(self.reference, self.simulated)
        W = eb.W

        S1 = weightedSum(eb.Sj[1], W)
        S2 = weightedSum(eb.Sj[2], W)
        S3 = weightedSum(eb.Sj[3], W)
        self.assertEqual(S1, 5.0 / 16)
        self.assertEqual(S2, 3.0 / 16)
        self.assertEqual(S3, 8.0 / 16)

    def test_NoNo(self):
        eb = EBudget(self.reference, self.simulated)

        noNo = eb.NoNo()
        answer = 1.0 / 3
        np.testing.assert_almost_equal(noNo, answer)

    def test_NoMed(self):
        eb = EBudget(self.reference, self.simulated)

        noM = eb.NoMed()
        answer = (6.0 * 5 / 16 + 6.0 * 3 / 16 + 4.0 * 8 / 16) / 16
        np.testing.assert_almost_equal(noM, answer)

    def test_MedMed(self):
        eb = EBudget(self.reference, self.simulated)
        medM = eb.MedMed()
        answer = 5.0 / 16
        np.testing.assert_almost_equal(medM, answer)

    def test_MedPer(self):
        eb = EBudget(self.reference, self.simulated)
        medP = eb.MedPer()
        answer = (
            min(6.0 / 16, 5.0 / 16)
            + min(6.0 / 16, 3.0 / 16)
            + min(4.0 / 16, 8.0 / 16)
        )
        np.testing.assert_almost_equal(medP, answer)

    def test_Mask(self):
        reference = Raster(self.examples_path / "data.tif")
        simulated = Raster(self.examples_path / "data1.tif")
        reference.resetMask([2])
        simulated.resetMask([2])

        eb = EBudget(reference, simulated)
        W = eb.W

        S1 = weightedSum(eb.Sj[1], W)
        S3 = weightedSum(eb.Sj[3], W)
        np.testing.assert_almost_equal(S1, 5.0 / 8)
        np.testing.assert_almost_equal(S3, 3.0 / 8)

        noNo = eb.NoNo()
        np.testing.assert_almost_equal(noNo, 0.5)

        noM = eb.NoMed()
        np.testing.assert_almost_equal(noM, (5.0 * 5 / 8 + 3.0 * 3 / 8) / 8)

        medM = eb.MedMed()
        np.testing.assert_almost_equal(medM, 4.0 / 8)

        medP = eb.MedPer()
        np.testing.assert_almost_equal(medP, 1.0)

    def test_coarse(self):
        reference = Raster(self.examples_path / "data.tif")
        simulated = Raster(self.examples_path / "data1.tif")
        reference.resetMask([2])
        simulated.resetMask([2])

        eb = EBudget(reference, simulated)
        eb.coarse(2)
        # W
        answer = np.array([[1.0, 0.25], [0.5, 0.25]])
        np.testing.assert_array_equal(eb.W, answer)
        # Rj
        answer1 = np.array([[1.0, 1.0], [0, 0]])
        answer3 = np.array([[0, 0], [1.0, 1.0]])
        ans = {1: answer1, 3: answer3}
        np.testing.assert_equal(eb.Rj, ans)
        # Sj
        answer1 = np.array([[3.0 / 4, 0.0], [1.0, 0]])
        answer3 = np.array([[1.0 / 4, 1.0], [0, 1.0]])
        ans = {1: answer1, 3: answer3}
        np.testing.assert_equal(eb.Sj, ans)

        eb.coarse(2)
        # W
        answer = np.array([[0.5]])
        np.testing.assert_array_equal(eb.W, answer)
        # Rj
        answer1 = np.array([[(1 + 1.0 / 4) / 2]])
        answer3 = np.array([[(1.0 / 2 + 1.0 / 4) / 2]])
        ans = {1: answer1, 3: answer3}
        np.testing.assert_equal(eb.Rj, ans)
        # Sj
        answer1 = np.array([[(3.0 / 4 + 0.5) / 2]])
        answer3 = np.array([[(1.0 / 4 + 1.0 / 4 + 1.0 / 4) / 2]])
        ans = {1: answer1, 3: answer3}
        np.testing.assert_equal(eb.Sj, ans)

    def test_getStat(self):
        reference = Raster(self.examples_path / "data.tif")
        simulated = Raster(self.examples_path / "data1.tif")
        reference.resetMask([2])
        simulated.resetMask([2])

        eb = EBudget(reference, simulated)
        stat = eb.getStat(nIter=3)
        ans0 = {
            "NoNo": 0.5,
            "NoMed": (5.0 * 5 / 8 + 3.0 * 3 / 8) / 8,
            "MedMed": 4.0 / 8,
            "MedPer": 1.0,
            "PerPer": 1.0,
        }
        for k in list(stat[0].keys()):
            np.testing.assert_almost_equal(stat[0][k], ans0[k])
        ans1 = {
            "NoNo": 0.5,
            "NoMed": (5.0 / 8 + 5.0 / 32 + 3.0 / 16 + 3.0 / 32) / 2,
            "MedMed": 4.0 / 8,
            "MedPer": 1.0,
            "PerPer": 1.0,
        }
        for k in list(stat[1].keys()):
            np.testing.assert_almost_equal(stat[0][k], ans1[k])


if __name__ == "__main__":
    unittest.main()
