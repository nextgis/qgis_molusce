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
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
from numpy import ma as ma
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst
from molusce.algorithms.models.serializer.serializer import (
    ModelParams,
    ModelParamsSerializer,
)
from molusce.algorithms.models.woe.manager import WoeManager
from molusce.algorithms.models.woe.model import woe


class TestWoEManager(unittest.TestCase):
    def setUp(self):
        self.examples_path = Path(__file__).parents[2] / "examples"
        self.factor = Raster(self.examples_path / "multifact.tif")
        # ~ [1,1,3]
        # ~ [3,2,1]
        # ~ [0,3,1]

        self.sites = Raster(self.examples_path / "sites.tif")
        # ~ [1,2,1],
        # ~ [1,2,1],
        # ~ [0,1,2]
        self.sites.resetMask(maskVals=[0])

        self.mask = [
            [False, False, False],
            [False, False, False],
            [True, False, False],
        ]
        fact = [[1, 1, 3], [3, 2, 1], [0, 3, 1]]
        site = [
            [False, True, False],
            [False, True, False],
            [False, False, True],
        ]
        self.factraster = ma.array(data=fact, mask=self.mask, dtype=int)
        self.sitesraster = ma.array(data=site, mask=self.mask, dtype=bool)

    def test_CheckBins(self):
        aa = AreaAnalyst(self.sites, self.sites)
        w1 = WoeManager([self.factor], aa, bins=None)
        self.assertTrue(w1.checkBins())
        w1 = WoeManager([self.factor], aa, bins={0: [None]})
        self.assertTrue(w1.checkBins())
        w1 = WoeManager([self.factor], aa, bins={0: [[1, 2, 3]]})
        self.assertTrue(w1.checkBins())
        w1 = WoeManager([self.factor], aa, bins={0: [[1, 4]]})
        self.assertFalse(w1.checkBins())
        w1 = WoeManager([self.factor], aa, bins={0: [[-1, 1]]})
        self.assertFalse(w1.checkBins())
        w1 = WoeManager([self.factor], aa, bins={0: [[2, 3, 1]]})
        self.assertFalse(w1.checkBins())

    def test_WoeManager(self):
        aa = AreaAnalyst(self.sites, self.sites)
        w1 = WoeManager([self.factor], aa)
        w1.train()
        p = w1.getPrediction(self.sites).getBand(1)
        answer = [[0, 3, 0], [0, 3, 0], [9, 0, 3]]
        answer = ma.array(data=answer, mask=self.mask)
        assert_array_equal(p, answer)

        initState = Raster(self.examples_path / "data.tif")
        # ~ [1,1,1,1],
        # ~ [1,1,2,2],
        # ~ [2,2,2,2],
        # ~ [3,3,3,3]
        finalState = Raster(self.examples_path / "data1.tif")
        # ~ [1,1,2,3],
        # ~ [3,1,2,3],
        # ~ [3,3,3,3],
        # ~ [1,1,3,2]
        aa = AreaAnalyst(initState, finalState)
        w = WoeManager([initState], aa)
        w.train()
        # print w.woe
        p = w.getPrediction(initState).getBand(1)
        self.assertEqual(p.dtype, np.uint8)

        # Calculate by hands:
        # 1->1 transition raster:
        r11 = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # 1->2 raster:
        r12 = [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # 1->3 raster:
        r13 = [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # 2->1
        r21 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # 2->2
        r22 = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # 2->3
        r23 = [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        # 3->1
        r31 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
        # 3->2
        r32 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
        # 3->3
        r33 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]
        geodata = initState.getGeodata()
        sites = {
            "11": r11,
            "12": r12,
            "13": r13,
            "21": r21,
            "22": r22,
            "23": r23,
            "31": r31,
            "32": r32,
            "33": r33,
        }
        woeDict = {}  # WoE of transitions
        for k in sites:
            if k != "21":  # !!! r21 is zero
                x = Raster()
                x.create([np.ma.array(data=sites[k])], geodata)
                sites[k] = x
                woeDict[k] = woe(initState.getBand(1), x.getBand(1))
        # w1max = np.maximum(woeDict['11'], woeDict['12'], woeDict['13'])
        # w2max = np.maximum(woeDict['22'], woeDict['23'])
        # w3max = np.maximum(woeDict['31'], woeDict['32'], woeDict['33'])
        # Answer is a transition code with max weight
        answer = [[0, 0, 0, 0], [0, 0, 5, 5], [5, 5, 5, 5], [6, 6, 6, 6]]
        assert_array_equal(p, answer)

        w = WoeManager([initState], aa, bins={0: [[2]]})
        w.train()
        p = w.getPrediction(initState).getBand(1)
        self.assertEqual(p.dtype, np.uint8)
        c = w.getConfidence().getBand(1)
        self.assertEqual(c.dtype, np.uint8)

    @patch("molusce.algorithms.models.serializer.serializer.pluginMetadata")
    @patch("molusce.algorithms.models.serializer.serializer.datetime")
    def test_serialization(
        self, datetime_mock: MagicMock, metadata_mock: MagicMock
    ) -> None:
        MOLUSCE_VERSION = "5.0.0"
        metadata_mock.return_value = MOLUSCE_VERSION

        FIXED_DATETIME = datetime(2006, 5, 4, 3, 2, 1)
        datetime_mock.now.return_value = FIXED_DATETIME

        area_analyst = AreaAnalyst(self.sites, self.sites)
        woe_manager = WoeManager([self.factor], area_analyst)
        woe_manager.train()

        factors = {str(uuid4()): factor for factor in [self.factor]}
        model_params = ModelParams.from_data(woe_manager, self.sites, factors)

        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path_str = temp_file.name
        try:
            ModelParamsSerializer.to_file(model_params, temp_file_path_str)
            loaded_params = ModelParamsSerializer.from_file(temp_file_path_str)
        finally:
            temp_file_path = Path(temp_file_path_str)
            if temp_file_path.exists():
                temp_file_path.unlink()

        self.assertTrue(
            model_params.model_type
            == loaded_params.model_type
            == "Weights of Evidence"
        )
        self.assertTrue(isinstance(loaded_params.model, WoeManager))
        self.assertTrue(
            (model_params.base_xsize, model_params.base_xsize)
            == (loaded_params.base_xsize, loaded_params.base_xsize)
            == (self.sites.getXSize(), self.sites.getYSize())
        )
        self.assertTrue(
            model_params.base_classes
            == loaded_params.base_classes
            == self.sites.getUniqueValues()
        )
        self.assertTrue(
            model_params.factors_metadata
            == loaded_params.factors_metadata
            == [
                {"name": uuid, "bandcount": bandcount.bandcount}
                for uuid, bandcount in factors.items()
            ]
        )
        self.assertTrue(
            model_params.molusce_version
            == loaded_params.molusce_version
            == MOLUSCE_VERSION
        )
        self.assertTrue(
            model_params.creation_ts
            == loaded_params.creation_ts
            == FIXED_DATETIME
        )

        # Check model
        # loaded_model = cast("WoeManager", loaded_params.model)
        # prediction = loaded_model.getPrediction(self.sites).getBand(1)
        # answer = [[0, 3, 0], [0, 3, 0], [9, 0, 3]]
        # answer = ma.array(data=answer, mask=self.mask)
        # assert_array_equal(prediction, answer)


if __name__ == "__main__":
    unittest.main()
