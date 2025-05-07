import unittest
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
from numpy.testing import assert_almost_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst
from molusce.algorithms.models.mce.mce import MCE
from molusce.algorithms.models.serializer.serializer import (
    ModelParams,
    ModelParamsSerializer,
)


class TestMCE(unittest.TestCase):
    def setUp(self):
        self.examples_path = Path(__file__).parents[2] / "examples"
        self.factor = Raster(self.examples_path / "multifact.tif")
        # ~ [1,1,3]
        # ~ [3,2,1]
        # ~ [0,3,1]

        self.state = Raster(self.examples_path / "sites.tif")
        self.state.resetMask(maskVals=[0])
        # ~ [1,2,1],
        # ~ [1,2,1],
        # ~ [0,1,2]

        self.areaAnalyst = AreaAnalyst(self.state, second=None)

    def test_MCE(self):
        data = [
            [1.0, 4.0, 6.0, 7.0],
            [1.0 / 4, 1.0, 3.0, 4.0],
            [1.0 / 6, 1.0 / 3, 1.0, 2.0],
            [1.0 / 7, 1.0 / 4, 1.0 / 2, 1],
        ]
        # Multiband
        factor = Raster(self.examples_path / "two_band.tif")

        mce = MCE(
            [self.factor, factor, self.factor], data, 1, 2, self.areaAnalyst
        )
        w = mce.getWeights()
        answer = [0.61682294, 0.22382863, 0.09723423, 0.06211421]
        assert_almost_equal(w, answer)

        # One-band
        mce = MCE(
            [self.factor, self.factor, self.factor, self.factor],
            data,
            1,
            2,
            self.areaAnalyst,
        )
        w = mce.getWeights()
        answer = [0.61682294, 0.22382863, 0.09723423, 0.06211421]
        assert_almost_equal(w, answer)

        mask = [
            [False, False, False],
            [False, False, False],
            [False, False, True],
        ]
        p = mce.getPrediction(self.state).getBand(1)
        self.assertEqual(p.dtype, np.uint8)
        answer = [  # The locations where the big numbers are stored must be masked (see mask and self.state)
            [1, 3, 1],
            [1, 3, 1],
            [100, 1, 100],
        ]
        answer = np.ma.array(data=answer, mask=mask)
        assert_almost_equal(p, answer)

        c = mce.getConfidence().getBand(1)
        self.assertEqual(c.dtype, np.uint8)
        answer = [  # The locations where the big numbers are stored must be masked (see mask and self.state)
            [sum((w * 100).astype(int)) // 3, 0, sum((w * 100).astype(int))],
            [sum((w * 100).astype(int)), 0, sum((w * 100).astype(int)) // 3],
            [10000, sum((w * 100).astype(int)), 10000],
        ]
        answer = np.ma.array(data=answer, mask=mask)
        assert_almost_equal(c, answer)

    @patch("molusce.algorithms.models.serializer.serializer.pluginMetadata")
    @patch("molusce.algorithms.models.serializer.serializer.datetime")
    def test_serialization(
        self, datetime_mock: MagicMock, metadata_mock: MagicMock
    ) -> None:
        MOLUSCE_VERSION = "5.0.0"
        metadata_mock.return_value = MOLUSCE_VERSION

        FIXED_DATETIME = datetime(2006, 5, 4, 3, 2, 1)
        datetime_mock.now.return_value = FIXED_DATETIME

        data = [
            [1.0, 4.0, 6.0, 7.0],
            [1.0 / 4, 1.0, 3.0, 4.0],
            [1.0 / 6, 1.0 / 3, 1.0, 2.0],
            [1.0 / 7, 1.0 / 4, 1.0 / 2, 1],
        ]
        mce = MCE(
            [self.factor, self.factor, self.factor, self.factor],
            data,
            1,
            2,
            self.areaAnalyst,
        )

        factors = {str(uuid4()): factor for factor in [self.factor]}
        model_params = ModelParams.from_data(mce, self.state, factors)

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
            == "Multi Criteria Evaluation"
        )
        self.assertTrue(isinstance(loaded_params.model, MCE))
        self.assertTrue(
            (model_params.base_xsize, model_params.base_ysize)
            == (loaded_params.base_xsize, loaded_params.base_ysize)
            == (self.state.getXSize(), self.state.getYSize())
        )
        self.assertTrue(
            model_params.base_classes
            == loaded_params.base_classes
            == self.state.getUniqueValues()
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
        weights = cast("MCE", loaded_params.model).getWeights()
        answer = [0.61682294, 0.22382863, 0.09723423, 0.06211421]
        assert_almost_equal(weights, answer)


if __name__ == "__main__":
    unittest.main()
