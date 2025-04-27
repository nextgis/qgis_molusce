import unittest
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.lr.lr import LR
from molusce.algorithms.models.serializer.serializer import (
    ModelParams,
    ModelParamsSerializer,
)


class TestLRManager(unittest.TestCase):
    def setUp(self):
        self.examples_path = Path(__file__).parents[2] / "examples"
        self.output = Raster(self.examples_path / "multifact.tif")
        # ~ [1,1,3]
        # ~ [3,2,1]
        # ~ [0,3,1]

        self.output.resetMask([0])
        self.state = self.output
        self.factors = [
            Raster(self.examples_path / "sites.tif"),
            Raster(self.examples_path / "sites.tif"),
        ]
        # ~ [1,2,1],
        # ~ [1,2,1],
        # ~ [0,1,2]

        self.output1 = Raster(self.examples_path / "data.tif")
        self.state1 = self.output1
        self.factors1 = [Raster(self.examples_path / "fact16.tif")]

    def test_LR(self):
        # ~ data = [
        # ~ [3.0, 1.0, 3.0],
        # ~ [3.0, 1.0, 3.0],
        # ~ [0,   3.0, 1.0]
        # ~ ]
        # ~ result = np.ma.array(data = data, mask = (data==0))

        lr = LR(ns=0)  # 3-class problem
        lr.setState(self.state)
        lr.setFactors(self.factors)
        lr.setOutput(self.output)
        lr.setTrainingData()
        lr.train()
        predict = lr.getPrediction(self.state, self.factors)
        predict = predict.getBand(1)
        assert_array_equal(predict, self.output.getBand(1))

        lr = LR(ns=1)  # Two-class problem (it's because of boundary effect)
        lr.setState(self.state1)
        lr.setFactors(self.factors1)
        lr.setOutput(self.output1)
        lr.setTrainingData()
        lr.train()
        predict = lr.getPrediction(
            self.state1, self.factors1, calcTransitions=True
        )
        predict = predict.getBand(1)
        self.assertEqual(predict.dtype, np.uint8)
        data = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        result = np.ma.array(data=data, mask=(data == 0))
        assert_array_equal(predict, result)

        # Confidence is zero
        confid = lr.getConfidence()
        self.assertEqual(confid.getBand(1).dtype, np.uint8)

        # Transition Potentials
        potentials = lr.getTransitionPotentials()
        # cats = self.output.getBandGradation(1)
        for cat in [1.0, 2.0]:
            potential_map = potentials[cat]
            self.assertEqual(potential_map.getBand(1).dtype, np.uint8)

    @patch("molusce.algorithms.models.serializer.serializer.pluginMetadata")
    @patch("molusce.algorithms.models.serializer.serializer.datetime")
    def test_serialization(
        self, datetime_mock: MagicMock, metadata_mock: MagicMock
    ) -> None:
        MOLUSCE_VERSION = "5.0.0"
        metadata_mock.return_value = MOLUSCE_VERSION

        FIXED_DATETIME = datetime(2006, 5, 4, 3, 2, 1)
        datetime_mock.now.return_value = FIXED_DATETIME

        lr = LR(ns=0)  # 3-class problem
        lr.setState(self.state)
        lr.setFactors(self.factors)
        lr.setOutput(self.output)
        lr.setTrainingData()
        lr.train()

        factors = {str(uuid4()): factor for factor in self.factors}
        model_params = ModelParams.from_data(lr, self.output, factors)

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
            == "Logistic Regression"
        )
        self.assertTrue(isinstance(loaded_params.model, LR))
        self.assertTrue(
            (model_params.base_xsize, model_params.base_ysize)
            == (loaded_params.base_xsize, loaded_params.base_ysize)
            == (self.output.getXSize(), self.output.getYSize())
        )
        self.assertTrue(
            model_params.base_classes
            == loaded_params.base_classes
            == self.output.getUniqueValues()
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
        loaded_model = cast("LR", loaded_params.model)
        predict = loaded_model.getPrediction(self.state, self.factors)
        predict = predict.getBand(1)
        assert_array_equal(predict, self.output.getBand(1))


if __name__ == "__main__":
    unittest.main()
