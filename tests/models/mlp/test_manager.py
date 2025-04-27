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
from molusce.algorithms.models.mlp.manager import MlpManager, sigmoid
from molusce.algorithms.models.serializer import (
    ModelParams,
    ModelParamsSerializer,
)


class TestMlpManager(unittest.TestCase):
    def setUp(self):
        self.examples_path = Path(__file__).parents[2] / "examples"
        self.factors = [Raster(self.examples_path / "multifact.tif")]
        self.output = Raster(self.examples_path / "sites.tif")
        # ~ sites.tif is 1-band 3x3 raster:
        # ~ [1,2,1],
        # ~ [1,2,1],
        # ~ [0,1,2]

        self.factors2 = [
            Raster(self.examples_path / "multifact.tif"),
            Raster(self.examples_path / "multifact.tif"),
        ]
        self.factors3 = [Raster(self.examples_path / "two_band.tif")]
        self.factors4 = [
            Raster(self.examples_path / "two_band.tif"),
            Raster(self.examples_path / "multifact.tif"),
        ]

        self.output1 = Raster(self.examples_path / "data.tif")
        self.state1 = self.output1
        self.factors1 = [Raster(self.examples_path / "fact16.tif")]

    def test_MlpManager(self):
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors2, self.output, [10])
        assert_array_equal(mng.getMlpTopology(), [2 * 9 + 2 * 9, 10, 3])

        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        assert_array_equal(mng.getMlpTopology(), [2 * 1 + 1 * 1, 10, 3])

    def test_setTrainingData(self):
        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        stat = self.factors[0].getBandStat(1)  # mean & std
        m, s = stat["mean"], stat["std"]
        mng.setTrainingData(
            self.output, self.factors, self.output, shuffle=False
        )

        minimum, maximum = mng.sigmin, mng.sigmax
        data = np.array(
            [
                ((0, 3), [0, 1], (1.0 - m) / s, [minimum, maximum, minimum]),
                ((1, 3), [0, 0], (1.0 - m) / s, [minimum, minimum, maximum]),
                ((2, 3), [0, 1], (3.0 - m) / s, [minimum, maximum, minimum]),
                ((0, 2), [0, 1], (3.0 - m) / s, [minimum, maximum, minimum]),
                ((1, 2), [0, 0], (2.0 - m) / s, [minimum, minimum, maximum]),
                ((2, 2), [0, 1], (1.0 - m) / s, [minimum, maximum, minimum]),
                ((0, 1), [1, 0], (0.0 - m) / s, [maximum, minimum, minimum]),
                ((1, 1), [0, 1], (3.0 - m) / s, [minimum, maximum, minimum]),
                ((2, 1), [0, 0], (1.0 - m) / s, [minimum, minimum, maximum]),
            ],
            dtype=[
                ("coords", float, 2),
                ("state", float, (2,)),
                ("factors", float, (1,)),
                ("output", float, 3),
            ],
        )
        self.assertEqual(mng.data.shape, (9,))
        for i in range(len(data)):
            assert_array_equal(data[i]["coords"], mng.data[i]["coords"])
            assert_array_equal(data[i]["factors"], mng.data[i]["factors"])
            assert_array_equal(data[i]["output"], mng.data[i]["output"])

        # two input rasters
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors2, self.output, [10])
        mng.setTrainingData(self.output, self.factors2, self.output)
        data = [
            {
                "factors": (
                    np.array(
                        [
                            1.0,
                            1.0,
                            3.0,
                            3.0,
                            2.0,
                            1.0,
                            0.0,
                            3.0,
                            1.0,
                            1.0,
                            1.0,
                            3.0,
                            3.0,
                            2.0,
                            1.0,
                            0.0,
                            3.0,
                            1.0,
                        ]
                    )
                    - stat["mean"]
                )
                / stat["std"],
                "output": np.array([minimum, minimum, maximum]),
                "state": np.array(
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
                ),
            }
        ]
        self.assertEqual(mng.data.shape, (1,))
        assert_array_equal(data[0]["factors"], mng.data[0]["factors"])
        assert_array_equal(data[0]["output"], mng.data[0]["output"])
        assert_array_equal(data[0]["state"], mng.data[0]["state"])

        # Multiband input
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors3, self.output, [10])
        stat1 = self.factors3[0].getBandStat(1)  # mean & std
        m1, s1 = stat1["mean"], stat1["std"]
        stat2 = self.factors3[0].getBandStat(2)  # mean & std
        m2, s2 = stat2["mean"], stat2["std"]
        mng.setTrainingData(self.output, self.factors3, self.output)

        data = [
            {
                "factors": np.array(
                    [
                        (1.0 - m1) / s1,
                        (2.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (2.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (0.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (2.0 - m1) / s1,
                        (1.0 - m2) / s2,
                        (1.0 - m2) / s2,
                        (3.0 - m2) / s2,
                        (3.0 - m2) / s2,
                        (2.0 - m2) / s2,
                        (1.0 - m2) / s2,
                        (0.0 - m2) / s2,
                        (3.0 - m2) / s2,
                        (1.0 - m2) / s2,
                    ]
                ),
                "output": np.array([minimum, minimum, maximum]),
                "state": np.array(
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
                ),
            }
        ]
        self.assertEqual(mng.data.shape, (1,))
        assert_array_equal(data[0]["factors"], mng.data[0]["factors"])
        assert_array_equal(data[0]["output"], mng.data[0]["output"])
        assert_array_equal(data[0]["state"], mng.data[0]["state"])

        # Complex case:
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors4, self.output, [10])
        stat1 = self.factors4[0].getBandStat(1)  # mean & std
        m1, s1 = stat1["mean"], stat1["std"]
        stat2 = self.factors4[0].getBandStat(2)  # mean & std
        m2, s2 = stat2["mean"], stat2["std"]
        stat3 = self.factors4[1].getBandStat(1)
        m3, s3 = stat3["mean"], stat2["std"]

        mng.setTrainingData(self.output, self.factors4, self.output)

        data = [
            {
                "factors": np.array(
                    [
                        (1.0 - m1) / s1,
                        (2.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (2.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (0.0 - m1) / s1,
                        (1.0 - m1) / s1,
                        (2.0 - m1) / s1,
                        (1.0 - m2) / s2,
                        (1.0 - m2) / s2,
                        (3.0 - m2) / s2,
                        (3.0 - m2) / s2,
                        (2.0 - m2) / s2,
                        (1.0 - m2) / s2,
                        (0.0 - m2) / s2,
                        (3.0 - m2) / s2,
                        (1.0 - m2) / s2,
                        (1.0 - m3) / s3,
                        (1.0 - m3) / s3,
                        (3.0 - m3) / s3,
                        (3.0 - m3) / s3,
                        (2.0 - m3) / s3,
                        (1.0 - m3) / s3,
                        (0.0 - m3) / s3,
                        (3.0 - m3) / s3,
                        (1.0 - m3) / s3,
                    ]
                ),
                "output": np.array([minimum, minimum, maximum]),
                "state": np.array(
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
                ),
            }
        ]
        self.assertEqual(mng.data.shape, (1,))
        assert_array_equal(data[0]["factors"], mng.data[0]["factors"])
        assert_array_equal(data[0]["output"], mng.data[0]["output"])
        assert_array_equal(data[0]["state"], mng.data[0]["state"])

    def test_train(self):
        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        mng.setTrainingData(self.output, self.factors, self.output)

        mng.train(1, valPercent=50)
        val = mng.getMinValError()
        mng.train(20, valPercent=50, continue_train=True)
        self.assertGreaterEqual(val, mng.getMinValError())

        mng = MlpManager(ns=1)
        mng.createMlp(self.state1, self.factors1, self.output1, [10])
        mng.setTrainingData(self.state1, self.factors1, self.output1)
        mng.train(1, valPercent=20)
        predict = mng.getPrediction(self.state1, self.factors1)
        mask = predict.getBand(1).mask

        self.assertTrue(not all(mask.flatten()))

    def test_predict(self):
        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        weights = mng.copyWeights()

        # Set weights=0
        # then the output must be sigmoid(0)
        layers = []
        for layer in weights:
            shape = layer.shape
            layers.append(np.zeros(shape))
        mng.setMlpWeights(layers)
        mng._predict(self.output, self.factors)

        # Prediction ( the output must be sigmoid(0) )
        raster = mng.getPrediction(
            self.output, self.factors, calcTransitions=True
        )
        assert_array_equal(raster.getBand(1), sigmoid(0) * np.ones((3, 3)))
        # Confidence is zero
        confid = mng.getConfidence()
        self.assertEqual(confid.getBand(1).dtype, np.uint8)
        assert_array_equal(confid.getBand(1), np.zeros((3, 3)))
        # Transition Potentials (is (sigmoid(0) - sigmin)/sigrange )
        potentials = mng.getTransitionPotentials()
        cats = self.output.getBandGradation(1)
        for cat in cats:
            potential_map = potentials[cat]
            self.assertEqual(potential_map.getBand(1).dtype, np.uint8)
            assert_array_equal(potential_map.getBand(1), 50 * np.ones((3, 3)))

    # Commented while we don't have free rasters to test
    # ~ def test_real(self):
    # ~ #inputs = [Raster('LPB_dem.tif'), Raster('LPB_luc_2007.tif')]
    # ~ inputs = [Raster('LPB_luc_2007.tif'),  Raster('LPB_luc_2007.tif')]
    # ~ output = Raster('LPB_luc_2007.tif')
    # ~ mng = MlpManager()
    # ~ mng.createMlp(output, inputs, output, [10])
    # ~ mng.setTrainingData(output, inputs, output)
    # ~ mng.train(1, valPercent=20)

    @patch("molusce.algorithms.models.serializer.serializer.pluginMetadata")
    @patch("molusce.algorithms.models.serializer.serializer.datetime")
    def test_serialization(
        self, datetime_mock: MagicMock, metadata_mock: MagicMock
    ) -> None:
        MOLUSCE_VERSION = "5.0.0"
        metadata_mock.return_value = MOLUSCE_VERSION

        FIXED_DATETIME = datetime(2006, 5, 4, 3, 2, 1)
        datetime_mock.now.return_value = FIXED_DATETIME

        mlp_manager = MlpManager()
        mlp_manager.createMlp(self.output, self.factors, self.output, [10])
        mlp_manager.setTrainingData(self.output, self.factors, self.output)
        mlp_manager.train(1, valPercent=50)

        factors = {str(uuid4()): factor for factor in self.factors}
        model_params = ModelParams.from_data(mlp_manager, self.output, factors)

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
            == "Artificial Neural Network (Multi-layer Perceptron)"
        )
        self.assertTrue(isinstance(loaded_params.model, MlpManager))
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
        loaded_model = cast("MlpManager", loaded_params.model)
        assert_array_equal(
            loaded_model.getMlpTopology(), [2 * 1 + 1 * 1, 10, 3]
        )


if __name__ == "__main__":
    unittest.main()
