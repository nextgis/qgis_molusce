import unittest
from pathlib import Path

import numpy as np
from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.crosstabs.manager import CrossTableManager
from numpy import ma as ma


class TestCrossTableManager(unittest.TestCase):
    def setUp(self):
        self.examples_path = Path(__file__).parents[2] / "examples"
        self.init = Raster(
            self.examples_path / "init.tif", maskVals={1: [255]}
        )
        self.final = Raster(
            self.examples_path / "final.tif", maskVals={1: [255]}
        )

    def test_getTransitionMatrix(self):
        table = CrossTableManager(self.init, self.final)

        m = table.getTransitionMatrix()
        for i in range(2):
            self.assertAlmostEqual(sum(m[i, :]), 1.0)

        # ~ print table.getCrosstable().T
        # ~ [[ 4065     4     2     1]
        # ~ [ 1657 62871   260   364]
        # ~ [  514   539 80689  1969]
        # ~ [    2    15     7  8677]]
        # ~ print table.pixelArea
        # ~ {'unit': 'metre', 'area': 624.5137844096937}

        stat = table.getTransitionStat()
        initArea = [
            2543020.1301162727,
            40688322.08186036,
            52278673.40671987,
            5433894.43814875,
        ]
        finalArea = [
            3895716.9871476693,
            39612284.83132246,
            50559386.95823998,
            6876521.28013514,
        ]
        np.testing.assert_almost_equal(initArea, stat["init"])
        np.testing.assert_almost_equal(finalArea, stat["final"])

    def test_different_classes_rasters(self):
        n_class_raster = Raster(self.examples_path / "raster_n_classes.tif")
        n_plus_1_class_raster = Raster(
            self.examples_path / "raster_n_plus_1_classes.tif"
        )

        table = CrossTableManager(n_class_raster, n_plus_1_class_raster)

        def get_inf_indices():
            temp_table = table.getCrosstable().getCrosstable()
            support_table = 1.0 / np.sum(temp_table, axis=1)
            inf_indices = []
            for index, item in enumerate(support_table):
                if not np.isposinf(item):
                    continue
                inf_indices.append(index)
            return inf_indices

        transition_matrix = table.getTransitionMatrix()

        for inf_index in get_inf_indices():
            for column in range(transition_matrix.shape[1]):
                value = 1.0 if column == inf_index else 0.0
                self.assertAlmostEqual(
                    transition_matrix[inf_index, column], value
                )


if __name__ == "__main__":
    unittest.main()
