import unittest
from pathlib import Path

from numpy import ma as ma
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst


class TestAreaAnalysisManager(unittest.TestCase):
    def setUp(self):
        examples_path = Path(__file__).parents[2] / "examples"
        self.r1 = Raster(examples_path / "multifact.tif")
        # r1 -> r1 transition
        self.r1r1 = [[5, 5, 15], [15, 10, 5], [0, 15, 5]]

        self.r2 = Raster(examples_path / "multifact.tif")
        self.r2.resetMask([0])
        self.r2r2 = [[0, 0, 8], [8, 4, 0], [100, 8, 0]]

        self.r3 = Raster(examples_path / "multifact.tif")
        self.r3.resetMask([2])

        # Rasters with an uneven number of classes
        self.r4 = Raster(examples_path / "raster_n_classes.tif")
        self.r5 = Raster(examples_path / "raster_n_plus_1_classes.tif")

    def test_AreaAnalyst(self):
        aa = AreaAnalyst(self.r1, self.r1)
        raster = aa.getChangeMap()
        band = raster.getBand(1)
        assert_array_equal(band, self.r1r1)

        # Masked raster
        aa = AreaAnalyst(self.r2, self.r2)
        raster = aa.getChangeMap()
        band = raster.getBand(1)
        assert_array_equal(band, self.r2r2)

        # Checking the equalization of class lists
        area_analyst = AreaAnalyst(self.r4, self.r5)
        self.assertEqual(
            area_analyst.categories, area_analyst.categoriesSecond
        )

    def test_encode(self):
        aa = AreaAnalyst(self.r1, self.r1)
        self.assertEqual(aa.categories, [0, 1, 2, 3])
        self.assertEqual(aa.encode(1, 2), 6)
        for initClass in range(4):
            for finalClass in range(4):
                k = aa.encode(initClass, finalClass)
                self.assertEqual(aa.decode(k), (initClass, finalClass))
        self.assertEqual(aa.finalCodes(0), [0, 1, 2, 3])
        self.assertEqual(aa.finalCodes(1), [4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
