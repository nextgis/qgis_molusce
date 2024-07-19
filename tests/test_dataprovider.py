import os
import unittest
from pathlib import Path

import numpy as np
from numpy import ma as ma
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import ProviderError, Raster


class TestRaster(unittest.TestCase):
    def setUp(self):
        self.sample_path1 = (
            Path(__file__).parents[0] / "examples" / "multifact.tif"
        )
        self.sample_path2 = (
            Path(__file__).parents[0] / "examples" / "sites.tif"
        )
        self.sample_path3 = (
            Path(__file__).parents[0] / "examples" / "two_band.tif"
        )
        self.sample_path4 = (
            Path(__file__).parents[0] / "examples" / "dist_roads.tif"
        )
        self.r1 = Raster(self.sample_path1)
        self.r2 = Raster(self.sample_path2)
        self.r3 = Raster(self.sample_path3)

        # r1
        data1 = np.array([[1, 1, 3], [3, 2, 1], [0, 3, 1]])
        # r2
        data2 = np.array([[1, 2, 1], [1, 2, 1], [0, 1, 2]])
        mask = [
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]
        self.data1 = ma.array(data=data1, mask=mask)
        self.data2 = ma.array(data=data2, mask=mask)

    def test_RasterInit(self):
        self.assertEqual(self.r1.getBandsCount(), 1)
        band = self.r1.getBand(1)
        shape = band.shape
        x = self.r1.getXSize()
        y = self.r1.getYSize()
        self.assertEqual(shape, (x, y))

        self.assertEqual(self.r2.getBandsCount(), 1)
        band = self.r2.getBand(1)
        assert_array_equal(band, self.data2)

        self.assertTrue(self.r1.geoDataMatch(self.r2))

    def test_create(self):
        raster = Raster()
        raster.create([self.data1], geodata=self.r1.getGeodata())
        self.assertTrue(raster.geoDataMatch(self.r1))
        self.assertEqual(raster.getBandsCount(), 1)
        self.assertEqual(set(raster.getBandGradation(1)), set([0, 1, 2, 3]))

    def test_roundBands(self):
        rast = Raster(self.sample_path1)
        rast.bands = rast.bands * 0.1
        rast.roundBands()
        answer = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        assert_array_equal(answer, rast.bands)

        rast = Raster(self.sample_path1)
        rast.bands = rast.bands * 1.1
        rast.roundBands(decimals=1)
        answer = np.array(
            [[[1.1, 1.1, 3.3], [3.3, 2.2, 1.1], [0.0, 3.3, 1.1]]]
        )
        assert_array_equal(answer, rast.bands)

    def test_isContinues(self):
        rast = Raster(self.sample_path1)
        self.assertFalse(rast.isCountinues(bandNo=1))
        rast = Raster(self.sample_path4)
        self.assertTrue(rast.isCountinues(bandNo=1))

    def test_getBandStat(self):
        stat = self.r1.getBandStat(1)
        self.assertAlmostEqual(stat["mean"], 15.0 / 9)
        self.assertAlmostEqual(stat["std"], np.sqrt(10.0 / 9))

    def test_normalize(self):
        multifact = [
            [1, 1, 3],
            [3, 2, 1],
            [0, 3, 1],
        ]

        # Normalize using std and mean
        r1 = Raster(self.sample_path1)
        r1.normalize()
        r1.denormalize()
        assert_array_equal(r1.getBand(1), multifact)

        # Normalize using min and max
        r1 = Raster(self.sample_path1)
        r1.normalize(mode="maxmin")
        r1.denormalize()
        assert_array_equal(r1.getBand(1), multifact)

        # Two normalization procedures
        r1 = Raster(self.sample_path1)
        r1.normalize()
        r1.normalize(mode="maxmin")
        r1.denormalize()
        assert_array_equal(r1.getBand(1), multifact)
        r1 = Raster(self.sample_path1)
        r1.normalize(mode="maxmin")
        r1.normalize()
        r1.denormalize()
        assert_array_equal(r1.getBand(1), multifact)

    def test_getNeighbours(self):
        neighbours = self.r2.getNeighbours(row=1, col=0, size=0)
        self.assertEqual(neighbours, [[1]])

        neighbours = self.r2.getNeighbours(row=1, col=1, size=1)
        assert_array_equal(neighbours, [self.data2])

        neighbours = self.r3.getNeighbours(row=1, col=1, size=1)
        assert_array_equal(neighbours, [self.data2, self.data1])

        # Check pixel on the raster bound and nonzero neighbour size
        self.assertRaises(
            ProviderError, self.r2.getNeighbours, col=1, row=0, size=1
        )
        self.assertRaises(
            ProviderError, self.r2.getNeighbours, col=1, row=1, size=2
        )

    def test_geodata(self):
        geodata = self.r1.getGeodata()
        self.r1.setGeoData(geodata)
        geodata["xSize"] = geodata["xSize"] + 10
        self.assertRaises(ProviderError, self.r1.setGeoData, geodata=geodata)

        self.assertTrue(self.r1.geoDataMatch(self.r1))
        self.assertTrue(
            self.r1.geoDataMatch(raster=None, geodata=self.r1.getGeodata())
        )

        self.assertTrue(self.r1.geoTransformMatch(self.r1))
        self.assertTrue(
            self.r1.geoTransformMatch(
                raster=None, geodata=self.r1.getGeodata()
            )
        )

    def test_save(self):
        try:
            filename = "temp.tiff"
            self.r1.save(filename)
            r2 = Raster(filename)
            self.assertEqual(r2.get_dtype(), self.r1.get_dtype())
            self.assertEqual(r2.getBandsCount(), self.r1.getBandsCount())
            for i in range(r2.getBandsCount()):
                assert_array_equal(r2.getBand(i + 1), self.r1.getBand(i + 1))
        finally:
            os.remove(filename)

    def test_getBandGradation(self):
        self.assertEqual(set(self.r1.getBandGradation(1)), set([0, 1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
