# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.sampler.sampler import Sampler


class TestSample (unittest.TestCase):
    def setUp(self):
        self.factors = [Raster('../../examples/multifact.tif')]
        self.output = Raster('../../examples/sites.tif')
        self.state = self.output

        self.factors2 = [Raster('../../examples/multifact.tif'), Raster('../../examples/multifact.tif')]
        self.factors3 = [Raster('../../examples/two_band.tif')]
        self.factors4 = [Raster('../../examples/two_band.tif'), Raster('../../examples/multifact.tif')]

    def test_setTrainingData(self):
        smp = Sampler(self.state, self.factors,  self.output, ns=0)
        smp.setTrainingData(self.state, self.output, shuffle=False)

        data = np.array(
            [
                ([0,3], 1.0, 1.0, 1.0),
                ([1,3], 2.0, 1.0, 2.0),
                ([2,3], 1.0, 3.0, 1.0),
                ([0,2], 1.0, 3.0, 1.0),
                ([1,2], 2.0, 2.0, 2.0),
                ([2,2], 1.0, 1.0, 1.0),
                ([0,1], 0.0, 0.0, 0.0),
                ([1,1], 1.0, 3.0, 1.0),
                ([2,1], 2.0, 1.0, 2.0),
            ],
            dtype=[('coords', float, 2), ('state', float, (1,)), ('factors', float, (1,)), ('output', float, 1)]
        )
        for i in range(len(data)):
            assert_array_equal(data[i]['coords'], smp.data[i]['coords'])
            assert_array_equal(data[i]['factors'], smp.data[i]['factors'])
            assert_array_equal(data[i]['output'], smp.data[i]['output'])
            assert_array_equal(data[i]['state'],  smp.data[i]['state'])

        # two factor_rasters
        smp = Sampler(self.state, self.factors2, self.output, ns=1)
        smp.setTrainingData(self.state, self.output)

        # Check all except coords
        data = np.array(
            [
                ([1,2,1,   1,2,1,  0,1,2],
                 [ 1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,
                            1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 2.0)
            ],
            dtype=[('state', float, (9,)), ('factors', float, (18,)), ('output', float, 1)]
        )
        assert_array_equal(data[0]['factors'], smp.data[0]['factors'])
        assert_array_equal(data[0]['output'], smp.data[0]['output'])
        assert_array_equal(data[0]['state'],  smp.data[0]['state'])

        # Multiband factors
        smp = Sampler(self.state, self.factors3, self.output, ns=1)
        smp.setTrainingData(self.state, self.output)

        # Check all except coords
        data = np.array(
            [
                ([1,2,1,   1,2,1,  0,1,2],
                 [ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 2.0)
            ],
            dtype=[('state', float, (9,)), ('factors', float, (18,)), ('output', float, 1)]
        )

        assert_array_equal(data[0]['factors'], smp.data[0]['factors'])
        assert_array_equal(data[0]['output'], smp.data[0]['output'])
        assert_array_equal(data[0]['state'],  smp.data[0]['state'])

        # Several factor bands, several factor rasters
        smp = Sampler(self.state, self.factors4, self.output, ns=1)
        smp.setTrainingData(self.state, self.output)
        # Check all except coords
        data = np.array(
            [
                ([1,2,1,   1,2,1,  0,1,2],
                 [ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,
                                     1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 2.0)
            ],
            dtype=[('state', float, (9,)), ('factors', float, (27,)), ('output', float, 1)]
        )
        assert_array_equal(data[0]['factors'], smp.data[0]['factors'])
        assert_array_equal(data[0]['output'], smp.data[0]['output'])
        assert_array_equal(data[0]['state'],  smp.data[0]['state'])

        # Mode = Normal
        # As the Multiband factors example, but 10 samples:
        data = np.array(
            [
                ([1,2,1,   1,2,1,  0,1,2],
                 [ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 2.0)
            ],
            dtype=[('state', float, (9,)), ('factors', float, (18,)), ('output', float, 1)]
        )
        smp = Sampler(self.state, self.factors3, self.output, ns=1)
        smp.setTrainingData(self.state, self.output, mode='Normal', samples=10)
        for i in range(10):
            assert_array_equal(data[0]['factors'], smp.data[i]['factors'])
            assert_array_equal(data[0]['output'], smp.data[i]['output'])
            assert_array_equal(data[0]['state'],  smp.data[i]['state'])

        # Mode = Balanced
        smp = Sampler(self.state, self.factors, self.output, ns=0)
        smp.setTrainingData(self.state, self.output, mode='Balanced', samples=15)
        out =  smp.data['output']
        out.sort()
        self.assertEqual(out[0],  0)
        self.assertEqual(out[4],  0)
        self.assertEqual(out[5],  1)
        self.assertEqual(out[9],  1)
        self.assertEqual(out[10], 2)


if __name__ == "__main__":
    unittest.main()
