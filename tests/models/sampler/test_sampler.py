from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.sampler.sampler import Sampler


class TestSample (unittest.TestCase):
    def setUp(self):
        self.examples_path = Path(__file__).parents[2] / "examples"
        self.factors = [Raster(self.examples_path / 'multifact.tif')]
        self.output = Raster(self.examples_path / 'sites.tif')
        self.state = self.output
        #~ sites.tif is 1-band 3x3 raster:
            #~ [1,2,1],
            #~ [1,2,1],
            #~ [0,1,2]

        self.factors2 = [Raster(self.examples_path / 'multifact.tif'), Raster(self.examples_path / 'multifact.tif')]
        self.factors3 = [Raster(self.examples_path / 'two_band.tif')]
        self.factors4 = [Raster(self.examples_path / 'two_band.tif'), Raster(self.examples_path / 'multifact.tif')]

    def test_cat2vect(self):
        smp = Sampler(self.state, self.factors,  self.output, ns=0)
        try:
            assert_array_equal(smp.cat2vect(0), [1, 0])
            assert_array_equal(smp.cat2vect(1), [0, 1])
            assert_array_equal(smp.cat2vect(2), [0, 0])
        except AssertionError as error:
            self.fail(error)

        inputRast = Raster(self.examples_path / 'sites.tif')
        inputRast.resetMask([0])
        smp = Sampler(inputRast, self.factors,  self.output, ns=0)
        try:
            assert_array_equal(smp.cat2vect(1), [1])
            assert_array_equal(smp.cat2vect(2), [0])
        except AssertionError as error:
            self.fail(error)

    def test_get_state(self):
        smp = Sampler(self.state, self.factors,  self.output, ns=0)
        try:
            assert_array_equal(smp.get_state(self.state, 1,1), [0, 0])
            assert_array_equal(smp.get_state(self.state, 0,0), [0, 1])
        except AssertionError as error:
            self.fail(error)

        smp = Sampler(self.state, self.factors,  self.output, ns=1)
        res = [ 0, 1, 0, 0, 0, 1,
                0, 1, 0, 0, 0, 1,
                1, 0, 0, 1, 0, 0,
            ]
        try:
            assert_array_equal(smp.get_state(self.state, 1,1), res)
        except AssertionError as error:
            self.fail(error)

        inputRast = Raster(self.examples_path / 'sites.tif')
        inputRast.resetMask([0])
        smp = Sampler(inputRast, self.factors,  self.output, ns=0)
        
        try:
            assert_array_equal(smp.get_state(self.state, 1,1), [0])
            assert_array_equal(smp.get_state(self.state, 0,0), [1])
        except AssertionError as error:
            self.fail(error)

    def test_setTrainingData(self):
        smp = Sampler(self.state, self.factors,  self.output, ns=0)
        smp.setTrainingData(self.state, self.output, shuffle=False)

        data = np.array(
            [
                ([0,3], [0, 1], 1.0, 1.0),
                ([1,3], [0, 0], 1.0, 2.0),
                ([2,3], [0, 1], 3.0, 1.0),
                ([0,2], [0, 1], 3.0, 1.0),
                ([1,2], [0, 0], 2.0, 2.0),
                ([2,2], [0, 1], 1.0, 1.0),
                ([0,1], [1, 0], 0.0, 0.0),
                ([1,1], [0, 1], 3.0, 1.0),
                ([2,1], [0, 0], 1.0, 2.0),
            ],
            dtype=[('coords', float, 2), ('state', float, (2,)), ('factors', float, (1,)), ('output', float, 1)]
        )
        for i in range(len(data)):
            try:
                assert_array_equal(data[i]['coords'], smp.data[i]['coords'])
                assert_array_equal(data[i]['factors'], smp.data[i]['factors'])
                assert_array_equal(data[i]['output'], smp.data[i]['output'])
                assert_array_equal(data[i]['state'],  smp.data[i]['state'])
            except AssertionError as error:
                self.fail(error)

        # two factor_rasters
        smp = Sampler(self.state, self.factors2, self.output, ns=1)
        smp.setTrainingData(self.state, self.output)

        # Check all except coords
        data = np.array(
            [
                (#State categories: [1,2,1,   1,2,1,  0,1,2],
                 [0,1,  0,0,  0,1,   0,1,  0,0,  0,1,   1,0,  0,1,  0,0],
                 # Factors:
                 [ 1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,
                            1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 # Output:
                 2.0)
            ],
            dtype=[('state', float, (18,)), ('factors', float, (18,)), ('output', float, 1)]
        )
        try:
            assert_array_equal(data[0]['factors'], smp.data[0]['factors'])
            assert_array_equal(data[0]['output'], smp.data[0]['output'])
            assert_array_equal(data[0]['state'],  smp.data[0]['state'])
        except AssertionError as error:
            self.fail(error)

        # Multiband factors
        smp = Sampler(self.state, self.factors3, self.output, ns=1)
        smp.setTrainingData(self.state, self.output)

        # Check all except coords
        data = np.array(
            [
                ([0,1,  0,0,  0,1,   0,1,  0,0,  0,1,   1,0,  0,1,  0,0],
                 [ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 2.0)
            ],
            dtype=[('state', float, (18,)), ('factors', float, (18,)), ('output', float, 1)]
        )

        try:
            assert_array_equal(data[0]['factors'], smp.data[0]['factors'])
            assert_array_equal(data[0]['output'], smp.data[0]['output'])
            assert_array_equal(data[0]['state'],  smp.data[0]['state'])
        except AssertionError as error:
            self.fail(error)

        # Several factor bands, several factor rasters
        smp = Sampler(self.state, self.factors4, self.output, ns=1)
        smp.setTrainingData(self.state, self.output)
        # Check all except coords
        data = np.array(
            [
                ([0,1,  0,0,  0,1,   0,1,  0,0,  0,1,   1,0,  0,1,  0,0],
                 [ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,
                                     1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 2.0)
            ],
            dtype=[('state', float, (18,)), ('factors', float, (27,)), ('output', float, 1)]
        )
        try:
            assert_array_equal(data[0]['factors'], smp.data[0]['factors'])
            assert_array_equal(data[0]['output'], smp.data[0]['output'])
            assert_array_equal(data[0]['state'],  smp.data[0]['state'])
        except AssertionError as error:
            self.fail(error)

        # Mode = Random
        # As the Multiband factors example, but 10 samples:
        data = np.array(
            [
                ([0,1,  0,0,  0,1,   0,1,  0,0,  0,1,   1,0,  0,1,  0,0],
                 [ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.],
                 2.0)
            ],
            dtype=[('state', float, (18,)), ('factors', float, (18,)), ('output', float, 1)]
        )
        smp = Sampler(self.state, self.factors3, self.output, ns=1)
        smp.setTrainingData(self.state, self.output, mode='Random', samples=10)
        for i in range(10):
            try:
                assert_array_equal(data[0]['factors'], smp.data[i]['factors'])
                assert_array_equal(data[0]['output'], smp.data[i]['output'])
                assert_array_equal(data[0]['state'],  smp.data[i]['state'])
            except AssertionError as error:
                self.fail(error)

        # Mode = Stratified
        smp = Sampler(self.state, self.factors, self.output, ns=0)
        smp.setTrainingData(self.state, self.output, mode='Stratified', samples=15)
        out =  smp.data['output']
        out.sort()
        self.assertEqual(out[0],  0)
        self.assertEqual(out[4],  0)
        self.assertEqual(out[5],  1)
        self.assertEqual(out[9],  1)
        self.assertEqual(out[10], 2)


if __name__ == "__main__":
    unittest.main()
