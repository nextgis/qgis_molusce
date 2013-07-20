# encoding: utf-8

import sys
sys.path.insert(0, '../../../../../')

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.mlp.manager import MlpManager, sigmoid



class TestMlpManager (unittest.TestCase):
    def setUp(self):
        self.factors = [Raster('../../examples/multifact.tif')]
        self.output = Raster('../../examples/sites.tif')
        #~ sites.tif is 1-band 3x3 raster:
            #~ [1,2,1],
            #~ [1,2,1],
            #~ [0,1,2]

        self.factors2 = [Raster('../../examples/multifact.tif'), Raster('../../examples/multifact.tif')]
        self.factors3 = [Raster('../../examples/two_band.tif')]
        self.factors4 = [Raster('../../examples/two_band.tif'), Raster('../../examples/multifact.tif')]

        self.output1  = Raster('../../examples/data.tif')
        self.state1   = self.output1
        self.factors1 = [Raster('../../examples/fact16.tif')]

    def test_MlpManager(self):
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors2, self.output, [10])
        assert_array_equal(mng.getMlpTopology(), [2*9 + 2*9, 10, 3])

        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        assert_array_equal(mng.getMlpTopology(), [2*1 + 1*1, 10, 3])

    def test_setTrainingData(self):
        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        stat = self.factors[0].getBandStat(1) # mean & std
        m,s = stat['mean'], stat['std']
        mng.setTrainingData(self.output, self.factors, self.output, shuffle=False)

        min, max = mng.sigmin, mng.sigmax
        data = np.array(
            [
                ((0,3), [0, 1], (1.0-m)/s, [min,  max, min]),
                ((1,3), [0, 0], (1.0-m)/s, [min,  min, max]),
                ((2,3), [0, 1], (3.0-m)/s, [min,  max, min]),
                ((0,2), [0, 1], (3.0-m)/s, [min,  max, min]),
                ((1,2), [0, 0], (2.0-m)/s, [min,  min, max]),
                ((2,2), [0, 1], (1.0-m)/s, [min,  max, min]),
                ((0,1), [1, 0], (0.0-m)/s, [max,  min, min]),
                ((1,1), [0, 1], (3.0-m)/s, [min,  max, min]),
                ((2,1), [0, 0], (1.0-m)/s, [min,  min, max]),
            ],
            dtype=[('coords', float, 2),('state', float, (2,)), ('factors', float, (1,)), ('output', float, 3)]
        )
        self.assertEqual(mng.data.shape, (9,))
        for i in range(len(data)):
            assert_array_equal(data[i]['coords'], mng.data[i]['coords'])
            assert_array_equal(data[i]['factors'], mng.data[i]['factors'])
            assert_array_equal(data[i]['output'], mng.data[i]['output'])

        # two input rasters
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors2, self.output, [10])
        mng.setTrainingData(self.output, self.factors2, self.output)
        data = [
            {
            'factors': (np.array([ 1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]) - stat['mean'])/stat['std'],
            'output': np.array([min, min,  max]),
            'state': np.array([0,1,  0,0,  0,1,   0,1,  0,0,  0,1,   1,0,  0,1,  0,0])
            }
        ]
        self.assertEqual(mng.data.shape, (1,))
        assert_array_equal(data[0]['factors'], mng.data[0]['factors'])
        assert_array_equal(data[0]['output'], mng.data[0]['output'])
        assert_array_equal(data[0]['state'], mng.data[0]['state'])

        # Multiband input
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors3, self.output, [10])
        stat1 = self.factors3[0].getBandStat(1) # mean & std
        m1,s1 = stat1['mean'], stat1['std']
        stat2 = self.factors3[0].getBandStat(2) # mean & std
        m2,s2 = stat2['mean'], stat2['std']
        mng.setTrainingData(self.output, self.factors3, self.output)

        data = [
            {
            'factors': np.array([ (1.-m1)/s1,  (2.-m1)/s1,  (1.-m1)/s1,  (1.-m1)/s1,  (2.-m1)/s1,  (1.-m1)/s1,  (0.-m1)/s1,  (1.-m1)/s1,  (2.-m1)/s1,
                                (1.-m2)/s2,  (1.-m2)/s2,  (3.-m2)/s2,  (3.-m2)/s2,  (2.-m2)/s2,  (1.-m2)/s2,  (0.-m2)/s2,  (3.-m2)/s2,  (1.-m2)/s2]),
            'output': np.array([min, min,  max]),
            'state': np.array([0,1,  0,0,  0,1,   0,1,  0,0,  0,1,   1,0,  0,1,  0,0])
            }
        ]
        self.assertEqual(mng.data.shape, (1,))
        assert_array_equal(data[0]['factors'], mng.data[0]['factors'])
        assert_array_equal(data[0]['output'], mng.data[0]['output'])
        assert_array_equal(data[0]['state'], mng.data[0]['state'])

        # Complex case:
        mng = MlpManager(ns=1)
        mng.createMlp(self.output, self.factors4, self.output, [10])
        stat1 = self.factors4[0].getBandStat(1) # mean & std
        m1,s1 = stat1['mean'], stat1['std']
        stat2 = self.factors4[0].getBandStat(2) # mean & std
        m2,s2 = stat2['mean'], stat2['std']
        stat3 = self.factors4[1].getBandStat(1)
        m3,s3 = stat3['mean'], stat2['std']

        mng.setTrainingData(self.output, self.factors4, self.output)

        data = [
            {
            'factors': np.array([ (1.-m1)/s1,  (2.-m1)/s1,  (1.-m1)/s1,  (1.-m1)/s1,  (2.-m1)/s1,  (1.-m1)/s1,  (0.-m1)/s1,  (1.-m1)/s1,  (2.-m1)/s1,
                                (1.-m2)/s2,  (1.-m2)/s2,  (3.-m2)/s2,  (3.-m2)/s2,  (2.-m2)/s2,  (1.-m2)/s2,  (0.-m2)/s2,  (3.-m2)/s2,  (1.-m2)/s2,
                                (1.-m3)/s3,  (1.-m3)/s3,  (3.-m3)/s3,  (3.-m3)/s3,  (2.-m3)/s3,  (1.-m3)/s3,  (0.-m3)/s3,  (3.-m3)/s3,  (1.-m3)/s3]),
            'output': np.array([min, min,  max]),
            'state': np.array([0,1,  0,0,  0,1,   0,1,  0,0,  0,1,   1,0,  0,1,  0,0])
            }
        ]
        self.assertEqual(mng.data.shape, (1,))
        assert_array_equal(data[0]['factors'], mng.data[0]['factors'])
        assert_array_equal(data[0]['output'], mng.data[0]['output'])
        assert_array_equal(data[0]['state'], mng.data[0]['state'])

    def test_train(self):
        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        mng.setTrainingData(self.output, self.factors, self.output)

        mng.train(1, valPercent=50)
        val = mng.getMinValError()
        tr  = mng.getTrainError()
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
        raster = mng.getPrediction(self.output, self.factors, calcTransitions=True)
        assert_array_equal(raster.getBand(1), sigmoid(0)*np.ones((3,3)))
        # Confidence is zero
        confid = mng.getConfidence()
        self.assertEquals(confid.getBand(1).dtype, np.uint8)
        assert_array_equal(confid.getBand(1), np.zeros((3,3)))
        # Transition Potentials (is (sigmoid(0) - sigmin)/sigrange )
        potentials = mng.getTransitionPotentials()
        cats = self.output.getBandGradation(1)
        for cat in cats:
            map = potentials[cat]
            self.assertEquals(map.getBand(1).dtype, np.uint8)
            assert_array_equal(map.getBand(1), 50*np.ones((3,3)))

    # Commented while we don't have free rasters to test
    #~ def test_real(self):
        #~ #inputs = [Raster('LPB_dem.tif'), Raster('LPB_luc_2007.tif')]
        #~ inputs = [Raster('LPB_luc_2007.tif'),  Raster('LPB_luc_2007.tif')]
        #~ output = Raster('LPB_luc_2007.tif')
        #~ mng = MlpManager()
        #~ mng.createMlp(output, inputs, output, [10])
        #~ mng.setTrainingData(output, inputs, output)
        #~ mng.train(1, valPercent=20)



if __name__ == "__main__":
    unittest.main()
