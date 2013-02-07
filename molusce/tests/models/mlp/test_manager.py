# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.dataprovider import Raster
from molusce.models.mlp.manager import MlpManager, sigmoid



class TestMlpManager (unittest.TestCase):
    def setUp(self):
        self.factors = [Raster('../../examples/multifact.tif')]
        self.output = Raster('../../examples/sites.tif')
        
        self.factors2 = [Raster('../../examples/multifact.tif'), Raster('../../examples/multifact.tif')]
        self.factors3 = [Raster('../../examples/two_band.tif')]
        
    def test_MlpManager(self):
        mng = MlpManager()
        mng.createMlp(self.output, self.factors2, self.output, [10], ns=1)
        assert_array_equal(mng.getMlpTopology(), [27, 10, 3])
        
        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        assert_array_equal(mng.getMlpTopology(), [2, 10, 3])
        
    def test_setTrainingData(self):
        mng = MlpManager()
        mng.createMlp(self.output, self.factors, self.output, [10])
        mng.setTrainingData(self.output, self.factors, self.output, shuffle=False)
        
        min, max = mng.sigmin, mng.sigmax
        data = np.array(
            [
                (1.0, 1.0, [min,  max, min]),
                (2.0, 1.0, [min,  min, max]),
                (1.0, 3.0, [min,  max, min]),
                (1.0, 3.0, [min,  max, min]),
                (2.0, 2.0, [min,  min, max]),
                (1.0, 1.0, [min,  max, min]),
                (0.0, 0.0, [max,  min, min]),
                (1.0, 3.0, [min,  max, min]),
                (2.0, 1.0, [min,  min, max]),
            ], 
            dtype=[('state', float, (1,)), ('factors', float, (1,)), ('output', float, 3)]
        )
        self.assertEqual(mng.data.shape, (9,))
        for i in range(len(data)):
            assert_array_equal(data[i]['factors'], mng.data[i]['factors'])
            assert_array_equal(data[i]['output'], mng.data[i]['output'])

        # two input rasters
        mng = MlpManager()
        mng.createMlp(self.output, self.factors2, self.output, [10], ns=1)
        mng.setTrainingData(self.output, self.factors2, self.output)
        data = [
            {
            'factors': np.array([ 1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,  
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]), 
            'output': np.array([min, min,  max]),
            'state': np.array([1,2,1,   1,2,1,  0,1,2])
            }
        ]
        self.assertEqual(mng.data.shape, (1,))
        assert_array_equal(data[0]['factors'], mng.data[0]['factors'])
        assert_array_equal(data[0]['output'], mng.data[0]['output'])
        assert_array_equal(data[0]['state'], mng.data[0]['state'])
        
        # Multiband input
        mng = MlpManager()
        mng.createMlp(self.output, self.factors3, self.output, [10], ns=1)
        mng.setTrainingData(self.output, self.factors3, self.output)
        data = [
            {
            'factors': np.array([ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,  
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]), 
            'output': np.array([min, min,  max]),
            'state': np.array([1,2,1,   1,2,1,  0,1,2])
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
        val = mng.getValError()
        tr  = mng.getTrainError()
        mng.train(20, valPercent=50, continue_train=True)
        self.assertGreaterEqual(val, mng.getValError())

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
        mng.predict(self.output, self.factors)
        
        # Prediction
        raster = mng.getPrediction()
        assert_array_equal(raster.getBand(1), sigmoid(0)*np.zeros((3,3)))
        # Confidence
        confid = mng.getConfidence()
        assert_array_equal(confid.getBand(1), sigmoid(0)*np.zeros((3,3)))

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
