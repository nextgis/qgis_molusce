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
        self.inputs = [Raster('../../examples/multifact.tif')]
        self.output = Raster('../../examples/sites.tif')
        
        self.inputs2 = [Raster('../../examples/multifact.tif'), Raster('../../examples/multifact.tif')]
        self.inputs3 = [Raster('../../examples/two_band.tif')]
        
    def test_MlpManager(self):
        mng = MlpManager()
        mng.createMlp(self.inputs2, self.output, [10], ns=1)
        assert_array_equal(mng.getMlpTopology(), [18, 10, 3])
        
        mng = MlpManager()
        mng.createMlp(self.inputs, self.output, [10])
        assert_array_equal(mng.getMlpTopology(), [1, 10, 3])
        
    def test_setTrainingData(self):
        mng = MlpManager()
        mng.createMlp(self.inputs, self.output, [10])
        mng.setTrainingData(self.inputs, self.output, shuffle=False)
        
        min, max = mng.sigmLimits
        
        data = [
            {'input': np.array([ 1.]), 'output': np.array([min,  max, min])}, 
            {'input': np.array([ 1.]), 'output': np.array([min,  min, max])}, 
            {'input': np.array([ 3.]), 'output': np.array([min,  max, min])}, 
            {'input': np.array([ 3.]), 'output': np.array([min,  max, min])}, 
            {'input': np.array([ 2.]), 'output': np.array([min,  min, max])}, 
            {'input': np.array([ 1.]), 'output': np.array([min,  max, min])}, 
            {'input': np.array([ 0.]), 'output': np.array([max,  min, min])}, 
            {'input': np.array([ 3.]), 'output': np.array([min,  max, min])}, 
            {'input': np.array([ 1.]), 'output': np.array([min,  min, max])}
        ]
        for i in range(len(data)):
            assert_array_equal(data[i]['input'], mng.data[i]['input'])
            assert_array_equal(data[i]['output'], mng.data[i]['output'])

        # two input rasters
        mng = MlpManager()
        mng.createMlp(self.inputs2, self.output, [10], ns=1)
        mng.setTrainingData(self.inputs2, self.output)
        data = [
            {
            'input': np.array([ 1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,  
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]), 
            'output': np.array([min, min,  max])
            }
        ]
        assert_array_equal(data[0]['input'], mng.data[0]['input'])
        assert_array_equal(data[0]['output'], mng.data[0]['output'])
        
        # Multiband input
        mng = MlpManager()
        mng.createMlp(self.inputs3, self.output, [10], ns=1)
        mng.setTrainingData(self.inputs3, self.output)
        data = [
            {
            'input': np.array([ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,  
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]), 
            'output': np.array([min, min,  max])
            }
        ]
        assert_array_equal(data[0]['input'], mng.data[0]['input'])
        assert_array_equal(data[0]['output'], mng.data[0]['output'])

    def test_train(self):
        mng = MlpManager()
        mng.createMlp(self.inputs, self.output, [10])
        mng.setTrainingData(self.inputs, self.output)
        
        mng.train(1, valPercent=50)
        val = mng.getValError()
        tr  = mng.getTrainError()
        mng.train(20, valPercent=50, continue_train=True)
        self.assertGreaterEqual(val, mng.getValError())

    def test_predict(self):
        mng = MlpManager()
        mng.createMlp(self.inputs, self.output, [10])
        weights = mng.copyWeights()
        
        # Set weights=0
        # then the output must be sigmoid(0)
        layers = []
        for layer in weights:
            shape = layer.shape
            layers.append(np.zeros(shape))
        mng.setMlpWeights(layers)
        raster = mng.predict(self.inputs)
        assert_array_equal(raster.getBand(1), sigmoid(0)*np.zeros((3,3)))
        
        
        
        
    # Commented while we don't have free rasters to test
    #~ def test_real(self):
        #~ #inputs = [Raster('LPB_dem.tif'), Raster('LPB_luc_2007.tif')]
        #~ inputs = [Raster('LPB_luc_2007.tif'),  Raster('LPB_luc_2007.tif')]
        #~ output = Raster('LPB_luc_2007.tif')
        #~ mng = MlpManager()
        #~ mng.createMlp(inputs, output, [10])
        #~ mng.setTrainingData(inputs, output, ns=0)
        #~ mng.train(1, valPercent=20)
        
        
    
if __name__ == "__main__":
    unittest.main()
