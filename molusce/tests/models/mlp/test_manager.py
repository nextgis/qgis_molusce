# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.dataprovider import Raster
from molusce.models.mlp.manager import MlpManager


class TestMlpManager (unittest.TestCase):
    def setUp(self):
        self.inputs = [Raster('../../examples/multifact.tif')]
        self.output = Raster('../../examples/sites.tif')
        
        self.inputs2 = [Raster('../../examples/multifact.tif'), Raster('../../examples/multifact.tif')]
        
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
        mng.setTrainingData(self.inputs, self.output, ns=0, shuffle=False)
        
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

        mng = MlpManager()
        mng.createMlp(self.inputs2, self.output, [10], ns=1)
        mng.setTrainingData(self.inputs2, self.output, ns=1)
        data = [
            {
            'input': np.array([ 1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,  
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]), 
            'output': np.array([min, min,  max])
            }
        ]
        assert_array_equal(data[0]['input'], mng.data[0]['input'])
        assert_array_equal(data[0]['output'], mng.data[0]['output'])
    
    
    def test_train(self):
        mng = MlpManager()
        mng.createMlp(self.inputs, self.output, [10])
        mng.setTrainingData(self.inputs, self.output, ns=0)
        
        mng.train(1, valPercent=50)
        val = mng.getValError()
        tr  = mng.getTrainError()
        mng.train(20, valPercent=50)
        self.assertGreaterEqual(val, mng.getValError())
        
    
if __name__ == "__main__":
    unittest.main()
