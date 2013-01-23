# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from molusce.dataprovider import Raster
from molusce.models.sampler.sampler import Sampler


class TestSample (unittest.TestCase):
    def setUp(self):
        self.inputs = [Raster('../../examples/multifact.tif')]
        self.output = Raster('../../examples/sites.tif')
        
        self.inputs2 = [Raster('../../examples/multifact.tif'), Raster('../../examples/multifact.tif')]
        self.inputs3 = [Raster('../../examples/two_band.tif')]
        
    def test_setTrainingData(self):
        smp = Sampler(self.inputs, self.output, ns=0)
        smp.setTrainingData(self.inputs, self.output, shuffle=False)
        
        data = [
            {'input': np.array([ 1.]), 'output': np.array([1])}, 
            {'input': np.array([ 1.]), 'output': np.array([2])}, 
            {'input': np.array([ 3.]), 'output': np.array([1])}, 
            {'input': np.array([ 3.]), 'output': np.array([1])}, 
            {'input': np.array([ 2.]), 'output': np.array([2])}, 
            {'input': np.array([ 1.]), 'output': np.array([1])}, 
            {'input': np.array([ 0.]), 'output': np.array([0])}, 
            {'input': np.array([ 3.]), 'output': np.array([1])}, 
            {'input': np.array([ 1.]), 'output': np.array([2])}
        ]
        for i in range(len(data)):
            assert_array_equal(data[i]['input'], smp.data[i]['input'])
            assert_array_equal(data[i]['output'], smp.data[i]['output'])

        # two input rasters
        smp = Sampler(self.inputs2, self.output, ns=1)
        smp.setTrainingData(self.inputs2, self.output)
        data = [
            {
            'input': np.array([ 1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.,  
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]), 
            'output': np.array([2])
            }
        ]
        assert_array_equal(data[0]['input'], smp.data[0]['input'])
        assert_array_equal(data[0]['output'], smp.data[0]['output'])
        
        # Multiband input
        smp = Sampler(self.inputs3, self.output, ns=1)
        smp.setTrainingData(self.inputs3, self.output)
        data = [
            {
            'input': np.array([ 1.,  2.,  1.,  1.,  2.,  1.,  0.,  1.,  2.,  
                                1.,  1.,  3.,  3.,  2.,  1.,  0.,  3.,  1.]), 
            'output': np.array([2])
            }
        ]
        assert_array_equal(data[0]['input'], smp.data[0]['input'])
        assert_array_equal(data[0]['output'], smp.data[0]['output'])
    
if __name__ == "__main__":
    unittest.main()
