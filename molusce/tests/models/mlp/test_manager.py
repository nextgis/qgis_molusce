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
        
        
    
if __name__ == "__main__":
    unittest.main()
