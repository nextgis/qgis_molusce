# encoding: utf-8

import sys
sys.path.insert(0, '../../../../')

import unittest
from numpy.testing import assert_array_equal

import numpy as np
from numpy import ma as ma

from molusce.models.crosstabs.model  import CrossTable
from molusce.models.area_analysis.manager import AreaAnalyst
from molusce.dataprovider import Raster
from molusce.models.simulator.sim import Simulator


class Model(object):
    '''
    Simple predicting model for Simulator tests
    '''
    def __init__(self, state):
        self.state = state
        self._predict(state)
    
    def getConfidence(self):
        return self.confidence
        
    def getPrediction(self, state, factors=None):
        self._predict(state, factors)
        return self.prediction
        
    def _predict(self, state, factors = None):
        geodata = self.state.getGeodata()
        band = state.getBand(1)
        rows, cols = state.geodata['xSize'], state.geodata['ySize']
        # Let the prediction is: 1 -> 2, 2- >3, 3 -> 1
        
        predicted_band  = np.copy(band)
        predicted_band[band == 1] = 2
        predicted_band[band == 2] = 3
        predicted_band[band == 3] = 1
        
        # Let the confidence is 1/(1+row+col), where row is row number of the cell, col is column number of the cell.
        confidence_band = np.zeros([rows, cols])
        for i in xrange(cols):
            for j in xrange(rows):
                confidence_band[i,j] = 1.0/(1+i+j)
        
        predicted_band  = [np.ma.array(data = predicted_band, mask = band.mask)]
        confidence_band = [np.ma.array(data = confidence_band, mask = band.mask)]
        self.prediction = Raster()
        self.prediction.create(predicted_band, state.geodata)
        self.confidence = Raster()
        self.confidence.create(confidence_band, state.geodata)
    
        

class TestCrossTable (unittest.TestCase):
    
    def setUp(self):

        # Raster1:
            #~ [1, 1, 3,],
            #~ [3, 2, 1,],
            #~ [0, 3, 1,]
        self.raster1 = Raster('../../examples/multifact.tif')
        self.raster1.setMask([0])
        
        self.X = np.array([
            [1, 2, 3],
            [3, 2, 1],
            [0, 1, 1]
        ])
        self.X = np.ma.array(self.X, mask=(self.X == 0))
        self.raster2 = Raster()
        self.raster2.create([self.X], self.raster1.getGeodata())
        
        self.aa = AreaAnalyst(self.raster1, self.raster2)
        
        self.crosstab = CrossTable(self.raster1.getBand(1), self.raster2.getBand(1))
        
        # Simple model
        self.model = Model(self.raster1)

    def test_compute_table(self):

        # print self.crosstab.T
        # CrossTab:
        #  [[ 3.  1.  0.]           
        #   [ 0.  1.  0.]           
        #   [ 1.  0.  2.]]          
        # prediction = self.model.getPrediction()
        # prediction = [[2.0 2.0 1.0]
                     #  [1.0 3.0 1.0]
                     #  [-- 1.0 2.0]]
        # confidence = self.model.getConfidence()
        # confidence =     [[1.0 0.5  0.33]
                         #  [0.5 0.33 0.25]
                         #  [--  0.25 0.2]]
        
        
        result = np.array([
            [2.0, 1.0, 3.0],
            [1.0, 2.0, 1.0],
            [0,   3.0, 1.0]
        ])
        result = np.ma.array(result, mask = (result==0))
        
        simulator = Simulator(self.raster1, None, self.model, self.crosstab)    # The model does't use factors
        simulator.sim()
        state = simulator.getState().getBand(1)
        assert_array_equal(result, state)
        
        result = np.array([
            [2.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
            [0,   3.0, 1.0]
        ])
        result = np.ma.array(result, mask = (result==0))
        
        simulator = Simulator(self.raster1, None, self.model, self.crosstab)
        simulator.simN(2)
        state = simulator.getState().getBand(1)
        assert_array_equal(result, state)
    
    
if __name__ == "__main__":
    unittest.main()
