import unittest

import numpy as np

from molusce.dataprovider import Raster
from molusce.models.mlp.manager import MlpManager


class Simulator(object):
    """
    Based on a model, controls simulation via cellular automaton
    over a number of cycles
    """
    
    def __init__(self, state, factors, model):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param model            Model that is used for predict.
        '''
        self.state = state
        self.factors = factors
        self.model  = model
        
    def predict(self):
        '''
        Predict new states via model and save the result into file.
        '''
        return self.model.predict(self.state, self.factors)
    
    def errorMap(self, answer):
        '''
        Create map of correct and incorrect prediction. 
        This function compares the answer and the result of predicting procedure,
        correct pixel is marked as 0, incorrect is marked as 1.
        '''
        result = self.predict()
        b = result.getBand(1)
        a = answer.getBand(1)
        diff = (a-b).astype(int)
        result.setBand(diff)
        
        return result
        

    
