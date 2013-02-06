import unittest

import numpy as np

from molusce.dataprovider import Raster
from molusce.models.mlp.manager import MlpManager


class Simulator(object):
    """
    Based on a model, controls simulation via cellular automaton
    over a number of cycles
    """
    
    def __init__(self, inputs, model):
        '''
        @param inputs           List of the input rasters.
        @param model            Model that is used for predict.
        '''
        self.inputs = inputs
        self.model  = model
        
    def predict(self):
        '''
        Predict new states via model and save the result into file.
        '''
        return self.model.predict(self.inputs)
        
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
        

    
