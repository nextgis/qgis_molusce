import unittest

import numpy as np

from molusce.dataprovider import Raster
from molusce.models.mlp.manager import MlpManager
from molusce.models.area_analysis.manager import AreaAnalyst

class Simulator(object):
    """
    Based on a model, controls simulation via cellular automaton
    over a number of cycles
    """
    
    def __init__(self, state, factors, model, crosstable):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param model            Model that is used for predict.
        @param crosstable       Crosstable, contains transition matrix between states T(i,j). 
                                The matrix contains number of pixels that are moved
                                from init class i to final class j.
        '''
        self.state = state
        self.factors = factors
        self.predicted = None      # Raster of predicted classes
        
        self.model  = model
        self.crosstable = crosstable
        
        self.updatePrediction(self.state)
    
    
    def getConfidence(self):
        '''
        Return raster of model's prediction confidence.
        '''
        return self.model.getConfidence()
        
    def getPrediction(self):
        '''
        Predict new states via model.
        '''
        return self.predicted
        
    def getState(self):
        return self.state
    
    def errorMap(self, answer):
        '''
        Create map of correct and incorrect prediction. 
        This function compares the answer and the result of predicting procedure,
        correct pixel is marked as 0, incorrect is marked as 1.
        '''
        result = self.prediction()
        b = result.getBand(1)
        a = answer.getBand(1)
        diff = (a-b).astype(int)
        result.setBand(diff)
        
        return result
        
    def sim(self):
        '''
        Make 1 iteracion of simulation.
        '''
        transition = self.crosstable
        
        prediction = self.getPrediction()
        state = self.getState()
        new_state = state.getBand(1).copy()         # New states (the transition result)
        analyst = AreaAnalyst(state, prediction)
        classes = analyst.classes
        changes = analyst.makeChangeMap().getBand(1)
        
        # Make transition between classes according to 
        # number of moved pixel in crosstable
        for initClass in classes:
            for finalClass in classes:
                if initClass == finalClass: continue
                
                n = transition.getTransition(initClass, finalClass)   # Number of pixels to be moved
                # Find n appropriate places for transition initClass -> finalClass
                class_code = analyst.encode(initClass, finalClass)
                places = (changes==class_code)      # Array of places where transitions initClass -> finalClass are occured
                confidence = self.getConfidence().getBand(1)
                confidence = confidence * places # The higher is number in cell, the higer is probability of transition in the cell
                indices = []
                for i in range(n):
                    index = np.unravel_index(confidence.argmax(), confidence.shape)     # Select the cell with biggest probability
                    indices.append(index)
                    confidence[index] = -1       # Mark the cell to prevent second selection
                
                # Now "indices" contains indices of the appropriate places,
                # make transition initClass -> finalClass
                for index in indices:
                    new_state[index] = finalClass 
        
        result = Raster()
        result.create([new_state], state.getGeodata())
        self.state = result
        self.updatePrediction(result)
                
    
    def simN(self, N):
        '''
        Make N iterations of simulation.
        '''
        for i in range(N):
            self.sim()
    
    def updatePrediction(self, state):
        '''
        Update prediction using new classes (raster "state")
        '''
        self.model.predict(state, self.factors)
        self.predicted = self.model.getPrediction()
        
