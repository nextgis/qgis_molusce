# encoding: utf-8


# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)



import numpy as np
from sklearn import linear_model as lm

from molusce.dataprovider import Raster, ProviderError
from molusce.models.sampler.sampler import Sampler


class LRError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class LR(object):
    """
    Implements Logistic Regression model definition and calibration
    (maximum liklihood parameter estimation).
    """
    
    def __init__(self, ns=0, logreg=None):
        if logreg:
            self.logreg = logreg
        else:
            self.logreg = lm.LogisticRegression()
        
        self.ns = ns            # Neighbourhood size of training rasters.
        self.data = None        # Training data
        self.classlist = None   # List of unique output values of the output raster
        
        # Results of the LR prediction
        self.prediction = None  # Raster of the LR prediction results
        self.confidence = None  # Raster of the LR results confidence 
    
    def getConfidence(self):
        return self.confidence
    
    def outputConfidence(self, input):
        '''
        Return confidence (difference between 2 biggest probabilities) of the LR output.
        '''
        out_scl = self.logreg.predict_proba(input)[0]
        # Calculate the confidence:
        out_scl.sort()
        return out_scl[-1] - out_scl[-2]
    
    def getPrediction(self, state, factors):
        self._predict(state, factors)
        return self.prediction
        
    def _predict(self, state, factors):
        '''
        Calculate output and confidence rasters using MLP model and input rasters
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        '''
        
        rows, cols = state.geodata['xSize'], state.geodata['ySize']
        for r in factors:
            if not state.geoDataMatch(r):
                raise LRError('Geometries of the input rasters are different!')
        
        predicted_band  = np.zeros([rows, cols])
        confidence_band = np.zeros([rows, cols])
        
        sampler = Sampler(state, factors, self.ns)
        mask = state.getBand(1).mask
        for i in xrange(rows):
            for j in xrange(cols):
                if not mask[i,j]:
                    input = sampler.get_inputs(state, factors, i,j)
                    if input != None:
                        out = self.logreg.predict(input)
                        predicted_band[i,j] = out
                        confidence = self.outputConfidence(input)
                        confidence_band[i, j] = confidence
                    else: # Input sample is incomplete => mask this pixel
                        mask[i, j] = True
        predicted_band  = [np.ma.array(data = predicted_band, mask = mask)]
        confidence_band = [np.ma.array(data = confidence_band, mask = mask)]
        
        self.prediction = Raster()
        self.prediction.create(predicted_band, state.geodata)
        self.confidence = Raster()
        self.confidence.create(confidence_band, state.geodata)
        
    def read(self):
        pass
        
    def save(self):
        pass
        
    def setTrainingData(self, state, factors, output):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param output           Raster that contains classes to predict.
        '''
        if not self.logreg:
            raise LRError('You must create a MLP before!')
        
        sampler = Sampler(state, factors, output, self.ns)
        sampler.setTrainingData(state, factors, output, shuffle=False)
        
        outputVecLen  = sampler.outputVecLen
        stateVecLen   = sampler.stateVecLen
        factorVectLen = sampler.factorVectLen
        size = len(sampler.data)
        
        self.data = sampler.data
        
    def train(self):
        X = np.vstack( (self.data['state'], self.data['factors']) )
        X = np.transpose(X)
        Y = self.data['output']
        self.logreg.fit(X, Y)


