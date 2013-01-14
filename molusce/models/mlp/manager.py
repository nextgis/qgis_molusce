# encoding: utf-8

import numpy as np

from molusce.dataprovider import Raster
from molusce.models.mlp.model import MLP

class MlpManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class MlpManager(object):
    '''This class gets the data extracted from the UI and
    pass it to multi-layer perceptron, then gets and stores the result.
    '''
    
    def __init__(self, MLP=None):
        self.MLP = MLP
        
        self.layers = None
        if self.MLP:
            self.layers = self.getMlpTopology()
    
    def createMlp(self, inputs, output, hidden_layers, ns=0):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param hidden_layers    List of neuron counts in hidden layers.
        @param ns               Neighbourhood size.
        '''
        
        if output.getBandsCount() != 1:
            raise MplManagerError('Output layer must have one band!')
        
        # total input bands count
        total_input_bands = 0
        for raster in inputs:
            total_input_bands = total_input_bands + raster.getBandsCount()
        
        # pixel count in the neighbourhood of ns size
        neighbours = (2*ns+1)**2
        
        # inputs of the MLP
        input_neurons = total_input_bands * neighbours

        # output class count
        band = output.getBand(1)
        classes = len(np.unique(band.compressed()))
        
        # set neuron counts in the MLP layers
        self.layers = hidden_layers
        self.layers.insert(0, input_neurons)
        self.layers.append(classes)
        
        self.MLP = MLP(*self.layers)
    
    def getOutput(self):
        pass
    
    def getMlpTopology(self):
        return self.MLP.shape
    
    def readMlp(self):
        pass
    
    def saveMlp(self):
        pass
        
    
    def setTrainData(self):
        pass
    
    def train(self):
        pass
    


