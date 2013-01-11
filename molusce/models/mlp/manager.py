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
    def __init__(self, inputs, output, hidden_layers, learning_rate, momentum, nz=0):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param hidden_layers    List of neuron counts in hidden layers.
        @param learning_rate    Learning rate for the multi-layer perceptron.
        @param momentum         Momentum for the multi-layer perceptron.
        @param ns               Neighbourhood size.
        '''
        
        # input-output rasters 
        # (needed to calculate MLP input/output layer sizes and training)
        self.output = output
        self.inputs = inputs
        if self.output.getBandsCount() != 1:
            raise MplManagerError('Output layer must have one band!')
        
        self.ns = nz
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # total input bands count
        total_input_bands = 0
        for raster in self.inputs:
            total_input_bands = self.total_input_bands + raster.getBandsCount()
        
        # pixel count in the neighbourhood of ns size
        neighbours = (2*ns+1)**2
        
        # inputs of the MLP
        input_neurons = total_input_bands * neighbours

        # output class count
        band = output.getBand(1)
        classes = np.unique(band.compressed())
        
        # set neuron counts in the MLP layers
        self.layers = hidden_layers.inser(0, input_neurons)
        self.layers.append(classes)
        
        self.MPL = MPL(self.layers)
    
        
    def getOutput(self):
        pass
    
    def readMPL(self):
        pass
    
    def saveMPL(self):
        pass
        
        
    
    def train(self):
        pass
    


