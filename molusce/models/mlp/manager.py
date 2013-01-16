# encoding: utf-8

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster, ProviderError
from molusce.models.mlp.model import MLP, sigmoid

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
            
        self.data = None        # Ttaining data
        self.classlist = None   # List of unique output values of the output raster
        
        # Outputs of the activation function for small and big numbers
        self.sigmLimits = (sigmoid(-1000), sigmoid(1000))
    
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
        
        # Input neuron count of the MLP
        input_neurons = total_input_bands * neighbours

        # Output class (neuron) count
        band = output.getBand(1)
        self.classlist = np.unique(band.compressed())
        classes = len(self.classlist)
        
        # set neuron counts in the MLP layers
        self.layers = hidden_layers
        self.layers.insert(0, input_neurons)
        self.layers.append(classes)
        
        self.MLP = MLP(*self.layers)
    
    def getInputVectLen(self):
        '''Length of input data vector of the MLP'''
        shape = self.getMlpTopology()
        return shape[0]
    
    
    def getOutput(self):
        pass
    
    def getOutputVectLen(self):
        '''Length of input data vector of the MLP'''
        shape = self.getMlpTopology()
        return shape[-1]
    def getOutputVector(self, val):
        '''Convert number val into vector .
        for example, set self.classlist = [1, 3, 4] then
        if val = 1, result = [ 1, -1, -1]
        if val = 3, result = [-1,  1, -1]
        where -1 is minimum of the sigmoid, 1 is max of the sigmoid
        '''
        min, max = self.sigmLimits
        size = self.getOutputVectLen()
        res = np.ones(size) * (min)
        ind = np.where(self.classlist==val)
        res[ind] = max
        return res
    
    def getMlpTopology(self):
        return self.MLP.shape
    
    def readMlp(self):
        pass
    
    def saveMlp(self):
        pass
        
    
    def setTrainingData(self, inputs, output, ns=0):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param ns               Neighbourhood size.
        '''
        for r in inputs:
            if not output.isGeoDataMatch(r):
                raise MlpManagerError('Geometries of the inputs and outputs are different!')
        
        pixel_count = (2*ns+1)**2 # Pixel count in the neighbourhood
        input_vect_len = self.getInputVectLen()
        output_vect_len = self.getOutputVectLen()
        
        
        (rows,cols) = (output.getXSize(), output.getYSize())
        first_sample = True
        for i in xrange(ns, rows - ns):         # Eliminate the raster boundary of (ns)-size because
            for j in xrange(ns, rows-ns):       # the samples are incomplete in that region
                inp = ma.zeros(input_vect_len)
                outp = ma.zeros(output_vect_len)
                sample = ma.zeros(1, dtype=[('input',  float, input_vect_len), ('output', float, output_vect_len)])
                sample_complete = True # Are pixels in the neighbourhood defined/unmasked?
                try: 
                    out = output.getNeighbours(i,j,0).flatten() # Get the pixel
                    if any(out.mask): # Eliminate incomplete samples
                        sample_complete = False
                        continue
                    else:
                        sample['output'] = self.getOutputVector(out)
                    for (k,r) in enumerate(inputs):
                        neighbours = r.getNeighbours(i,j,ns).flatten()
                        if any(neighbours.mask): # Eliminate incomplete samples
                            sample_complete = False
                            break
                        sample['input'][k*pixel_count: (k+1)*pixel_count] = neighbours
                except ProviderError:
                    continue
                if sample_complete:
                    if first_sample:
                        self.data = sample
                        first_sample = False
                    else:
                        self.data = np.vstack((self.data, sample))
        
    
    def train(self):
        pass
    


