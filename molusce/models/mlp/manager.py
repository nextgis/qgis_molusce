# encoding: utf-8

import copy

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster, ProviderError
from molusce.models.mlp.model import MLP, sigmoid
from molusce.models.sampler.sampler import Sampler

class MlpManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class MlpManager(object):
    '''This class gets the data extracted from the UI and
    pass it to multi-layer perceptron, then gets and stores the result.
    '''
    
    # TODO: Perform normalization of inputs
    
    def __init__(self, MLP=None):
        self.MLP = MLP
        
        self.layers = None
        if self.MLP:
            self.layers = self.getMlpTopology()
        
        self.ns = None          # Neighbourhood size of training rasters.
        self.data = None        # Training data
        self.classlist = None   # List of unique output values of the output raster
        self.train_error = None # Error on training set
        self.val_error = None   # Error on validation set
        
        # Outputs of the activation function for small and big numbers
        self.sigmLimits = (sigmoid(-100), sigmoid(100))
    
    def computeMlpError(self, sample):
        '''Get MLP error on the sample'''
        out = self.getOutput( sample['input'] )
        err = ((sample['output'] - out)**2).sum()/len(out)
        return err
    
    def computePerformance(self, train_indexes, val_ind):
        '''Check errors of training and validation sets
        @param train_indexes     Tuple that contains indexes of the first and last elements of the training set.
        @param val_ind           Tuple that contains indexes of the first and last elements of the validation set.
        '''
        train_error = 0
        train_sampl = train_indexes[1] - train_indexes[0]       # Count of training samples
        for i in range(train_indexes[0], train_indexes[1]):
            train_error = train_error + self.computeMlpError(sample = self.data[i])
        self.setTrainError(train_error/train_sampl)
        
        if val_ind:
            val_error = 0
            val_sampl = val_ind[1] - val_ind[0]
            for i in range(val_ind[0], val_ind[1]):
                val_error = val_error + self.computeMlpError(sample = self.data[i])
            self.setValError(val_error/val_sampl)
    
    def copyWeights(self):
        '''Deep copy of the MLP weights'''
        return copy.deepcopy(self.MLP.weights)
    
    def createMlp(self, inputs, output, hidden_layers, ns=0):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param hidden_layers    List of neuron counts in hidden layers.
        @param ns               Neighbourhood size.
        '''
        
        if output.getBandsCount() != 1:
            raise MplManagerError('Output layer must have one band!')
        
        self.ns = ns
        
        input_neurons = 0
        for raster in inputs:
            input_neurons = input_neurons+ raster.getNeighbourhoodSize(ns)
        

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
    def getOutput(self, input_vector):
        out = self.MLP.propagate_forward( input_vector )
        return out


    
    def getOutputVectLen(self):
        '''Length of input data vector of the MLP'''
        shape = self.getMlpTopology()
        return shape[-1]
    
    def getOutputVector(self, val):
        '''Convert a number val into vector,
        for example, let self.classlist = [1, 3, 4] then
        if val = 1, result = [ 1, -1, -1]
        if val = 3, result = [-1,  1, -1]
        if val = 4, result = [-1, -1,  1]
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
    
    def getTrainError(self):
        return self.train_error
    def getValError(self):
        return self.val_error
    
    def predict(self, inputs):
        '''
        Calculate output raster using MLP model and input rasters
        @param inputs           List of the input rasters.
        '''
        first = inputs[0]           # First input raster
        
        rows, cols = first.geodata['xSize'], first.geodata['ySize']
        for r in inputs:
            if not first.geoDataMatch(r):
                raise SamplerError('Geometries of the inputs and output rasters are different!')
        
        band = np.zeros([rows, cols])
        
        sampler = Sampler(inputs, self.ns)
        mask = first.getBand(1).mask
        for i in xrange(rows):
            for j in xrange(cols):
                if not mask[i,j]:
                    input = sampler.get_input(inputs, i,j)
                    out = self.getOutput(input)
                    if out != None:
                        # Get index of the biggest output value as the result
                        biggest = max(out)
                        res = list(out).index(biggest)
                        band[i, j] = res
                    else: # Input sample is incomplete => mask this pixel
                        mask[i, j] = True
        band = [np.ma.array(data = band, mask = mask)]
        raster = Raster()
        raster.create(band, first.geodata)
        return raster
    
    def readMlp(self):
        pass
    
    def resetErrors(self):
        self.val_error = np.finfo(np.float).max
        self.train_error = np.finfo(np.float).max
    
    def resetMlp(self):
        self.MLP.reset()
        self.resetErrors()
    
    def saveMlp(self):
        pass
    
    def setMlpWeights(self, w):
        '''Set weights of the MLP'''
        self.MLP.weights = w
    
    def setTrainingData(self, inputs, output, shuffle=True):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param shuffle          Perform random shuffle.
        '''
        if not self.MLP:
            raise MlpManagerError('You must create a MLP before!')
        sampler = Sampler(inputs, output, self.ns)
        sampler.setTrainingData(inputs, output, shuffle)
        self.data = [{'input': sample['input'], 'output': self.getOutputVector(sample['output'][0])} for sample in sampler.data] 
    
    def setTrainError(self, error):
        self.train_error = error
    
    def setValError(self, error):
        self.val_error = error
    
    def train(self, epochs, valPercent=20, lrate=0.1, momentum=0.1, continue_train=False):
        '''Perform the training procedure on the MLP and save the best neural net
        @param epoch            Max iteration count.
        @param valPercent       Percent of the validation set.
        @param lrate            Learning rate.
        @param momentum         Learning momentum.
        @param continue_train   If False then it is new training cycle, reset weights training and validation error. If True, then continue training.
        '''
        samples_count = len(self.data)
        val_sampl_count = samples_count*valPercent/100
        apply_validation = True if val_sampl_count>0 else False # Use validation set
        train_sampl_count = samples_count - val_sampl_count
        
        # Set first train_sampl_count as training set, the other as validation set
        train_indexes = (0, train_sampl_count)
        val_indexes = (train_sampl_count, samples_count) if apply_validation else None
        
        if not continue_train: self.resetMlp()
        min_val_error = self.getValError()  # The minimum error that is achieved on the validation set
        last_train_err = self.getTrainError()
        best_weights = self.copyWeights()   # The MLP weights when minimum error that is achieved on the validation set
        
        for epoch in range(epochs):
            self.trainEpoch(train_indexes, lrate, momentum)
            self.computePerformance(train_indexes, val_indexes)
            if apply_validation and (self.getValError() < min_val_error):
                min_val_error = self.getValError()
                last_train_err = self.getTrainError()
                best_weights = self.copyWeights()
        if apply_validation:
            self.setMlpWeights(best_weights)
            self.setValError(min_val_error)
            self.setTrainError(last_train_err)
        
                
    def trainEpoch(self, train_indexes, lrate=0.1, momentum=0.1):
        '''Perform a training epoch on the MLP
        @param train_ind        Tuple of the min&max indexes of training samples in the samples data.
        @param val_ind          Tuple of the min&max indexes of validation samples in the samples data.
        @param lrate            Learning rate.
        @param momentum         Learning momentum.
        '''
        train_sampl = train_indexes[1] - train_indexes[0]
        
        for i in range(train_sampl):
            n = np.random.randint( *train_indexes )
            sample = self.data[n]
            self.getOutput( sample['input'] )
            self.MLP.propagate_backward( sample['output'], lrate, momentum )

