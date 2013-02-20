# encoding: utf-8


# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)



import copy

import numpy as np
from numpy import ma as ma

from algorithms.dataprovider import Raster, ProviderError
from algorithms.models.mlp.model import MLP, sigmoid
from algorithms.models.sampler.sampler import Sampler

class MlpManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class MlpManager(object):
    '''This class gets the data extracted from the UI and
    pass it to multi-layer perceptron, then gets and stores the result.
    '''
    
    # TODO: Perform normalization of inputs
    
    def __init__(self, ns=0, MLP=None):
        self.MLP = MLP
        
        self.layers = None
        if self.MLP:
            self.layers = self.getMlpTopology()
        
        self.ns = ns            # Neighbourhood size of training rasters.
        self.data = None        # Training data
        self.classlist = None   # List of unique output values of the output raster
        self.train_error = None # Error on training set
        self.val_error = None   # Error on validation set
        
        # Results of the MLP prediction
        self.prediction = None  # Raster of the MLP prediction results
        self.confidence = None  # Raster of the MLP results confidence 
        
        # Outputs of the activation function for small and big numbers
        self.sigmax, self.sigmin = sigmoid(100), sigmoid(-100)  # Max and Min of the sigmoid function
        self.sigrange = self.sigmax - self.sigmin               # Range of the sigmoid
    
    def computeMlpError(self, sample):
        '''Get MLP error on the sample'''
        input = np.hstack( (sample['state'], sample['factors']) )
        out = self.getOutput( input )
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
    
    def createMlp(self, state, factors, output, hidden_layers):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param hidden_layers    List of neuron counts in hidden layers.
        @param ns               Neighbourhood size.
        '''
        
        if output.getBandsCount() != 1:
            raise MplManagerError('Output layer must have one band!')
        
        input_neurons = 0
        for raster in [state] + factors:
            input_neurons = input_neurons+ raster.getNeighbourhoodSize(self.ns)
        

        # Output class (neuron) count
        band = output.getBand(1)
        self.classlist = np.unique(band.compressed())
        classes = len(self.classlist)
        
        # set neuron counts in the MLP layers
        self.layers = hidden_layers
        self.layers.insert(0, input_neurons)
        self.layers.append(classes)
        
        self.MLP = MLP(*self.layers)
    
    def getConfidence(self):
        return self.confidence
    
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
        size = self.getOutputVectLen()
        res = np.ones(size) * (self.sigmin)
        ind = np.where(self.classlist==val)
        res[ind] = self.sigmax
        return res
    
    def getMlpTopology(self):
        return self.MLP.shape
    
    def getPrediction(self, state, factors):
        self._predict(state, factors)
        return self.prediction
    
    def getTrainError(self):
        return self.train_error
        
    def getValError(self):
        return self.val_error
    
    def outputConfidence(self, output):
        '''
        Return confidence (difference between 2 biggest values) of the MLP output.
        '''
        # Scale the output to range [0,1]
        out_scl = 1.0 * (output - self.sigmin) / self.sigrange
        
        # Calculate the confidence:
        out_scl.sort()
        return out_scl[-1] - out_scl[-2]
    
    
    def _predict(self, state, factors):
        '''
        Calculate output and confidence rasters using MLP model and input rasters
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        '''
        
        rows, cols = state.geodata['xSize'], state.geodata['ySize']
        for r in factors:
            if not state.geoDataMatch(r):
                raise MlpManagerError('Geometries of the input rasters are different!')
        
        predicted_band  = np.zeros([rows, cols])
        confidence_band = np.zeros([rows, cols])
        
        sampler = Sampler(state, factors, self.ns)
        mask = state.getBand(1).mask
        for i in xrange(rows):
            for j in xrange(cols):
                if not mask[i,j]:
                    input = sampler.get_inputs(state, factors, i,j)
                    if input != None:
                        out = self.getOutput(input)
                        # Get index of the biggest output value as the result
                        biggest = max(out)
                        res = list(out).index(biggest)
                        predicted_band[i, j] = res
                        
                        confidence = self.outputConfidence(out)
                        confidence_band[i, j] = confidence
                    else: # Input sample is incomplete => mask this pixel
                        mask[i, j] = True
        predicted_band  = [np.ma.array(data = predicted_band, mask = mask)]
        confidence_band = [np.ma.array(data = confidence_band, mask = mask)]
        
        self.prediction = Raster()
        self.prediction.create(predicted_band, state.geodata)
        self.confidence = Raster()
        self.confidence.create(confidence_band, state.geodata)

    
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
    
    def setTrainingData(self, state, factors, output, shuffle=True):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param output           Raster that contains classes to predict.
        @param shuffle          Perform random shuffle.
        '''
        if not self.MLP:
            raise MlpManagerError('You must create a MLP before!')
        
        sampler = Sampler(state, factors, output, self.ns)
        sampler.setTrainingData(state, factors, output, shuffle)
        
        outputVecLen  = self.getOutputVectLen()
        stateVecLen   = sampler.stateVecLen
        factorVectLen = sampler.factorVectLen
        size = len(sampler.data)
        
        self.data = np.zeros(size, dtype=[('state', float, stateVecLen), ('factors',  float, factorVectLen), ('output', float, outputVecLen)])
        self.data['state'] = sampler.data['state']
        self.data['factors'] = sampler.data['factors']
        self.data['output'] = [self.getOutputVector(sample['output']) for sample in sampler.data] 
            
    
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
            input = np.hstack( (sample['state'],sample['factors']) )
            self.getOutput( input )
            self.MLP.propagate_backward( sample['output'], lrate, momentum )

