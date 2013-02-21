# encoding: utf-8

import numpy as np
from numpy import ma as ma

from molusce.algorithms.dataprovider import Raster, ProviderError

class SamplerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


class Sampler(object):
    '''Create training set based on input-output rasters.
    
    A sample is a set of input data for a model and output data that has to be predicted via the model.
    
    input data consists of 2 parts: 
        state is data readed from 1-band raster, this raster contains initaial states (classes).
        factors is list of rasters (multiband probably) that explain transition between states (classes).
    output data is is data readed from 1-band raster, this raster contains final states.
    
    In the simplest case we have pixel-by-pixel model. In such case:
        sample = np.array(
            (pixel_from_state_raster, [pixel_from_factor1, ..., pixel_from_factorN], pixel_from_output_raster), 
            dtype=[('state', float, 1),('factors',  float, N), ('output', float, 1)]
        )
    But we can use moving windows to collect samples, then input data contains several (eg 3x3) pixels for every raster (band).
    For example if we use 1-pixel neighbourhood (3x3 moving windows):
        sample = np.array(
            ( [1-pixel_from_state_raster, ..., 9-pixel_from_state_raster],
              [1-pixel_from_factor1, ..., 9-pixel_from_factor1, ..., 1-pixel_from_factorN..., 9-pixel_from_factorN], 
              pixel_from_output_raster
            ), 
            dtype=[('state', float, 9),('factors',  float, 9*N), ('output', float, 1)]
        )
    '''
    def __init__(self, state, factors, output=None, ns=0):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param output           Raster that contains states (classes) to predict.
        @param ns               Neighbourhood size.
        '''
        
        self.data = None        # Training data
        self.ns = ns
        
        self.outputVecLen = 1                                   # Len of output vector
        self.stateVecLen  = state.getNeighbourhoodSize(self.ns) # Len of the vector of input states
        self.factorVectLen = 0                                  # Length of factor vector
        for raster in factors:
            self.factorVectLen = self.factorVectLen + raster.getNeighbourhoodSize(self.ns)
    
    def get_inputs(self, state, factors, row, col):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        '''
        try:
            state_data = self.get_state(state, row,col)
            if state_data == None: # Eliminate incomplete samples
                return None        
            factors_data = self.get_factors(factors, row,col)
            if factors_data == None: # Eliminate incomplete samples
                return None
        except ProviderError:
            return None
        return np.hstack( (state_data, factors_data) )
    
    def get_factors(self, factors, row, col):
        '''
        Get input sample at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        '''
        sample = np.zeros(self.factorVectLen)
        for (k,raster) in enumerate(factors):
            neighbours = raster.getNeighbours(row,col, self.ns).flatten()
            if any(neighbours.mask): # Eliminate incomplete samples
                return None
            pixel_count = raster.getNeighbourhoodSize(self.ns)
            sample[k*pixel_count: (k+1)*pixel_count] = neighbours
        return sample
    
    
    def get_state(self, state, row, col):
        '''
        Get current state at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        '''
        neighbours = state.getNeighbours(row,col, self.ns).flatten()
        if any(neighbours.mask): # Eliminate incomplete samples
            return None
        return neighbours
    
    def get_output(self, output, row, col):
        '''
        Get output sample at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        '''
        sample = output.getNeighbours(i,j,0).flatten() # Get the pixel
        if any(out.mask): # Eliminate masked samples
            return None
        return out
        
    def setTrainingData(self, state, factors, output, shuffle=True):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param ns               Neighbourhood size.
        @param shuffle          Perform random shuffle.
        '''
        
        for r in factors+[state]:
            if not output.geoDataMatch(r):
                raise SamplerError('Geometries of the inputs and output rasters are different!')
        
        # Approximate sample count:
        band = state.getBand(1)
        nulls  =  band.mask.sum() # Count of NA
        (rows,cols) = (state.getXSize(), state.getYSize())
        pixels = rows * cols - nulls
        
        # Array for samples
        self.data = np.zeros(pixels, dtype=[('state', float, self.stateVecLen),('factors',  float, self.factorVectLen), ('output', float, self.outputVecLen)])
        
        # Real count of the samples
        samples_count = 0
        
        # i,j  are pixel indexes
        for i in xrange(self.ns, rows - self.ns):         # Eliminate the raster boundary (of (ns)-size width) because
            for j in xrange(self.ns, cols - self.ns):     # the samples are incomplete in that region
                try: 
                    out_data = output.getNeighbours(i,j,0).flatten() # Get the pixel
                    if out_data == None:                            # Eliminate masked samples
                        continue
                        
                    state_data = self.get_state(state, i,j)
                    if state_data == None: # Eliminate incomplete samples
                        continue
                    
                    factors_data = self.get_factors(factors, i,j)
                    if factors_data == None: # Eliminate incomplete samples
                        continue

                except ProviderError:
                    continue
                self.data[samples_count] = (state_data, factors_data, out_data)
                samples_count = samples_count + 1
        self.data = self.data[:samples_count]
        
        if shuffle: 
            np.random.shuffle(self.data)

