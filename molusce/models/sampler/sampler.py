# encoding: utf-8

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster, ProviderError

class SamplerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


class Sampler(object):
    '''Create training set based on input-output rasters'''
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
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param ns               Neighbourhood size.
        @param shuffle          Perform random shuffle.
        '''
        for r in factors+[state]:
            if not output.geoDataMatch(r):
                raise SamplerError('Geometries of the inputs and output rasters are different!')
        
        
        (rows,cols) = (output.getXSize(), output.getYSize())
        
        # i,j  are pixel indexes
        for i in xrange(self.ns, rows - self.ns):         # Eliminate the raster boundary (of (ns)-size width) because
            for j in xrange(self.ns, cols - self.ns):     # the samples are incomplete in that region
                sample = np.zeros(1, dtype=[('state', float, self.stateVecLen),('factors',  float, self.factorVectLen), ('output', float, self.outputVecLen)])
                try: 
                    out = output.getNeighbours(i,j,0).flatten() # Get the pixel
                    if out == None:                            # Eliminate masked samples
                        continue
                    else:
                        sample['output'] = out           
                        
                    neighbours = self.get_state(state, i,j)
                    if neighbours == None: # Eliminate incomplete samples
                        continue
                    sample['state'] = neighbours
                    
                    neighbours = self.get_factors(factors, i,j)
                    if neighbours == None: # Eliminate incomplete samples
                        continue
                    sample['factors'] = neighbours
                except ProviderError:
                    continue
                if self.data !=None:
                    self.data = np.hstack( (self.data,sample) )
                else:      # This is the first sample
                    self.data = sample
        if shuffle: 
            np.random.shuffle(self.data)

