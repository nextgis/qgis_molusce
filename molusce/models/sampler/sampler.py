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
    def __init__(self, inputs, output=None, ns=0):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param ns               Neighbourhood size.
        '''
        
        self.data = None        # Training data
        self.ns = ns
        
        self.inputVectLen = 0   # Length of input vector
        for raster in inputs:
            self.inputVectLen = self.inputVectLen + raster.getNeighbourhoodSize(self.ns)
    
    def get_input(self, inputs, row, col):
        '''
        Get input sample at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        '''
        sample = np.zeros(self.inputVectLen)
        for (k,raster) in enumerate(inputs):
            neighbours = raster.getNeighbours(row,col, self.ns).flatten()
            if any(neighbours.mask): # Eliminate incomplete samples
                return None
            pixel_count = raster.getNeighbourhoodSize(self.ns)
            sample[k*pixel_count: (k+1)*pixel_count] = neighbours
        return sample
    
    def get_output(self, output, row, col):
        '''
        Get output sample at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        '''
        sample = output.getNeighbours(i,j,0).flatten() # Get the pixel
        if any(out.mask): # Eliminate masked samples
            return None
        return out
        
    def setTrainingData(self, inputs, output, shuffle=True):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param ns               Neighbourhood size.
        @param shuffle          Perform random shuffle.
        '''
        for r in inputs:
            if not output.geoDataMatch(r):
                raise SamplerError('Geometries of the inputs and output rasters are different!')
        
        #input_vect_len = self.inputVectLen
        #output_vect_len = 1
        
        (rows,cols) = (output.getXSize(), output.getYSize())
        
        # i,j  are pixel indexes
        for i in xrange(self.ns, rows - self.ns):         # Eliminate the raster boundary (of (ns)-size width) because
            for j in xrange(self.ns, cols - self.ns):     # the samples are incomplete in that region
                sample = {}
                try: 
                    out = output.getNeighbours(i,j,0).flatten() # Get the pixel
                    if out == None:                             # Eliminate masked samples
                        #sample_complete = False
                        continue
                    else:
                        sample['output'] = out                    
                    neighbours = self.get_input(inputs, i,j)
                    if neighbours == None: # Eliminate incomplete samples
                        continue
                    sample['input'] = neighbours
                except ProviderError:
                    continue
                try:
                    self.data.append(sample)
                except AttributeError:      # This is the first sample
                    self.data = [sample]
        if shuffle: 
            np.random.shuffle(self.data)

