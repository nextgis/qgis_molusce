# encoding: utf-8

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster, ProviderError

class Sampler(object):
    '''Create training set based on input-output rasers
    '''
    def __init__(self, inputs, output, ns=0):
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
    
    
    def setTrainingData(self, inputs, output, shuffle=True):
        '''
        @param inputs           List of the input rasters.
        @param output           Raster that contains classes to predict.
        @param ns               Neighbourhood size.
        @param shuffle          Perform random shuffle.
        '''
        for r in inputs:
            if not output.isGeoDataMatch(r):
                raise MlpManagerError('Geometries of the inputs and output rasters are different!')
        
        input_vect_len = self.inputVectLen
        output_vect_len = 1
        
        (rows,cols) = (output.getXSize(), output.getYSize())
        
        # i,j  are pixel indexes
        for i in xrange(self.ns, rows - self.ns):         # Eliminate the raster boundary (of (ns)-size width) because
            for j in xrange(self.ns, rows-self.ns):       # the samples are incomplete in that region
                sample = {'input': np.zeros(input_vect_len), 'output': np.zeros(output_vect_len)}
                sample_complete = True # Are the pixels in the neighbourhood defined/unmasked?
                try: 
                    out = output.getNeighbours(i,j,0).flatten() # Get the pixel
                    if any(out.mask): # Eliminate incomplete samples
                        sample_complete = False
                        continue
                    else:
                        sample['output'] = out
                    
                    for (k,raster) in enumerate(inputs):
                        neighbours = raster.getNeighbours(i,j,self.ns).flatten()
                        if any(neighbours.mask): # Eliminate incomplete samples
                            sample_complete = False
                            break
                        pixel_count = raster.getNeighbourhoodSize(self.ns)
                        sample['input'][k*pixel_count: (k+1)*pixel_count] = neighbours
                except ProviderError:
                    continue
                if sample_complete:
                    try:
                        self.data.append(sample)
                    except AttributeError:
                        self.data = [sample]
        if shuffle: 
            np.random.shuffle(self.data)

