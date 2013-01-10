# encoding: utf-8

'''
Some array utilites
'''

import numpy as np


def reclass( raster, trueList ):
    '''Raster binning.
    
    @param trueList     List of raster values converted into true
    @return raster      Binary raster
    '''
    f = np.vectorize(lambda x: True if x in trueList else False )
    return f(raster)
    
    
    
