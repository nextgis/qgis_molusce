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
    

def sizes_equal(X, Y):
    '''
    Define equality dimensions of the two rasters
    @param X    First raster
    @param Y    Second raster
    '''

    return (np.shape(X) == np.shape(Y))

def masks_identity(X, Y):
    '''
    Each raster has a mask. This function verify the identity of masks.
    If the mask is not equal, we have to do both raster mask identical
    by combining masks. Function return updated arrays
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    maskX = X.mask
    maskY = Y.mask
    mask = np.ma.mask_or(maskX, maskY)

    X = np.ma.array(X, mask = mask)
    Y = np.ma.array(Y, mask = mask)
    return X, Y

def get_gradations(band):
    return list(np.unique(np.array(band)))
