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
    
    
def masks_identity(X, Y):
    '''
    Each raster has a mask. This function verify the identity of masks.
    If the mask is not equal, we have to do both raster mask identical
    by combining masks. Function return updated arrays
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    #form the masks as 1-Dimension arrays
    mask_x = np.ma.array(np.matrix.flatten(X.mask))
    mask_y = np.ma.array(np.matrix.flatten(Y.mask))
    #if there are differences between the mask
    if all(np.equal(mask_x, mask_y))!= True:
        # np.equal -> array: False if !=; True if ==
        #combining masks
        X = np.ma.array(X, mask = mask_y)
        Y = np.ma.array(Y, mask = mask_x)
    return X, Y
