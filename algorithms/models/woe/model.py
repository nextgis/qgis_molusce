# encoding: utf-8

import numpy as np
from numpy import ma as ma
from collections import namedtuple


from molusce.algorithms.utils import binaryzation, get_gradations


EPSILON = 4*np.finfo(np.float).eps # Small number > 0

Weights = namedtuple('Weights', ['wPlus', 'wMinus'])


class WoeError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

def _binary_woe(factor, sites, unitcell=1):
    '''
    Weight of evidence method (binary form).
    
    @param factor     Binary pattern raster used for prediction of point objects (sites).
    @param sites      Raster layer consisting of the locations at which the point objects are known to occur.
    
    @return (W+, W-)  Tuple of the factor's weights (w+, w-).
    ''' 
    
    # Check rasters type
    if factor.dtype != np.bool:
        raise WoeError('Factor raster must be binary in this mode of the method!')
    if sites.dtype != np.bool:
        raise WoeError('Site raster must be binary in this mode of the method!')
    # Check rasters dimentions
    if factor.shape != sites.shape:
        raise WoeError('Factor and sites rasters have different shapes!')
    # Check masked areas of sites and factors are the same
    if not np.array_equal( factor.mask, sites.mask ):
        raise WoeError('Masked areas of factor and sites rasters are different!')
    

    fm = factor.compressed( )  # masked factor
    sm = sites.compressed ( )  # masked sites
    
    A  = 1.0 * len(fm)/unitcell               # Total map area in unit cells
    B  = 1.0 * len(fm[fm==True])/unitcell     # Total factor area in unit cells
    N  = 1.0 * len(sm[sm==True])              # Count of sites
    
    # Count of sites inside area where the factor occurs:
    siteAndPatten = fm&sm       # Sites inside area where the factor occurs
    Nb = 1.0 * len(siteAndPatten[siteAndPatten==True]) # Count of sites inside factor area 
    
    # Check areas size
    if A == 0:
        raise WoeError('Unmasked area is zero-size!')
    if B == 0:
        raise WoeError('Unmasked area of factor (pattern) is zero-size!')
    if N == 0:
        raise WoeError('Unmasked area of sites is zero-size!')
    if (Nb > N) or (N >= A):
        raise WoeError('Unit cell size is too big for your data!')
    
    pSiteFactor = Nb/N
    pNonSiteFactor = (B - Nb)/(A - N)
    pSiteNonFactor = (N - Nb)/N
    pNonSiteNonFactor = (A - B - N +Nb)/(A - N)
    
    # Add a small number to prevent devision by zero or log(0):
    pSiteFactor = pSiteFactor + EPSILON
    pNonSiteFactor = pNonSiteFactor + EPSILON
    pSiteNonFactor = pSiteNonFactor + EPSILON
    pNonSiteNonFactor = pNonSiteNonFactor + EPSILON

    # Weights
    wPlus  = np.math.log(pSiteFactor/pNonSiteFactor)
    wMinus = np.math.log(pSiteNonFactor/pNonSiteNonFactor)

    return Weights(wPlus, wMinus)
    
def woe(factor, sites, unit_cell=1):
    '''Weight of evidence method (multiclass form).
    
    @param factor     Multiclass pattern array used for prediction of point objects (sites).
    @param sites      Array layer consisting of the locations at which the point objects are known to occur.
    @param unit_cell  Method parameter, pixelsize of resampled rasters.
    
    @return [wMap1, wMap2, ...]   Total weights of each factor.
    '''
    
    result =np.zeros(ma.shape(factor))
    # Get list of classes from the factor raster
    classes = get_gradations(factor.compressed())
    
    weights = [] # list of the weights of evidence
    if len(classes) >= 2:
        for cl in classes:
            fct = binaryzation(factor, [cl])
            weights.append(_binary_woe(fct, sites, unit_cell))
    else:
        raise WoeError('Wrong count of classes in the factor raster!') 
    
    wTotalMin = sum([w[1] for w in weights])
    wMap = [w[0] + wTotalMin - w[1] for w in weights]
    
    # If len(classes) = 2, then [w[0] + wTotalMin - w[1] for w in weights] increases the answer.
    # In this case:
    if len(classes) == 2:
        wMap = [w/2 for w in wMap]
    
    for i,cl in enumerate(classes):
        result[factor==cl] = wMap[i]
    
    result = ma.array(data=result, mask=factor.mask)
    return result
    
    
def contrast(wPlus, wMinus):
    'Weight contrast'
    return wPlus - wMinus
    
    
    
    
    
    



