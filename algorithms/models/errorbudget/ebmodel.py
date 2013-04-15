#!/usr/bin/env python
# -*- coding: utf-8 -*-


#   The module implements multiple resolution comparison
#   introdused in
#   article{pontius2004useful,
#       title={Useful techniques of validation for spatially explicit land-change models},
#       author={Pontius Jr, Robert Gilmore and Huffaker, Diana and Denman, Kevin},
#       journal={Ecological Modelling},
#       volume={179},
#       number={4},
#       pages={445--461},
#       year={2004},
#       publisher={Elsevier}
#   }

import numpy as np
from numpy import ma as ma

from molusce.algorithms.utils import binaryzation, masks_identity

class EBError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


def weightedSum(arr, weights):
    """
    Returns weighted sum of the array's pixels.
    @param arr      The array.
    @param weights  The weights of the array.
    """
    s1 = np.sum(weights * arr)
    s2 = np.sum(weights)
    return s1/s2




class EBudget(object):
    """Error Budget model"""

    def __init__ (self, referenceMap, simulatedMap):
        """
        @param referenceMap     Reference raster
        @param simulatedMap     Simulated raster
        """

        if referenceMap.getBandsCount() + simulatedMap.getBandsCount() !=2:
            raise EBError('The reference and simulated rasters must be 1-band rasters!')
        if not referenceMap.geoDataMatch(simulatedMap):
            raise EBError('Geometries of the reference and simulated rasters are different!')

        statS = simulatedMap.getBandStat(1)
        statR = referenceMap.getBandStat(1)
        self.categories = statR['gradation']
        for s in statS['gradation']:
            if not s in self.categories:
                raise EBError('Categories in the reference and simulated rasters are different!')

        R = referenceMap.getBand(1)
        S = simulatedMap.getBand(1)
        self.shape = R.shape
        R, S = masks_identity(R,S)
        self.W = 1 - np.ma.getmask(R)     # Array for weight
        R = np.ma.filled(R, 0)
        S = np.ma.filled(S, 0)

        # Proportion of category j in pixel n at the beginning resolution of the reference map
        self.Rj = {}
        for j in self.categories:
            self.Rj[j] = 1.0*binaryzation(R, [j])
        # Proportion of category j in pixel n at the beginning resolution of the simulated map
        self.Sj = {}
        for j in self.categories:
            self.Sj[j] = 1.0*binaryzation(S, [j])


    def coarse(self, scale):
        """Coarsen the scale of Rj and Sj.

        @param scale    An integer number is the number of merged raster cells.
        """
        rows, cols = self.shape
        if (rows < scale) or (cols< scale): # Nothing to do
            return

        newRows, newCols = rows/scale, cols/scale

        newW = np.zeros((newRows, newCols))
        newSj, newRj = {}, {}
        for cat in self.categories:
            newSj[cat] = np.zeros((newRows, newCols))
            newRj[cat] = np.zeros((newRows, newCols))
        r = 0
        while r/scale < newRows:
            c = 0
            while c/scale < newCols:
                newW[r/scale, c/scale] = 1.0*np.sum(self.W[r: r+scale, c: c+scale])/(scale*scale)
                for cat in self.categories:
                    S = self.Sj[cat]
                    newSj[cat][r/scale, c/scale] = 1.0*np.sum(S[r: r+scale, c: c+scale]*self.W[r: r+scale, c: c+scale])/(np.sum(self.W[r: r+scale, c: c+scale]))
                    R = self.Rj[cat]
                    newRj[cat][r/scale, c/scale] = 1.0*np.sum(R[r: r+scale, c: c+scale]*self.W[r: r+scale, c: c+scale])/(np.sum(self.W[r: r+scale, c: c+scale]))
                c = c + scale
            r = r + scale

        self.W = newW
        self.Rj = newRj
        self.Sj = newSj
        self.shape = (newRows, newCols)


    def getStat(self, nIter, scale=2):
        '''
        Perform nIter iterations of error budget calculation and rescaling to coarse scale.
        '''
        result = {}
        for i in xrange(nIter):
            result[i] = {'NoNo': self.NoNo(), 'NoMed': self.NoMed(), 'MedMed': self.MedMed(), 'MedPer': self.MedPer(), 'PerPer': self.PerPer()}
            self.coarse(scale)

        return result


    # Proportion correct between the two
    # maps after the predicted map has been adjusted for
    # various levels of information of quantity and/or location.

    def NoNo(self):
        """
        No information about quantity, no information about location
        """
        arr = np.ma.zeros(self.shape)
        size = len(self.categories)
        for j in self.categories:
            arr = arr + np.minimum(self.Rj[j], 1.0/size)
        arr = self.W * arr
        return np.sum(arr)/np.sum(self.W)

    def NoMed(self):
        """
        No information about quantity, medium information about location
        """
        arr = np.ma.zeros(self.shape)
        for j in self.categories:
            S = weightedSum(self.Sj[j], self.W)
            arr = arr + np.minimum(self.Rj[j], S)
        arr = self.W * arr
        return np.sum(arr)/np.sum(self.W)

    def MedMed(self):
        """
        Medium information about quantity, medium information about location
        """
        arr = np.ma.zeros(self.shape)
        for j in self.categories:
            arr = arr + np.minimum(self.Rj[j], self.Sj[j])
        arr = self.W * arr
        return np.sum(arr)/np.sum(self.W)

    def MedPer(self):
        """
        Medium information about quantity, perfect information about location
        """
        arr = 0
        for j in self.categories:
            S = weightedSum(self.Sj[j], self.W)
            R = weightedSum(self.Rj[j], self.W)
            arr = arr + np.minimum(R, S)
        return arr

    def PerPer(self):
        """
        Perfect information about quantity, perfect information about location
        """
        return 1.0











