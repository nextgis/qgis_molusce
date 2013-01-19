# encoding: utf-8

import numpy as np

from molusce.dataprovider import Raster
from molusce.models.correlation.model import correlation, cramer, jiu

class CoeffManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class CoeffManager(object):
    '''This class gets the data extracted from the UI and
    pass it to the model's functions, then gets and stores the result.
    '''
    def __init__(self, raster1, raster2):
        '''
        @param raster1    First raster, which must have one bands
        @param raster2    Second raster, which must have one bands
        '''
        self.X = raster1
        self.Y = raster2
        self.correlation = []
        self.coef = []
        self.name = self.Y.getFileName()      
        if self.X.getBandsCount() != 1 :
            raise CoeffManagerError('The first raster must have one band!')
        self.X = self.X.getBand(1)
        for i in range(1, self.Y.getBandsCount()+1):
            band_y = self.Y.getBand(i)
            self.coef = [self.compute_coeff(self.X, band_y)]
        self.correlation.append(self.coef)
        
    def compute_coeff(self, X, Y):
        return self.getCorr(X, Y), self.getCramer(X, Y), self.getJIU(X, Y)
        
    def getCorr(self, X, Y):
        return correlation(X, Y)
        
    def getCramer(self, X, Y):
        return cramer(X, Y)
    
    def getJIU(self, X, Y):
        return jiu(X, Y)
        
    def getName(self):
        return self.name
    
    def get_correlation(self):
        return self.correlation
        
    
    


