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
        self.name = self.Y.getFileName()      
        if self.X.getBandsCount() != 1 or self.Y.getBandsCount() != 1:
            raise CoeffManagerError('The rasters must have one band!')
        self.Y = self.Y.getBand(1)
        self.X = self.X.getBand(1)
        
    def getCorr(self):
        return correlation(self.X, self.Y)
        
    def getCramer(self):
        return cramer(self.X, self.Y)
    
    def getJIU(self):
        return jiu(self.X, self.Y)
        
    def getName(self):
        return self.name
        
    
    


