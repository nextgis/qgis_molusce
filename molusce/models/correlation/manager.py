# encoding: utf-8

import numpy as np

from molusce.dataprovider import Raster
from model import *
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
        @param raster2    Second raster, which may have few baands
        '''
        self.X = raster1
        self.Y = raster2 
        self.coeff = []      
        
        if self.X.getBandsCount() != 1 or self.Y.getBandsCount() != 1:
            raise CoeffManagerError('The rasters must have one band!')
        description = {'name': self.Y.getFileName()}
        self.Y = self.Y.getBand(0)
        self.X = self.X.getBand(0)
        self.corr   = correlation(self.X, self.Y)
        self.cramer = cramer(self.X, self.Y)
        self.joint  = jiu(self.X, self.Y)
        description['second_raster'] = (self.corr, self.cramer, self.joint)
        self.coeff.append(description)
        
    def getCoeff(self):
        return self.coeff
        
    
    


