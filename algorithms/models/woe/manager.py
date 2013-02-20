# encoding: utf-8

import numpy as np

from algorithms.dataprovider import Raster
from model import woe

class WoeManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class WoeManager(object):
    '''This class gets the data extracted from the UI and
    pass it to woe function, then gets and stores the result.
    '''
    def __init__(self, factors, sites, unit_cell=1):
        '''
        @param factors    List of the pattern rasters used for prediction of point objects (sites).
        @param sites      Binary raster layer consisting of the locations at which the point objects are known to occur.
        @param unit_cell  Method parameter, pixelsize of resampled rasters.
        '''
        self.factors = factors
        self.sites   = sites
        
        
        self.woe = []
        
        if self.sites.getBandsCount() != 1:
            raise WoeManagerError('Sites layer must have one band!')
        
        for fact in self.factors:
            description = {'name': fact.getFileName()}
            for i in range(1, fact.getBandsCount()+1):
                band = fact.getBand(i)
                sites = self.sites.getBand(0)
                weights = woe(band, sites, unit_cell)
                description['band'+str(i)] = weights
            self.woe.append(description)
        
    def getWoe(self):
        return self.woe
        
    
    


