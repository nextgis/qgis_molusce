# encoding: utf-8

import numpy as np

from molusce.algorithms.dataprovider import Raster
from model import woe
from molusce.algorithms.utils import get_gradations, binaryzation


class WoeManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class WoeManager(object):
    '''This class gets the data extracted from the UI and
    pass it to woe function, then gets and stores the result.
    '''
    def __init__(self, factors, areaAnalyst, unit_cell=1):
        '''
        @param factors    List of the pattern rasters used for prediction of point objects (sites).
        @param sites      Binary raster layer consisting of the locations at which the point objects are known to occur.
        @param unit_cell  Method parameter, pixelsize of resampled rasters.
        '''
        
        self.factors = factors
        self.changeMap   = areaAnalyst.getChangeMap()
        
        rows, cols = self.changeMap.geodata['ySize'], self.changeMap.geodata['xSize']
        for r in self.factors:
            if not self.changeMap.geoDataMatch(r):
                raise WoeManagerError('Geometries of the input rasters are different!')
        
        if self.changeMap.getBandsCount() != 1:
            raise WoeManagerError('Change map must have one band!')
        
        # Get list of classes from the changeMap raster
        cMap = self.changeMap.getBand(1)
        self.classes = get_gradations(cMap.compressed())
        
        self.woe = {}
        for cl in self.classes:
            sites = binaryzation(cMap, [cl])
            # TODO: reclass factors (continuous factor -> ordinal factor)
            wMap = np.ma.zeros(cMap.shape)
            for fact in factors:
                for i in range(1, fact.getBandsCount()+1):
                    band = fact.getBand(i)
                    weights = woe(band, sites, unit_cell)
                    wMap = wMap + weights
            self.woe[cl]=wMap
    
    def getConfidence(self):
        return self.confidence
    
    def getPrediction(self, state, factors):
        return self.prediction
    
    def getWoe(self):
        return self.woe
    
    def _predict(self, state):
        '''
        Predict changes.
        '''
        
    
