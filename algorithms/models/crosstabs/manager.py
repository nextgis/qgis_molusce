#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from molusce.algorithms.models.crosstabs.model  import CrossTable
    
class CrossTabManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg
    
class CrossTableManager(object):

    def __init__(self, initRaster, finalRaster):
        if not initRaster.geoDataMatch(finalRaster):
            raise CrossTabManagerError('Geometries of the raster maps are different!')
        
        self.pixelArea = initRaster.getPixelArea()
        
        self.crosstable = CrossTable(initRaster.getBand(1), finalRaster.getBand(1))
    
    def getCrosstable(self):
        return self.crosstable
    
    def getTransitionMatrix(self):
        tab = self.getCrosstable()
        tab = tab.T
        s = 1.0/np.sum(tab, axis=1)
        return tab*s[:,None]
        
    def getTransitionStat(self):
        pixelArea = self.pixelArea['area']
        stat = {'unit': self.pixelArea['unit']}
        tab = self.getCrosstable()
        
        initArea = tab.compute_sum_rows()
        initArea = pixelArea * initArea
        initPerc = 100.0 * initArea / sum(initArea)
        stat['init'] = initArea
        stat['initPerc'] = initPerc
        
        finalArea = tab.compute_sum_cols()
        finalArea = pixelArea * finalArea
        finalPerc = 100.0 * finalArea / sum(finalArea)
        stat['final'] = finalArea
        stat['finalPerc'] = finalPerc
        
        deltas = finalArea - initArea
        deltasPerc = finalPerc - initPerc
        stat['deltas'] = deltas
        stat['deltasPerc'] = deltasPerc
        
        return stat
