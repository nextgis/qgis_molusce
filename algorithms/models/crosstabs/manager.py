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
            raise CrossTabManagerError('Geomerties of the rasters are different!')
        
        self.crosstable = CrossTable(initRaster.getBand(1), finalRaster.getBand(1))
    
    def getCrosstable(self):
        return self.crosstable
    
    def getTransitionMatrix(self):
        tab = self.getCrosstable()
        tab = tab.T
        s = 1.0/np.sum(tab, axis=1)
        return tab*s[:,None]
