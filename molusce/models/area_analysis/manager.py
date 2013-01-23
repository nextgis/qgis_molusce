# encoding: utf-8

import numpy as np
from numpy import ma as ma

from molusce.dataprovider import Raster
from molusce.utils import masks_identity, get_gradations


class AreaAnalizerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


class AreaAnalyst(object):
    '''Generates an output raster, with geometry
    copied from the initial land use map.  The output is a 1-band raster
    with classes corresponding the (r,c) elements of the m-matrix of
    classes transitions, so that if for a given pixel the initial class is r,
    the final class c, and there are m classes, the output pixel will have
    value k = r*m + c
    '''
    def __init__(self, first, second):
        '''
        @param first        Raster of the first stage (the state before transition).
        @param second       Raster of the second stage (the state after transition).
        '''
        
        if not first.geoDataMatch(second):
            raise AreaAnalizerError('Geometries of the rasters are different!')
        if first.getBandsCount() + second.getBandsCount() > 2:
            raise AreaAnalizerError('Raster mast have 1 band!')
        
        self.geodata = first.getGeodata()
        first, second = masks_identity(first.getBand(1), second.getBand(1))
        
        self.first = first
        self.second = second
        self.classes = get_gradations(self.first.compressed())
        if get_gradations(self.second.compressed()) != self.classes:
            raise AreaAnalizerError('Raster mast have the same classes!')
        
        
    def makeChangeMap(self):
        f, s = self.first, self.second
        rows, cols = self.geodata['xSize'], self.geodata['ySize']
        band = np.zeros([rows, cols])
        m = len(self.classes)
        for i in xrange(rows):
            for j in xrange(cols):
                if not f.mask[i,j]:
                    r = self.classes.index(f[i,j])
                    c = self.classes.index(s[i,j])
                    band[i, j] = r*m + c
        band = [np.ma.array(data = band, mask = f.mask)]
        raster = Raster()
        raster.create(band, self.geodata)
        return raster
    
    
