# encoding: utf-8

import numpy as np
from numpy import ma as ma

from PyQt4.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.utils import masks_identity


class AreaAnalizerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


class AreaAnalyst(QObject):
    '''Generates an output raster, with geometry
    copied from the initial land use map.  The output is a 1-band raster
    with categories corresponding the (r,c) elements of the m-matrix of
    categories transitions, so that if for a given pixel the initial category is r,
    the final category c, and there are m categories, the output pixel will have
    value k = r*m + c
    '''

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal(object)
    logMessage = pyqtSignal(str)

    def __init__(self, first, second=None):
        '''
        @param first        Raster of the first stage (the state before transition).
        @param second       Raster of the second stage (the state after transition).
        '''
        QObject.__init__(self)

        if second != None and (not first.geoDataMatch(second)):
            raise AreaAnalizerError('Geometries of the rasters are different!')

        if first.getBandsCount() != 1:
            raise AreaAnalizerError('First raster mast have 1 band!')

        if second !=None and second.getBandsCount() != 1:
            raise AreaAnalizerError('Second raster mast have 1 band!')

        self.geodata = first.getGeodata()
        self.categories = first.getBandGradation(1)

        if second != None:
            self.categoriesSecond = second.getBandGradation(1)
            first, second = masks_identity(first.getBand(1), second.getBand(1))

        self.first = first
        self.second = second

        if second != None:
            for cat in self.categoriesSecond:
                if cat not in self.categories:
                    raise AreaAnalizerError("List of categories of the first raster doesn't contains a category of the second raster!")

        self.changeMap = None

    def codes(self, initialClass):
        '''
        Get list of possible encodes for initialClass (see 'encode').
        '''
        return [self.encode(initialClass, f) for f in self.categories]

    def decode(self, code):
        '''
        Decode transition (initialClass -> finalClass).
        The procedure is the back operation of "encode" (see encode):
            code = initialClass*m + finalClass,
            the result is tuple of (initialClass, finalClass).
        '''
        m = len(self.categories)
        initialClassIndex = code/m
        finalClassIndex   = code - initialClassIndex*m
        try:
            initClass, finalClass = (self.categories[initialClassIndex], self.categories[finalClassIndex])
        except ValueError:
            raise AreaAnalizerError('The code is not in list!')
        return (initClass, finalClass)


    def encode(self, initialClass, finalClass):
        '''
        Encode transition (initialClass -> finalClass):
            if for a given pixel the initial category is initialClass,
            the final category finalClass, and there are m categories, the output pixel will have
            value k = initialClass*m + finalClass
        '''
        m = len(self.categories)
        return self.categories.index(initialClass) * m + self.categories.index(finalClass)

    def finalCodes(self, initialClass):
        '''
        For given initial category return codes of possible final categories. (see 'encode')
        '''
        return [self.encode(initialClass, c) for c in self.categories]

    def getChangeMap(self):
        if self.changeMap == None:
            self.makeChangeMap()
        return self.changeMap

    def makeChangeMap(self):
        f, s = self.first, self.second
        rows, cols = self.geodata['ySize'], self.geodata['xSize']
        band = np.zeros([rows, cols])
        try:
            self.rangeChanged.emit(self.tr("Creating change map %p%"), rows)
            for i in xrange(rows):
                for j in xrange(cols):
                    if (f.mask.shape == ()) or (not f.mask[i,j]):
                        r = f[i,j]
                        c = s[i,j]
                        band[i, j] = self.encode(r, c)
                self.updateProgress.emit()
            bands = [np.ma.array(data = band, mask = f.mask)]
            raster = Raster()
            raster.create(bands, self.geodata)
            self.changeMap = raster
        finally:
            self.processFinished.emit(raster)
