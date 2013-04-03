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
    with classes corresponding the (r,c) elements of the m-matrix of
    classes transitions, so that if for a given pixel the initial class is r,
    the final class c, and there are m classes, the output pixel will have
    value k = r*m + c
    '''

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal(object)
    logMessage = pyqtSignal(str)

    def __init__(self, first, second):
        '''
        @param first        Raster of the first stage (the state before transition).
        @param second       Raster of the second stage (the state after transition).
        '''
        QObject.__init__(self)


        if not first.geoDataMatch(second):
            raise AreaAnalizerError('Geometries of the rasters are different!')
        if first.getBandsCount() + second.getBandsCount() > 2:
            raise AreaAnalizerError('Rasters mast have 1 band!')

        self.geodata = first.getGeodata()
        statFirst    = first.getBandStat(1)
        statSecong   = second.getBandStat(1)
        self.classes = statFirst['gradation']
        self.classesSecond = statSecong['gradation']

        first, second = masks_identity(first.getBand(1), second.getBand(1))

        self.first = first
        self.second = second

        for cl in self.classesSecond:
            if cl not in self.classes:
                raise AreaAnalizerError("List of classes of the first raster doesn't contains a class of the second raster!")

        self.changeMap = None

    def codes(self, initialClass):
        '''
        Get list of possible encodes for initialClass (see 'encode').
        '''
        return [self.encode(initialClass, f) for f in self.classes]

    def decode(self, code):
        '''
        Decode transition (initialClass -> finalClass).
        The procedure is the back operation of "encode" (see encode):
            code = initialClass*m + finalClass,
            the result is tuple of (initialClass, finalClass).
        '''
        m = len(self.classes)
        initialClassIndex = code/m
        finalClassIndex   = code - initialClassIndex*m
        try:
            initClass, finalClass = (self.classes[initialClassIndex], self.classes[finalClassIndex])
        except ValueError:
            raise AreaAnalizerError('The code is not in list!')
        return (initClass, finalClass)


    def encode(self, initialClass, finalClass):
        '''
        Encode transition (initialClass -> finalClass):
            if for a given pixel the initial class is initialClass,
            the final class finalClass, and there are m classes, the output pixel will have
            value k = initialClass*m + finalClass
        '''
        m = len(self.classes)
        return self.classes.index(initialClass) * m + self.classes.index(finalClass)

    def finalCodes(self, initialClass):
        '''
        For given initial class return codes of possible final classes. (see 'encode')
        '''
        return [self.encode(initialClass, c) for c in self.classes]

    def getChangeMap(self):
        if self.changeMap == None:
            self.makeChangeMap()
        return self.changeMap

    def makeChangeMap(self):
        f, s = self.first, self.second
        rows, cols = self.geodata['ySize'], self.geodata['xSize']
        band = np.zeros([rows, cols])
        self.rangeChanged.emit(self.tr("Creating change map %p%"), rows)
        for i in xrange(rows):
            for j in xrange(cols):
                if not f.mask[i,j]:
                    r = f[i,j]
                    c = s[i,j]
                    band[i, j] = self.encode(r, c)
            self.updateProgress.emit()
        bands = [np.ma.array(data = band, mask = f.mask)]
        raster = Raster()
        raster.create(bands, self.geodata)
        self.processFinished.emit(raster)
        self.changeMap = raster
