# encoding: utf-8

import numpy as np
from numpy import ma as ma

from PyQt4.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.utils import masks_identity, get_gradations


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
        first, second = masks_identity(first.getBand(1), second.getBand(1))

        self.first = first
        self.second = second

        self.classes = get_gradations(self.first.compressed())
        for cl in get_gradations(self.second.compressed()):
            if cl not in self.classes:
                raise AreaAnalizerError("List of classes of the first raster doesn't contains a class of the second raster!")


    def encode(self, initialClass, finalClass):
        '''
        Encode transition (initialClass -> finalClass):
            if for a given pixel the initial class is initialClass,
            the final class finalClass, and there are m classes, the output pixel will have
            value k = initialClass*m + finalClass
        '''
        m = len(self.classes)
        return self.classes.index(initialClass) * m + self.classes.index(finalClass)


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
        band = [np.ma.array(data = band, mask = f.mask)]
        raster = Raster()
        raster.create(band, self.geodata)
        self.processFinished.emit(raster)
        print "DONE"


