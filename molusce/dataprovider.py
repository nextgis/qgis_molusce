# encoding: utf-8

import gdal

import numpy as np
from numpy import ma as ma

from utils import reclass

class ProviderError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class Raster(object):
    def __init__(self, filename):
        # TODO: Get mask values from the raster metadata.
        self.filename = filename
        self.maskVals = None    # List of the "transparent" pixel values
        self.bands = None       # List of the bands (stored as numpy mask array)
        self.geodata = None     # Georeferensing information
        self._read()
        
    
    def binaryzation(self, trueVals, bandNum):
        '''Reclass band bandNum to true/false mode. Set true for pixels from trueVals.'''
        r = self.getBand(bandNum)
        r = reclass(r, trueVals)
        self.setBand(r, bandNum)
    
    def isGeoDataMatch(self, raster):
        '''Return true if RasterSize, Projection and GetGeoTransform of the rasters is matched'''
        for key in self.geodata.keys():
            if self.geodata[key] != raster.geodata[key]:
                return False
        return True
    
    def getBand(self, band):
        return self.bands[band-1]
    
    def getBandsCount(self):
        return len(self.bands)
    
    def getNeighbours(self, row, col, size):
        '''Return subset of the bands -- neighbourhood of the central pixel (row,col)'''
        bcount = self.getBandsCount()
        row_size = 2*size+1 # Length of the neighbourhood square side
        pixel_count = row_size**2 # Count of pixels in the neighbourhood
        neighbours = ma.zeros(pixel_count * bcount)
        for i in range(1,bcount+1):
            band = self.getBand(i)
            neighbourhood = band[row-size:(row+size+1), col-size:(col+size+1)]
            neighbourhood = neighbourhood.flatten()
            if len(neighbourhood) != pixel_count:
                raise ProviderError('Incorrect neighbourhood size or the central pixel lies on the raster boundary.')
            neighbours[(i-1)*pixel_count: (i)*pixel_count] = neighbourhood
        neighbours.shape = (bcount, row_size, row_size)
        return neighbours
        
        
    def getFileName(self):
        return self.filename
        
    def get_dtype(self):
        if self.getBandsCount() != 1:
            raise ProviderError('You can get dtype of the one-band raster only!')
        band = self.getBand(0)
        return band.dtype
    
    def getXSize(self):
        return self.geodata['xSize']
        
    def getYSize(self):
        return self.geodata['ySize']
    
    def setBand(self, raster, bandNum):
        self.bands[bandNum-1] = raster
    
    def setMask(self):
        #TODO: Get mask values from the raster metadata.
        #      Don't use mask now.
        maskVals = []
        
        for i in range(self.getBandsCount()):
            r = self.getBand(i)
            mask = reclass(r, maskVals)
            r = ma.array(data = r, mask=mask)
            self.setBand(r, i)
        
    def _read(self):
        data = gdal.Open( self.filename )
        if data is None:
            raise ProviderError("Can't read the file '%s'" % self.filename)
        
        self.geodata = {}
        self.geodata['xSize'] = data.RasterXSize
        self.geodata['ySize'] = data.RasterYSize
        self.geodata['proj']  = data.GetProjection()
        self.geodata['transform']  = data.GetGeoTransform()
        
        self.bands = []
        for i in range(1, data.RasterCount+1):
            r = data.GetRasterBand(i)
            r = r.ReadAsArray()
            self.bands.append(r)
        self.setMask()
        
        
        
