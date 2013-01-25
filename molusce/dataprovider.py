# encoding: utf-8

import gdal

import numpy as np
from numpy import ma as ma

from utils import reclass

class ProviderError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class FormatConverter(object):
    '''Tarnslates formats between GDAL and numpy data formats'''
    def __init__(self):
        self.dtypes = np.bool, np.int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float, np.float16, np.float32, np.float64
        self.GDT = gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64
        self.dtype2GDT = {
            np.dtype('bool'): gdal.GDT_Byte,
            np.dtype('int'): gdal.GDT_Int32,#!!!
            np.dtype('int8'): gdal.GDT_Int16,
            np.dtype('int16'): gdal.GDT_Int16,
            np.dtype('int32'): gdal.GDT_Int32,
            np.dtype('int64'): gdal.GDT_Int32,#!!!
            np.dtype('uint8'): gdal.GDT_Byte,
            np.dtype('uint16'): gdal.GDT_UInt16,
            np.dtype('uint32'): gdal.GDT_UInt32,
            np.dtype('uint64'): gdal.GDT_UInt32,#!!!
            np.dtype('float'): gdal.GDT_Float64,
            np.dtype('float16'): gdal.GDT_Float32,
            np.dtype('float32'): gdal.GDT_Float32,
            np.dtype('float64'): gdal.GDT_Float64
        }
    

class Raster(object):
    def __init__(self, filename=None):
        # TODO: Get mask values from the raster metadata.
        self.filename = filename
        self.maskVals = None    # List of the "transparent" pixel values
        self.bands = None       # List of the bands (stored as numpy mask array)
        self.geodata = None     # Georeferensing information
        if self.filename: self._read()
        
    
    def binaryzation(self, trueVals, bandNum):
        '''Reclass band bandNum to true/false mode. Set true for pixels from trueVals.'''
        r = self.getBand(bandNum)
        r = reclass(r, trueVals)
        self.setBand(r, bandNum)
    
    def create(self, bands, geodata):
        self.bands = bands
        self.geodata = geodata
    
    def geoDataMatch(self, raster):
        '''Return true if RasterSize, Projection and GetGeoTransform of the rasters are matched'''
        for key in ['xSize', 'ySize', 'proj']:
            if self.geodata[key] != raster.geodata[key]:
                return False
        if not self.geoTransformMatch(raster):
            return False
        return True
        
    def geoTransformMatch(self, raster):
        '''Return True if GetGeoTransform of the rasters are matched, ie:
        the difference of the top left x less then pixel size,
        the difference of the top left y less then pixel size,
        for x and y:
        (difference of the pixel sizes) * (pixel count) < (pixel size),
        rotations are equal.
        '''
        indexes = (156835.0, 90.0, 0.0, 2338905.0, 0.0, -90.0)
        s_cornerX, s_width, s_rot1, s_cornerY, s_rot2, s_height  = self.geodata['transform']
        r_cornerX, r_width, r_rot1, r_cornerY, r_rot2, r_height  = raster.geodata['transform']
        if (s_rot1!=r_rot1) or (s_rot2!=r_rot2):
            return False
        
        dx = abs(s_cornerX - r_cornerX)
        if dx > min(abs(s_width), abs(r_width)):
            return False
        dy = abs(s_cornerY - r_cornerY)
        if dy > min(abs(s_height), abs(r_height)):
            return False
        dw = abs(s_width - r_width)
        if dw * self.geodata['xSize'] > 1.5* min(abs(s_width), abs(r_width)):
            return False 
        dh = abs(s_height - r_height)
        if dh * self.geodata['ySize'] > 1.5* min(abs(s_height), abs(r_height)):
            return False
        
        return True
    
    def getBand(self, band):
        return self.bands[band-1]
    
    def getBandsCount(self):
        if self.bands:
            return len(self.bands)
        else:
            return 0
    
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
    
    def getNeighbourhoodSize(self, ns):
        '''Return pixel count in the neighbourhood of ns size'''
        # pixel count in the 1-band neighbourhood of ns size
        neighbours = (2*ns+1)**2
        return self.getBandsCount() * neighbours
        
    def getFileName(self):
        return self.filename
        
    def get_dtype(self):
        # All bands of the raster have the same dtype now
        band = self.getBand(1)
        return band.dtype
    
    def getGeodata(self):
        return self.geodata
    
    def getXSize(self):
        return self.geodata['xSize']
        
    def getYSize(self):
        return self.geodata['ySize']
    
    #~ def normalize(self):
        #~ '''Rescale all bands of the raster: new mean becames 0, new std becames 1'''
        #~ for i in range(1, self.getBandsCount()+1):
            #~ r = self.getBand(i)
            #~ m = np.mean(r)
            #~ s = np.std(r)
            #~ self.setBand((r-m)/s,i)
    
    def save(self, filename, format="GTiff", rastertype=None):
        driver = gdal.GetDriverByName(format)
        metadata = driver.GetMetadata()
        if metadata.has_key(gdal.DCAP_CREATE) and metadata[gdal.DCAP_CREATE] == "YES":
            if not rastertype:
                dtype = self.get_dtype()
                conv = FormatConverter()
                rastertype = conv.dtype2GDT[dtype]
            xsize, ysize = self.getXSize(), self.getYSize()
            bandcount = self.getBandsCount()
            outRaster = driver.Create(filename, xsize, ysize, bandcount, rastertype)
            geodata = self.getGeodata()
            outRaster.SetProjection(geodata['proj'])
            outRaster.SetGeoTransform(geodata['transform'])
            for i in range(bandcount):
                band = self.getBand(i+1)
                outRaster.GetRasterBand( i + 1 ).WriteArray(band)
            outRaster = None
        else:
          raise ProviderError("Driver %s does not support Create() method!" % format)
          
    
    def setBand(self, raster, bandNum):
        self.bands[bandNum-1] = raster
    
    def setGeoData(self, geodata):
        # Check raster's geometry
        if self.getBandsCount() > 0:
            band = self.getBand(1)
            if band.shape != (geodata['xSize'], geodata['ySize']):
                raise ProviderError('Existing bands dont match new geometry!')

        self.geodata['xSize'] = geodata['xSize']
        self.geodata['ySize'] = geodata['ySize']
        
        
        self.geodata['proj']  = geodata['proj']
        self.geodata['transform']  = geodata['transform']
    
    def setMask(self, maskVals = None):
        #TODO: Get mask values from the raster metadata.
        #      Don't use mask now.
        
        if not maskVals: maskVals = []
        
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
        
        
        
