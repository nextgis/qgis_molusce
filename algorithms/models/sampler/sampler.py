# encoding: utf-8

import numpy as np
from numpy import ma as ma

import osgeo.ogr as ogr
import osgeo.osr as osr

import os.path

from PyQt4.QtCore import *

from molusce.algorithms.dataprovider import Raster, ProviderError

class SamplerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


class Sampler(QObject):
    '''Create training set based on input-output rasters.

    A sample is a set of input data for a model and output data that has to be predicted via the model.

    A sample contains:

    coordinates of pixel,
    input data consists of 2 parts:
        state is data that is read from 1-band raster, this raster contains initaial states (categories).
        factors is list of rasters (multiband probably) that explain transition between states (categories).
    output data is read from 1-band raster, this raster contains final states.

    In the simplest case we have pixel-by-pixel model. In such case:
        sample = np.array(
            (pixel_from_state_raster, [pixel_from_factor1, ..., pixel_from_factorN], pixel_from_output_raster),
            dtype=[('state', float, 1),('factors',  float, N), ('output', float, 1)]
        )
    But we can use moving windows to collect samples, then input data contains several (eg 3x3) pixels for every raster (band).
    For example if we use 1-pixel neighbourhood (3x3 moving windows):
        sample = np.array(
            ( [1st-pixel_from_state_raster, ..., 9th-pixel_from_state_raster],
              [1st-pixel_from_factor1, ..., 9th-pixel_from_factor1, ..., 1st-pixel_from_factorN..., 9th-pixel_from_factorN],
              pixel_from_output_raster
            ),
            dtype=[('state', float, 9),('factors',  float, 9*N), ('output', float, 1)]
        )
    '''

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, state, factors, output=None, ns=0):
        '''
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        @param output           Raster that contains states (categories) to predict.
        @param ns               Neighbourhood size.
        '''
        QObject.__init__(self)

        self.data = None        # Training data
        self.proj = None        # Projection of the data coordinates
        self.ns = ns

        self.outputVecLen = 1                                   # Len of output vector
        self.stateVecLen  = state.getNeighbourhoodSize(self.ns) # Len of the vector of input states
        self.factorVectLen = 0                                  # Length of factor vector
        for raster in factors:
            self.factorVectLen = self.factorVectLen + raster.getNeighbourhoodSize(self.ns)

    def get_inputs(self, state, factors, row, col):
        '''
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        '''
        try:
            state_data = self.get_state(state, row,col)
            if state_data == None: # Eliminate incomplete samples
                return None
            factors_data = self.get_factors(factors, row,col)
            if factors_data == None: # Eliminate incomplete samples
                return None
        except ProviderError:
            return None
        return np.hstack( (state_data, factors_data) )

    def get_factors(self, factors, row, col):
        '''
        Get input sample at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        '''
        sample = np.zeros(self.factorVectLen)
        n = 0 # The number of samples item processed
        for (k,raster) in enumerate(factors):
            neighbours = raster.getNeighbours(row,col, self.ns).flatten()

            # Mask neighbours.mask can be boolean array or single boolean => set it as iterable object
            mask = neighbours.mask
            if mask.shape == (): mask = [mask]

            if any(mask): # Eliminate incomplete samples
                return None
            pixel_count = raster.getNeighbourhoodSize(self.ns)
            sample[n: n + pixel_count] = neighbours
            n = n + pixel_count
        return sample

    def get_state(self, state, row, col):
        '''
        Get current state at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        '''
        neighbours = state.getNeighbours(row,col, self.ns).flatten()

        # Mask neighbours.mask can be boolean array or single boolean => set it as iterable object
        mask = neighbours.mask
        if mask.shape == (): mask = [mask]

        if any(mask): # Eliminate incomplete samples
            return None
        return neighbours

    def _getSample(self, state, factors, output, row, col):
        '''
        Get one sample from (row,col) pixel. See params in setTrainingData.
        '''
        data = np.zeros(1, dtype=[('coords', float, 2), ('state', float, self.stateVecLen),('factors',  float, self.factorVectLen), ('output', float, self.outputVecLen)])
        try:
            out_data = output.getPixelFromBand(row, col, band=1)  # Get the pixel
            if out_data == None:                                 # Eliminate masked samples
                return None
            else: data['output'] = out_data

            state_data = self.get_state(state, row,col)
            if state_data == None: # Eliminate incomplete samples
                return None
            else: data['state'] = state_data

            factors_data = self.get_factors(factors, row,col)
            if factors_data == None: # Eliminate incomplete samples
                return None
            else: data['factors'] = factors_data
        except ProviderError:
            return None
        x,y = state.getPixelCoords(col,row)
        data['coords'] = x,y
        return data # (coords, state_data, factors_data, out_data)

    def saveSamples(self, fileName):
        workdir = os.path.dirname(fileName)
        fileName = os.path.splitext(os.path.basename(fileName))[0]

        driver = ogr.GetDriverByName('ESRI Shapefile')
        sr = osr.SpatialReference()
        sr.ImportFromWkt(self.proj)

        ds = driver.CreateDataSource(workdir)
        lyr = ds.CreateLayer(fileName.encode('utf-8'), sr, ogr.wkbPoint)

        fieldnames = ['state' + str(i) for i in range(self.stateVecLen)]
        fieldnames = fieldnames + ['factor' + str(i) for i in range(self.factorVectLen)]
        fieldnames = fieldnames + ['out' + str(i) for i in range(self.outputVecLen)]

        for name in fieldnames:
            field_defn = ogr.FieldDefn(name, ogr.OFTReal)
            if lyr.CreateField ( field_defn ) != 0:
                raise SamplerError("Creating Name field failed!")

        for row in self.data:
            x,y = row['coords']
            if x and y:
                feat = ogr.Feature(lyr.GetLayerDefn())
                if self.stateVecLen>1:
                    for i in range(self.stateVecLen):
                        name = fieldnames[i]
                        r = row['state'][i]
                        feat.SetField(name, r)
                else:
                    name = fieldnames[0]
                    r = row['state']
                    feat.SetField(name, r)
                if self.factorVectLen > 1:
                    for i in range(self.factorVectLen):
                        name = fieldnames[i+self.stateVecLen]
                        r = row['factors'][i]
                        feat.SetField(name, r)
                else:
                    name = fieldnames[self.stateVecLen]
                    r = row['factors']
                    feat.SetField(name, r)
                if self.outputVecLen > 1:
                    for i in range(self.outputVecLen):
                        name = fieldnames[i+self.stateVecLen+self.factorVectLen]
                        r = row['output'][i]
                        feat.SetField(name, r)
                else:
                    name = fieldnames[self.stateVecLen+self.factorVectLen]
                    r = row['output']
                    feat.SetField(name, r)
                pt = ogr.Geometry(ogr.wkbPoint)
                pt.SetPoint_2D(0, x, y)
                feat.SetGeometry(pt)
                if lyr.CreateFeature(feat) != 0:
                    raise SamplerError("Failed to create feature in shapefile!")
                feat.Destroy()
        ds = None

    def setTrainingData(self, state, factors, output, shuffle=True, mode='All', samples=None):
        '''
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        @param ns               Neighbourhood size.
        @param shuffle          Perform random shuffle.
        @param mode             Type of sampling method:
                                    All             Get all pixels
                                    Normal          Get samples. Count of samples in the data=samples.
                                    Balanced        Undersampling of major categories and/or oversampling of minor categories.
        @samples                Sample count of the training data (doesn't used in 'All' mode).
        '''

        for r in factors+[state]:
            if not output.geoDataMatch(r):
                raise SamplerError('Geometries of the inputs and output rasters are different!')
        geodata = state.getGeodata()
        self.proj = geodata['proj']

        # Real count of the samples
        # (if self.ns>0 some samples may be incomplete because a neighbour has NoData value)
        samples_count = 0

        rows, cols = state.getXSize(), state.getYSize()

        if mode == 'All':
            # Approximate sample count:
            band = state.getBand(1)
            nulls  =  band.mask.sum() # Count of NA
            samples = rows * cols - nulls

        # Array for samples
        self.data = np.zeros(samples, dtype=[('coords', float, 2), ('state', float, self.stateVecLen),('factors',  float, self.factorVectLen), ('output', float, self.outputVecLen)])

        if mode == 'All':
            self.rangeChanged.emit(self.tr("Sampling..."), rows - 2*self.ns)
            # i,j  are pixel indexes
            for i in xrange(self.ns, rows - self.ns):         # Eliminate the raster boundary (of (ns)-size width) because
                for j in xrange(self.ns, cols - self.ns):     # the samples are incomplete in that region
                    sample = self._getSample(state, factors, output, i,j)
                    if sample != None:
                        self.data[samples_count] = sample
                        samples_count = samples_count + 1
                self.updateProgress.emit()
            self.data = self.data[:samples_count]   # Crop unused part of the array

        elif mode == 'Normal':
            self.rangeChanged.emit(self.tr("Sampling..."), samples)
            while samples_count< samples:
                row = np.random.randint(rows)
                col = np.random.randint(cols)
                sample = self._getSample(state, factors, output, row,col)
                if sample != None:
                    self.data[samples_count] = sample
                    samples_count = samples_count + 1
                    self.updateProgress.emit()
        elif mode == 'Balanced':
            # Analyze output categories:
            stat = output.getBandStat(1)
            categories = stat['gradation']
            band = output.getBand(1)

            # Select pixels
            average = 1.0*samples / len(categories)

            samples_count = 0
            self.rangeChanged.emit(self.tr("Sampling..."), samples)
            # Get counts[i] samples of "cat" categories
            for i,cat in enumerate(categories):
                # Find indices of "cat"-category pixels
                rows, cols = np.where(band == cat)
                indices = [ (rows[i], cols[i]) for i in xrange(len(cols))]

                # Get samples
                count = 0
                while count< average:
                    index = np.random.randint(len(indices))
                    row, col = indices[index]
                    sample = self._getSample(state, factors, output, row,col)
                    if sample != None:
                        self.data[samples_count] = sample
                        samples_count = samples_count + 1
                        count = count + 1
                        self.updateProgress.emit()
        else:
            raise SamplerError('The mode of sampling is unknown!')

        if shuffle:
            np.random.shuffle(self.data)
        self.processFinished.emit()
