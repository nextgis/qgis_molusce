# encoding: utf-8

import numpy as np

from PyQt4.QtCore import *

from molusce.algorithms.dataprovider import Raster
from model import woe
from molusce.algorithms.utils import binaryzation, masks_identity, reclass


def sigmoid(x):
    return 1/(1+np.exp(-x))

class WoeManagerError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class WoeManager(QObject):
    '''This class gets the data extracted from the UI and
    pass it to woe function, then gets and stores the result.
    '''

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, factors, areaAnalyst, unit_cell=1, bins = None):
        '''
        @param factors      List of the pattern rasters used for prediction of point objects (sites).
        @param areaAnalyst  AreaAnalyst that contains map of the changes, encodes and decodes category numbers.
        @param unit_cell    Method parameter, pixelsize of resampled rasters.
        @param bins         Dictionary of bins. Bins are binning boundaries that used for reduce count of categories.
                                For example if factors = [f0, f1], then bins could be (for example) {0:[bins for f0], 1:[bins for f1]} = {0:[[10, 100, 250]],1:[[0.2, 1, 1.5, 4]]}.
                                List of list used because a factor can be a multiband raster, we need get a list of bins for every band. For example:
                                factors = [f0, 2-band-factor], bins= {0: [[10, 100, 250]], 1:[[0.2, 1, 1.5, 4], [3, 4, 7]] }
        '''

        QObject.__init__(self)

        self.factors = factors
        self.analyst = areaAnalyst
        self.changeMap  = areaAnalyst.getChangeMap()
        self.bins       = bins
        self.unit_cell  = unit_cell

        self.prediction = None
        self.confidence = None

        if (bins != None) and (len(self.factors) != len(bins.keys())):
            raise WoeManagerError('Lengths of bins and factors are different!')

        for r in self.factors:
            if not self.changeMap.geoDataMatch(r):
                raise WoeManagerError('Geometries of the input rasters are different!')

        if self.changeMap.getBandsCount() != 1:
            raise WoeManagerError('Change map must have one band!')

        # Get list of codes from the changeMap raster
        categories = self.changeMap.getBandGradation(1)

        self.codes = [int(c) for c in categories]    # Codes of transitions initState->finalState (see AreaAnalyst.encode)

        self.woe = {}

    def getConfidence(self):
        return self.confidence

    def getPrediction(self, state, factors=None):
        '''
        Most of the models use factors for prediction, but WoE takes list of factors only once (during the initialization).
        '''
        self._predict(state)
        return self.prediction

    def getWoe(self):
        return self.woe

    def _predict(self, state):
        '''
        Predict the changes.
        '''
        self.rangeChanged.emit(self.tr("Initialize model %p%"), 1)
        geodata = self.changeMap.getGeodata()
        rows, cols = geodata['ySize'], geodata['xSize']
        if not self.changeMap.geoDataMatch(state):
            raise WoeManagerError('Geometries of the state and changeMap rasters are different!')

        prediction = np.zeros((rows,cols))
        confidence = np.zeros((rows,cols))
        mask = np.zeros((rows,cols))

        stateBand = state.getBand(1)

        self.updateProgress.emit()
        self.rangeChanged.emit(self.tr("Prediction %p%"), rows)

        for r in xrange(rows):
            for c in xrange(cols):
                oldMax, currMax = -1000, -1000  # Small numbers
                indexMax = -1                   # Index of Max weight
                initCat = stateBand[r,c]        # Init category (state before transition)
                try:
                    codes = self.analyst.codes(initCat)   # Possible final states
                    for code in codes:
                        try: # If not all possible transitions are presented in the changeMap
                            map = self.woe[code]     # Get WoE map of transition 'code'
                        except KeyError:
                            continue
                        w = map[r,c]        # The weight in the (r,c)-pixel
                        if w > currMax:
                            indexMax, oldMax, currMax = code, currMax, w
                    prediction[r,c] = indexMax
                    confidence[r,c] = sigmoid(currMax) - sigmoid(oldMax)
                except ValueError:
                    mask[r,c] = 1
            self.updateProgress.emit()

        predicted_band = np.ma.array(data=prediction, mask=mask)
        self.prediction = Raster()
        self.prediction.create([predicted_band], geodata)
        confidence_band = np.ma.array(data=confidence, mask=mask)
        self.confidence = Raster()
        self.confidence.create([confidence_band], geodata)
        self.processFinished.emit()

    def train(self):
        '''
        Train the model
        '''
        iterCount = len(self.codes)*len(self.factors)
        self.rangeChanged.emit(self.tr("Training WoE... %p%"), iterCount)
        changeMap = self.changeMap.getBand(1)
        for code in self.codes:
            sites = binaryzation(changeMap, [code])
            # Reclass factors (continuous factor -> ordinal factor)
            wMap = np.ma.zeros(changeMap.shape) # The map of summary weight of the all factors
            for k in xrange(len(self.factors)):
                fact = self.factors[k]
                if self.bins: # Get bins of the factor
                    bin = self.bins[k]
                    if (bin != None) and fact.getBandsCount() != len(bin):
                        raise WoeManagerError("Count of bins list for multiband factor is't equal to band count!")
                else: bin = None
                for i in range(1, fact.getBandsCount()+1):
                    band = fact.getBand(i)
                    if bin and bin[i-1]:
                        band = reclass(band, bin[i-1])
                    band, sites = masks_identity(band, sites)   # Combine masks of the rasters
                    weights = woe(band, sites, self.unit_cell)  # WoE for the 'code' (initState->finalState) transition and current 'factor'.
                    wMap = wMap + weights
                self.updateProgress.emit()
            # Reclassification finished => set WoE coefficients
            self.woe[code]=wMap             # WoE for all factors and the transition code.
        self.processFinished.emit()


