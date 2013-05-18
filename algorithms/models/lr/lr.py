# encoding: utf-8

# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)

from PyQt4.QtCore import *

import numpy as np
#from sklearn import linear_model as lm

import multinomial_logistic_regression as mlr

from molusce.algorithms.dataprovider import Raster, ProviderError
from molusce.algorithms.models.sampler.sampler import Sampler
from molusce.algorithms.models.correlation.model import DependenceCoef

class LRError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class LR(QObject):
    """
    Implements Logistic Regression model definition and calibration
    (maximum liklihood parameter estimation).
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    samplingFinished = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, ns=0, logreg=None):

        QObject.__init__(self)

        if logreg:
            self.logreg = logreg
        else:
            self.logreg = mlr.MLR()

        self.ns = ns            # Neighbourhood size of training rasters.
        self.data = None        # Training data

        self.sampler = None     # Sampler

        # Results of the LR prediction
        self.prediction = None  # Raster of the LR prediction results
        self.confidence = None  # Raster of the LR results confidence
        self.accuracy   = None  # Intern accuracy score
        self.Kappa      = 0     #  Kappa value

    def getAccuracy(self):
        return self.accuracy

    def getCoef(self):
        return self.logreg.get_weights().T

    def getConfidence(self):
        return self.confidence

    def getIntercept(self):
        return self.logreg.get_intercept()

    def getKappa(self):
        return self.Kappa

    def getPrediction(self, state, factors):
        self._predict(state, factors)
        return self.prediction

    def _outputConfidence(self, input):
        '''
        Return confidence (difference between 2 biggest probabilities) of the LR output.
        '''
        out_scl = self.logreg.predict_proba(input)[0]
        # Calculate the confidence:
        out_scl.sort()
        return out_scl[-1] - out_scl[-2]

    def _predict(self, state, factors):
        '''
        Calculate output and confidence rasters using LR model and input rasters
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        '''
        try:
            self.rangeChanged.emit(self.tr("Initialize model %p%"), 1)
            geodata = state.getGeodata()
            rows, cols = geodata['ySize'], geodata['xSize']
            for r in factors:
                if not state.geoDataMatch(r):
                    raise LRError('Geometries of the input rasters are different!')

            # Normalize factors before prediction:
            for f in factors:
                f.normalize(mode = 'mean')

            predicted_band  = np.zeros([rows, cols])
            confidence_band = np.zeros([rows, cols])

            self.sampler = Sampler(state, factors, ns=self.ns)
            mask = state.getBand(1).mask.copy()
            if mask.shape == ():
                mask = np.zeros([rows, cols])
            self.updateProgress.emit()
            self.rangeChanged.emit(self.tr("Prediction %p%"), rows)
            for i in xrange(rows):
                for j in xrange(cols):
                    if not mask[i,j]:
                        input = self.sampler.get_inputs(state, i,j)
                        if input != None:
                            input = np.array([input])
                            out = self.logreg.predict(input)
                            predicted_band[i,j] = out
                            confidence = self._outputConfidence(input)
                            confidence_band[i, j] = confidence
                        else: # Input sample is incomplete => mask this pixel
                            mask[i, j] = True
                self.updateProgress.emit()
            predicted_bands  = [np.ma.array(data = predicted_band, mask = mask)]
            confidence_bands = [np.ma.array(data = confidence_band, mask = mask)]

            self.prediction = Raster()
            self.prediction.create(predicted_bands, geodata)
            self.confidence = Raster()
            self.confidence.create(confidence_bands, geodata)
        finally:
            self.processFinished.emit()

    def __propagateSamplerSignals(self):
        self.sampler.rangeChanged.connect(self.__samplerProgressRangeChanged)
        self.sampler.updateProgress.connect(self.__samplerProgressChanged)
        self.sampler.samplingFinished.connect(self.__samplerFinished)

    def __samplerFinished(self):
        self.sampler.rangeChanged.disconnect(self.__samplerProgressRangeChanged)
        self.sampler.updateProgress.disconnect(self.__samplerProgressChanged)
        self.sampler.samplingFinished.disconnect(self.__samplerFinished)
        self.samplingFinished.emit()

    def __samplerProgressRangeChanged(self, message, maxValue):
        self.rangeChanged.emit(message, maxValue)

    def __samplerProgressChanged(self):
        self.updateProgress.emit()

    def save(self):
        pass

    def saveSamples(self, fileName):
        self.sampler.saveSamples(fileName)

    def setTrainingData(self, state, factors, output, mode='All', samples=None):
        '''
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        @param output           Raster that contains categories to predict.
        @param mode             Type of sampling method:
                                    All             Get all pixels
                                    Normal          Get samples. Count of samples in the data=samples.
                                    Balanced        Undersampling of major categories and/or oversampling of minor categories.
        @samples                Sample count of the training data (doesn't used in 'All' mode).
        '''
        if not self.logreg:
            raise LRError('You must create a Logistic Regression model before!')

        # Normalize factors before sampling:
        for f in factors:
            f.normalize(mode = 'mean')

        self.sampler = Sampler(state, factors, output, ns=self.ns)
        self.__propagateSamplerSignals()
        self.sampler.setTrainingData(state, output, shuffle=False, mode=mode, samples=samples)

        outputVecLen  = self.sampler.outputVecLen
        stateVecLen   = self.sampler.stateVecLen
        factorVectLen = self.sampler.factorVectLen
        size = len(self.sampler.data)

        self.data = self.sampler.data

    def train(self):
        X = np.column_stack( (self.data['state'], self.data['factors']) )
        Y = self.data['output']
        self.logreg.fit(X, Y, maxiter=100)
        self.accuracy = 100
        out = self.logreg.predict(X)
        depCoef = DependenceCoef(np.ma.array(out), np.ma.array(Y), expand=True)
        self.Kappa = depCoef.kappa(mode=None)



