# encoding: utf-8

# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)

from PyQt4.QtCore import *

import numpy as np

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
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    def __init__(self, ns=0, logreg=None):

        QObject.__init__(self)

        if logreg:
            self.logreg = logreg
        else:
            self.logreg = mlr.MLR()

        self.state = None
        self.factors = None
        self.output = None
        self.mode = "All"
        self.samples = None
        self.catlist = None

        self.ns = ns            # Neighbourhood size of training rasters.
        self.data = None        # Training data
        self.maxiter = 100      # Maximum of fitting iterations

        self.sampler = None     # Sampler

        # Results of the LR prediction
        self.prediction = None  # Raster of the LR prediction results
        self.confidence = None  # Raster of the LR results confidence (1 = the maximum confidence, 0 = the least confidence)
        self.Kappa      = 0     # Kappa value
        self.pseudoR    = 0     # Pseudo R-squared (Count) (http://www.ats.ucla.edu/stat/mult_pkg/faq/general/Psuedo_RSquareds.htm)
        self.transitionPotentials = None # Dictionary of transition potencial maps: {category1: map1, category2: map2, ...}

    def getCoef(self):
        return self.logreg.get_weights().T

    def getConfidence(self):
        return self.confidence

    def getIntercept(self):
        return self.logreg.get_intercept()

    def getKappa(self):
        return self.Kappa

    def getStdErrIntercept(self):
        X = np.column_stack( (self.data['state'], self.data['factors']) )
        return self.logreg.get_stderr_intercept(X)

    def getStdErrWeights(self):
        X = np.column_stack( (self.data['state'], self.data['factors']) )
        return self.logreg.get_stderr_weights(X).T

    def get_PvalIntercept(self):
        X = np.column_stack( (self.data['state'], self.data['factors']) )
        return self.logreg.get_pval_intercept(X)

    def get_PvalWeights(self):
        X = np.column_stack( (self.data['state'], self.data['factors']) )
        return self.logreg.get_pval_weights(X).T

    def getPrediction(self, state, factors, calcTransitions=False):
        self._predict(state, factors, calcTransitions)
        return self.prediction

    def getPseudoR(self):
        return self.pseudoR

    def getTransitionPotentials(self):
        return self.transitionPotentials

    def _outputConfidence(self, input):
        '''
        Return confidence (difference between 2 biggest probabilities) of the LR output.
        1 = the maximum confidence, 0 = the least confidence
        '''
        out_scl = self.logreg.predict_proba(input)[0]
        # Calculate the confidence:
        out_scl.sort()
        return int(100 * (out_scl[-1] - out_scl[-2]) )

    def outputTransitions(self, input):
        '''
        Return transition potential of the outputs
        '''
        out_scl = self.logreg.predict_proba(input)[0]
        out_scl = [int(100 * x) for x in out_scl]
        result = {}
        for r, v in enumerate(out_scl):
            cat = self.catlist[r]
            result[cat] = v
        return result

    def _predict(self, state, factors, calcTransitions=False):
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

            self.transitionPotentials = None    # Reset tr.potentials if they exist

            # Normalize factors before prediction:
            for f in factors:
                f.normalize(mode = 'mean')

            predicted_band  = np.zeros([rows, cols], dtype=np.uint8)
            confidence_band = np.zeros([rows, cols], dtype=np.uint8)
            if calcTransitions:
                self.transitionPotentials = {}
                for cat in self.catlist:
                    self.transitionPotentials[cat] = np.zeros([rows, cols], dtype=np.uint8)

            self.sampler = Sampler(state, factors, ns=self.ns)
            mask = state.getBand(1).mask.copy()
            if mask.shape == ():
                mask = np.zeros([rows, cols], dtype=np.bool)
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

                            if calcTransitions:
                                potentials = self.outputTransitions(input)
                                for cat in self.catlist:
                                    map = self.transitionPotentials[cat]
                                    map[i, j] = potentials[cat]
                        else: # Input sample is incomplete => mask this pixel
                            mask[i, j] = True
                self.updateProgress.emit()
            predicted_bands  = [np.ma.array(data = predicted_band,  mask = mask, dtype=np.uint8)]
            confidence_bands = [np.ma.array(data = confidence_band, mask = mask, dtype=np.uint8)]

            self.prediction = Raster()
            self.prediction.create(predicted_bands, geodata)
            self.confidence = Raster()
            self.confidence.create(confidence_bands, geodata)

            if calcTransitions:
                for cat in self.catlist:
                    band = [np.ma.array(data=self.transitionPotentials[cat], mask=mask, dtype=np.uint8)]
                    self.transitionPotentials[cat] = Raster()
                    self.transitionPotentials[cat].create(band, geodata)
        except MemoryError:
            self.errorReport.emit(self.tr("The system out of memory during LR prediction"))
            raise
        except:
            self.errorReport.emit(self.tr("An unknown error occurs during LR prediction"))
            raise
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

    def setMaxIter(self, maxiter):
        self.maxiter = maxiter

    def setTrainingData(self):
        state, factors, output, mode, samples = self.state, self.factors, self.output, self.mode, self.samples
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
        self.catlist = np.unique(self.data['output'])

    def train(self):
        X = np.column_stack( (self.data['state'], self.data['factors']) )
        Y = self.data['output']
        self.labelCodes = np.unique(Y)
        self.logreg.fit(X, Y, maxiter=self.maxiter)
        out = self.logreg.predict(X)
        depCoef = DependenceCoef(np.ma.array(out), np.ma.array(Y), expand=True)
        self.Kappa = depCoef.kappa(mode=None)
        self.pseudoR = depCoef.correctness(percent = False)

    def setState(self, state):
        self.state = state

    def setFactors(self, factors):
        self.factors = factors

    def setOutput(self, output):
        self.output = output

    def setMode(self, mode):
        self.mode = mode

    def setSamples(self, samples):
        self.samples = samples

    def startTrain(self):
        try:
            self.setTrainingData()
            self.train()
        except MemoryError:
            self.errorReport.emit(self.tr("The system out of memory during LR training"))
            raise
        except:
            self.errorReport.emit(self.tr("An unknown error occurs during LR trainig"))
            raise
        finally:
            self.finished.emit()
