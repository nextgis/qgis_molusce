# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)

from typing import Optional

import numpy as np
from qgis.PyQt.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.correlation.model import (
    CoeffError,
    DependenceCoef,
)
from molusce.algorithms.models.sampler.sampler import Sampler

from . import multinomial_logistic_regression as mlr


class LRError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class LR(QObject):
    """Implements Logistic Regression model definition and calibration
    (maximum liklihood parameter estimation).
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    samplingFinished = pyqtSignal()
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    logreg: mlr.MLR
    sampler: Optional[Sampler]
    data: Optional[np.ndarray]
    maxiter: int

    def __init__(self, ns=0, logreg: Optional[mlr.MLR] = None) -> None:
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

        self.ns = ns  # Neighbourhood size of training rasters.
        self.data = None  # Training data
        self.maxiter = 100  # Maximum of fitting iterations

        self.sampler = None  # Sampler

        # Results of the LR prediction
        self.prediction = None  # Raster of the LR prediction results
        self.confidence = None  # Raster of the LR results confidence (1 = the maximum confidence, 0 = the least confidence)
        self.Kappa = 0  # Kappa value
        self.pseudoR = 0  # Pseudo R-squared (Count) (http://www.ats.ucla.edu/stat/mult_pkg/faq/general/Psuedo_RSquareds.htm)
        self.transitionPotentials = None  # Dictionary of transition potencial maps: {category1: map1, category2: map2, ...}

    def getCoef(self):
        return self.logreg.get_weights().T

    def getConfidence(self):
        return self.confidence

    def getIntercept(self):
        return self.logreg.get_intercept()

    def getKappa(self):
        return self.Kappa

    def getStdErrIntercept(self):
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_stderr_intercept(X)

    def getStdErrWeights(self):
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_stderr_weights(X).T

    def get_PvalIntercept(self):
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_pval_intercept(X)

    def get_PvalWeights(self):
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_pval_weights(X).T

    def getPrediction(self, state, factors, calcTransitions=False):
        self._predict(state, factors, calcTransitions)
        return self.prediction

    def getPseudoR(self):
        return self.pseudoR

    def getTransitionPotentials(self):
        return self.transitionPotentials

    def _outputConfidence(self, input_data):
        """Return confidence (difference between 2 biggest probabilities) of the LR output.
        1 = the maximum confidence, 0 = the least confidence
        """
        out_scl = self.logreg.predict_proba(input_data)[0]
        # Calculate the confidence:
        out_scl.sort()
        return int(100 * (out_scl[-1] - out_scl[-2]))

    def outputTransitions(self, input_data):
        """Return transition potential of the outputs"""
        out_scl = self.logreg.predict_proba(input_data)[0]
        out_scl = [int(100 * x) for x in out_scl]
        result = {}
        for r, v in enumerate(out_scl):
            cat = self.catlist[r]
            result[cat] = v
        return result

    def _predict(self, state, factors, calcTransitions=False):
        """Calculate output and confidence rasters using LR model and input rasters
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        """
        try:
            self.rangeChanged.emit(self.tr("Initialize model %p%"), 1)
            geodata = state.getGeodata()
            rows, cols = geodata["ySize"], geodata["xSize"]
            for r in factors:
                if not state.geoDataMatch(r):
                    raise LRError(
                        self.tr(
                            "Geometries of the input rasters are different!"
                        )
                    )

            self.transitionPotentials = (
                None  # Reset tr.potentials if they exist
            )

            # Normalize factors before prediction:
            for f in factors:
                f.normalize(mode="mean")

            predicted_band = np.zeros([rows, cols], dtype=np.uint8)
            confidence_band = np.zeros([rows, cols], dtype=np.uint8)
            if calcTransitions:
                self.transitionPotentials = {}
                for cat in self.catlist:
                    self.transitionPotentials[cat] = np.zeros(
                        [rows, cols], dtype=np.uint8
                    )

            self.sampler = Sampler(state, factors, ns=self.ns)
            mask = state.getBand(1).mask.copy()
            if mask.shape == ():
                mask = np.zeros([rows, cols], dtype=bool)
            self.updateProgress.emit()
            self.rangeChanged.emit(self.tr("Prediction %p%"), rows)
            for i in range(rows):
                for j in range(cols):
                    if not mask[i, j]:
                        input_data = self.sampler.get_inputs(state, i, j)
                        if input_data is not None:
                            input_data = np.array([input_data])
                            out = self.logreg.predict(input_data)
                            predicted_band[i, j] = out
                            confidence = self._outputConfidence(input_data)
                            confidence_band[i, j] = confidence

                            if calcTransitions:
                                potentials = self.outputTransitions(input_data)
                                for cat in self.catlist:
                                    potential_map = self.transitionPotentials[
                                        cat
                                    ]
                                    potential_map[i, j] = potentials[cat]
                        else:  # Input sample is incomplete => mask this pixel
                            mask[i, j] = True
                self.updateProgress.emit()
            predicted_bands = [
                np.ma.array(data=predicted_band, mask=mask, dtype=np.uint8)
            ]
            confidence_bands = [
                np.ma.array(data=confidence_band, mask=mask, dtype=np.uint8)
            ]

            self.prediction = Raster()
            self.prediction.create(predicted_bands, geodata)
            self.confidence = Raster()
            self.confidence.create(confidence_bands, geodata)

            if calcTransitions:
                for cat in self.catlist:
                    band = [
                        np.ma.array(
                            data=self.transitionPotentials[cat],
                            mask=mask,
                            dtype=np.uint8,
                        )
                    ]
                    self.transitionPotentials[cat] = Raster()
                    self.transitionPotentials[cat].create(band, geodata)
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during LR prediction")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during LR prediction")
            )
            raise
        finally:
            self.processFinished.emit()

    def __propagateSamplerSignals(self):
        self.sampler.rangeChanged.connect(self.__samplerProgressRangeChanged)
        self.sampler.updateProgress.connect(self.__samplerProgressChanged)
        self.sampler.samplingFinished.connect(self.__samplerFinished)

    def __samplerFinished(self):
        self.sampler.rangeChanged.disconnect(
            self.__samplerProgressRangeChanged
        )
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

    def setMaxIter(self, maxiter: int) -> None:
        self.maxiter = maxiter

    def setTrainingData(self):
        state, factors, output, mode, samples = (
            self.state,
            self.factors,
            self.output,
            self.mode,
            self.samples,
        )
        if not self.logreg:
            raise LRError(
                self.tr("You must create a Logistic Regression model before!")
            )

        # Normalize factors before sampling:
        for f in factors:
            f.normalize(mode="mean")

        self.sampler = Sampler(state, factors, output, ns=self.ns)
        self.__propagateSamplerSignals()
        self.sampler.setTrainingData(
            state, output, shuffle=False, mode=mode, samples=samples
        )

        self.data = self.sampler.data
        assert self.data is not None
        self.catlist = np.unique(self.data["output"])

    def train(self):
        assert self.data is not None
        X = np.column_stack((self.data["state"], self.data["factors"]))
        Y = self.data["output"]
        self.labelCodes = np.unique(Y)
        self.logreg.fit(X, Y, maxiter=self.maxiter)
        out = self.logreg.predict(X)
        depCoef = DependenceCoef(np.ma.array(out), np.ma.array(Y), expand=True)
        self.Kappa = depCoef.kappa(mode=None)
        self.pseudoR = depCoef.correctness(percent=False)

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
        except CoeffError as error:
            QMessageBox.warning(
                None,
                self.tr("Model training failed"),
                str(error),
            )
            return
        except LRError as error:
            QMessageBox.warning(
                None,
                self.tr("Missed LR model"),
                str(error),
            )
            return
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during LR training")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during LR training")
            )
            raise
        finally:
            self.finished.emit()
