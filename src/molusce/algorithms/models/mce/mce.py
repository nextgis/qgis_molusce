# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)

import numpy as np
from qgis.PyQt.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.utils import binaryzation
from molusce.molusceutils import PickleQObjectMixin


class MCEError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class MCE(PickleQObjectMixin, QObject):
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    randomConsistencyIndex = {
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
        11: 1.51,
        12: 1.48,
        13: 1.56,
        14: 1.57,
        15: 1.59,
        16: 1.60,
        17: 1.61,
        18: 1.62,
        19: 1.63,
        20: 1.63,
        21: 1.64,
        22: 1.65,
        23: 1.65,
        24: 1.66,
        25: 1.66,
        26: 1.67,
        27: 1.67,
        28: 1.67,
        29: 1.68,
        30: 1.68,
        31: 1.68,
        32: 1.69,
        33: 1.69,
        34: 1.69,
        35: 1.69,
        36: 1.70,
        37: 1.70,
        38: 1.70,
        39: 1.70,
    }

    def __init__(
        self, factors, wMatr, initStateNum, finalStateNum, areaAnalyst
    ):
        """Multicriteria evaluation based on Saaty method. It defines transition probability of two categories (initStateNum, finalStateNum).
        @param factors          List of the factor rasters used for prediction.
        @param wMatr            List of lists -- NxN comparison matrix.
        @param initStateNum     Number of initial state (the state before transition).
        @param finalStateNum    Number of final state (the state after transition).
        """
        super().__init__()

        self.factors = factors
        self.initStateNum = initStateNum
        self.finalStateNum = finalStateNum
        self.areaAnalyst = areaAnalyst

        # Check matrix dimension and factor count, apply normalization
        self.dim = 0
        for f in factors:
            self.dim = self.dim + f.getBandsCount()
            f.normalize(mode="maxmin")
        if self.dim != len(wMatr):
            raise MCEError(
                self.tr(
                    "Matrix size is different from the number of variables!"
                )
            )

        # Check if the matrix is valid
        for i in range(self.dim):
            if len(wMatr[i]) != self.dim:
                raise MCEError(self.tr("The weight matrix is not NxN!"))
        EPSILON = 0.000001  # A small number
        for i in range(self.dim):
            if wMatr[i][i] != 1:
                raise MCEError(self.tr("w[i,i] not equal 1 !"))
            for j in range(i + 1, self.dim):
                if abs(wMatr[i][j] * wMatr[j][i] - 1) > EPSILON:
                    raise MCEError(self.tr("w[i,j] * w[j,i] not equal 1 !"))

        self.wMatr = np.array(wMatr)

        self.weights = None  # Weights of the factors, calculated using wMatr
        # It's a list, the length is self.dim
        # first element is the weight of first band of the first factor and so on:
        # [W_f1, ... weights of 1-st factors ... , W_f2, ... weights of 2-nd factors..., W_fn, ...]

        self.consistency = None  # Consistency ratio of the comparison matrix.

        self.prediction = None  # Raster of the prediction results
        self.confidence = None  # Raster of the results confidence(1 = the maximum confidence, 0 = the least confidence)
        self.transitionPotentials = None  # Dictionary of transition potencial maps: {category1: map1, category2: map2, ...}

    def getConsistency(self):
        if self.consistency is None:
            self.setWeights()
        return self.consistency

    def getConfidence(self):
        return self.confidence

    def getTransitionPotentials(self):
        return self.transitionPotentials

    def getPrediction(self, state, factors=None, calcTransitions=False):
        """Most of the models use factors for prediction, but MCE takes list of factors only once (during the initialization)."""
        self._predict(state, calcTransitions)
        return self.prediction

    def getWeights(self):
        if self.weights is None:
            self.setWeights()
        return self.weights

    def _predict(self, state, calcTransitions=False):
        """Predict the changes."""
        try:
            geodata = state.getGeodata()
            rows, cols = geodata["ySize"], geodata["xSize"]

            self.transitionPotentials = (
                None  # Reset tr.potentials if they exist
            )

            # Get locations where self.initStateNum is occurs
            band = state.getBand(1)
            initStateMask = binaryzation(band, [self.initStateNum])
            mask = band.mask

            # Calculate summary map of factors weights
            # Transition potentials:
            #   current implementation: potential and confidence are equal (two-class implementation)
            # Confidence:
            #   confidence is summary map of factors scaled to 0-100, if current state = self.initState
            #   confidence is 0, if current state != self.initState
            # Prediction:
            #   predicted value is a constant = areaAnalyst.encode(initStateNum, finalStateNum), if current state = self.initState
            #   predicted value is the transition code current_state -> current_state, if current state != self.initState
            confidence = np.zeros((rows, cols), dtype=np.uint8)
            weights = self.getWeights()
            weightNum = 0  # Number of processed weights
            for f in self.factors:
                if not f.geoDataMatch(state):
                    raise MCEError(
                        self.tr(
                            "Geometries of the state and factor rasters are different!"
                        )
                    )
                f.normalize(mode="maxmin")
                for i in range(f.getBandsCount()):
                    band = f.getBand(i + 1)
                    confidence = confidence + (
                        band * weights[weightNum] * 100
                    ).astype(np.uint8)
                    mask = np.ma.mask_or(mask, band.mask)
                    weightNum = weightNum + 1
            confidence = confidence * initStateMask
            prediction = np.copy(state.getBand(1))
            for code in self.areaAnalyst.categories:
                if code != self.initStateNum:
                    prediction[prediction == code] = self.areaAnalyst.encode(
                        code, code
                    )
                else:
                    prediction[prediction == code] = self.areaAnalyst.encode(
                        self.initStateNum, self.finalStateNum
                    )

            predicted_band = np.ma.array(
                data=prediction, mask=mask, dtype=np.uint8
            )
            self.prediction = Raster()
            self.prediction.create([predicted_band], geodata)
            confidence_band = np.ma.array(
                data=confidence, mask=mask, dtype=np.uint8
            )
            self.confidence = Raster()
            self.confidence.create([confidence_band], geodata)

            code = self.areaAnalyst.encode(
                self.initStateNum, self.finalStateNum
            )
            self.transitionPotentials = {code: self.confidence}
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during MCE prediction")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during MCE prediction")
            )
            raise

    def setWeights(self):
        """Calculate the weigths and consistency ratio."""
        # Weights
        w, v = np.linalg.eig(self.wMatr)
        maxW = np.max(w)
        maxInd = list(w).index(maxW)  # Index of the biggest eigenvalue
        maxW = maxW.real
        v = v[:, maxInd]  # The eigen vector
        self.weights = [x.real for x in v]  # Maxtix v can be complex
        self.weights = self.weights / sum(self.weights)

        # Consistency ratio
        if self.dim > 2:
            ci = (maxW - self.dim) / (self.dim - 1)
            try:
                ri = self.randomConsistencyIndex[self.dim]
                self.consistency = ci / ri
            except KeyError:
                self.consistency = -1
        else:
            self.consistency = 0
