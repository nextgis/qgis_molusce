import gc
from os.path import basename

import numpy as np
from qgis.PyQt.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.area_analysis.manager import (
    AreaAnalizerCategoryError,
)
from molusce.algorithms.models.woe.model import WoeError
from molusce.algorithms.utils import binaryzation, masks_identity, reclass
from molusce.molusceutils import PickleQObjectMixin

from .model import woe


def sigmoid(x):
    """Sigmoid function."""

    # Large absolute value of x causes overflows in Exp function.
    # To prevent in we truncate x: the result is almost the same (0 or 1),
    # This is simple approach, but it works ))
    # Numericaly stable implementation of sigmoid see:
    # https://stackoverflow.com/a/64717799
    limit_val = 100

    x = np.maximum(-limit_val, np.minimum(x, limit_val))
    return 1 / (1 + np.exp(-x))


class WoeManagerError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class WoeManager(PickleQObjectMixin, QObject):
    """This class gets the data extracted from the UI and
    pass it to woe function, then gets and stores the result.
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    def __init__(self, factors, areaAnalyst, unit_cell=1, bins=None):
        """@param factors      List of the pattern rasters used for prediction of point objects (sites).
        @param areaAnalyst  AreaAnalyst that contains map of the changes, encodes and decodes category numbers.
        @param unit_cell    Method parameter, pixelsize of resampled rasters.
        @param bins         Dictionary of bins. Bins are binning boundaries that used for reduce count of categories.
                                For example if factors = [f0, f1], then bins could be (for example) {0:[bins for f0], 1:[bins for f1]} = {0:[[10, 100, 250]],1:[[0.2, 1, 1.5, 4]]}.
                                List of list used because a factor can be a multiband raster, we need get a list of bins for every band. For example:
                                factors = [f0, 2-band-factor], bins= {0: [[10, 100, 250]], 1:[[0.2, 1, 1.5, 4], [3, 4, 7]] }
        """
        super().__init__()

        self.factors = factors
        self.analyst = areaAnalyst
        self.changeMap = areaAnalyst.getChangeMap()
        self.bins = bins
        self.unit_cell = unit_cell

        self.prediction = None  # Raster of the prediction results
        self.confidence = None  # Raster of the results confidence(1 = the maximum confidence, 0 = the least confidence)

        if (bins is not None) and (len(self.factors) != len(bins)):
            raise WoeManagerError(
                self.tr("Lengths of bins and factors are different!")
            )

        for r in self.factors:
            if not self.changeMap.geoDataMatch(r):
                raise WoeManagerError(
                    self.tr("Geometries of the input rasters are different!")
                )

        if self.changeMap.getBandsCount() != 1:
            raise WoeManagerError(self.tr("Change map must have one band!"))

        self.geodata = self.changeMap.getGeodata()

        # Denormalize factors if they are normalized
        for r in self.factors:
            r.denormalize()

        # Get list of codes from the changeMap raster
        categories = self.changeMap.getBandGradation(1)

        self.codes = [
            int(c) for c in categories
        ]  # Codes of transitions initState->finalState (see AreaAnalyst.encode)
        self.woe = {}  # Maps of WoE results of every transition code

        self.weights = {}  # Weights of WoE (of raster band code)
        # { # The format is: {Transition_code: {factorNumber1: [list of the weights], factorNumber2: [list of the weights]}, ...}
        #  # for example:
        #   0: {0: {1: [...]}, 1: {1: [...]}},
        #   1: {0: {1: [...]}, 1: {1: [...]}},
        #   2: {0: {1: [...]}, 1: {1: [...]}},
        #   ...
        # }
        #
        self.transitionPotentials = None  # Dictionary of transition potencial maps: {category1: map1, category2: map2, ...}

    def checkBins(self):
        """Check if bins are applicable to the factors"""
        if self.bins is not None:
            for i, factor in enumerate(self.factors):
                factor.denormalize()
                boundary_bin = self.bins[i]
                if (boundary_bin is not None) and (boundary_bin != [None]):
                    for j in range(factor.getBandsCount()):
                        b = boundary_bin[j]
                        tmp = b[:]
                        tmp.sort()
                        if b != tmp:  # Mast be sorted
                            return False
                        b0, bMax = b[0], b[len(b) - 1]
                        bandStat = factor.getBandStat(j + 1)
                        if bandStat["min"] > b0 or bandStat["max"] < bMax:
                            return False
        return True

    def getConfidence(self):
        return self.confidence

    def getPrediction(self, state, factors=None, calcTransitions=False):
        """Most of the models use factors for prediction, but WoE takes list of factors only once (during the initialization)."""
        self._predict(state, calcTransitions)
        return self.prediction

    def getTransitionPotentials(self):
        return self.transitionPotentials

    def getWoe(self):
        return self.woe

    def _predict(self, state, calcTransitions=False):
        """Predict the changes."""
        try:
            self.rangeChanged.emit(self.tr("Initialize model %p%"), 1)

            rows, cols = self.geodata["ySize"], self.geodata["xSize"]
            if not self.changeMap.geoDataMatch(state):
                raise WoeManagerError(
                    self.tr(
                        "Geometries of the state and changeMap rasters are different!"
                    )
                )

            prediction = np.zeros((rows, cols), dtype=np.uint8)
            confidence = np.zeros((rows, cols), dtype=np.uint8)
            mask = np.zeros((rows, cols), dtype=np.byte)

            stateBand = state.getBand(1)

            self.updateProgress.emit()
            self.rangeChanged.emit(self.tr("Prediction %p%"), rows)

            for r in range(rows):
                for c in range(cols):
                    oldMax, currMax = -1000, -1000  # Small numbers
                    indexMax = -1  # Index of Max weight
                    initCat = stateBand[
                        r, c
                    ]  # Init category (state before transition)
                    try:
                        codes = self.analyst.codes(
                            initCat
                        )  # Possible final states
                        for code in codes:
                            try:  # If not all possible transitions are presented in the changeMap
                                transition_map = self.woe[
                                    code
                                ]  # Get WoE map of transition 'code'
                            except KeyError:
                                continue
                            w = transition_map[
                                r, c
                            ]  # The weight in the (r,c)-pixel
                            if w > currMax:
                                indexMax, oldMax, currMax = code, currMax, w
                        prediction[r, c] = indexMax
                        confidence[r, c] = int(
                            100 * (sigmoid(currMax) - sigmoid(oldMax))
                        )
                    except AreaAnalizerCategoryError:
                        mask[r, c] = 1
                self.updateProgress.emit()

            predicted_band = np.ma.array(
                data=prediction, mask=mask, dtype=np.uint8
            )
            self.prediction = Raster()
            self.prediction.create([predicted_band], self.geodata)
            confidence_band = np.ma.array(
                data=confidence, mask=mask, dtype=np.uint8
            )
            self.confidence = Raster()
            self.confidence.create([confidence_band], self.geodata)
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during WOE prediction")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during WoE prediction")
            )
            raise
        finally:
            self.processFinished.emit()

    def train(self):
        """Train the model"""
        self.transitionPotentials = {}
        try:
            iterCount = len(self.codes) * len(self.factors)
            self.rangeChanged.emit(self.tr("Training WoE... %p%"), iterCount)
            changeMap = self.changeMap.getBand(1)
            for code in self.codes:
                sites = binaryzation(changeMap, [code])
                # Reclass factors (continuous factor -> ordinal factor)
                wMap = np.ma.zeros(
                    changeMap.shape
                )  # The map of summary weight of the all factors
                self.weights[
                    code
                ] = {}  # Dictionary for storing wheights of every raster's band
                for k in range(len(self.factors)):
                    fact = self.factors[k]
                    self.weights[code][k] = {}  # Weights of the factor
                    factorW = self.weights[code][k]
                    if self.bins:  # Get bins of the factor
                        boundary_bin = self.bins[k]
                        if (
                            boundary_bin is not None
                        ) and fact.getBandsCount() != len(boundary_bin):
                            raise WoeManagerError(
                                self.tr(
                                    "Count of bins list for multiband factor is't equal to band count!"
                                )
                            )
                    else:
                        boundary_bin = None
                    for i in range(1, fact.getBandsCount() + 1):
                        band = fact.getBand(i)
                        if boundary_bin and boundary_bin[i - 1]:  #
                            band = reclass(band, boundary_bin[i - 1])
                        band, sites = masks_identity(
                            band, sites, dtype=np.uint8
                        )  # Combine masks of the rasters
                        try:
                            woeRes = woe(
                                band, sites, self.unit_cell
                            )  # WoE for the 'code' (initState->finalState) transition and current 'factor'.
                        except WoeError as error:
                            QMessageBox(
                                None,
                                self.tr("Error"),
                                str(error),
                            )
                            return
                        weights = woeRes["map"]
                        wMap = wMap + weights
                        factorW[i] = woeRes["weights"]
                    self.updateProgress.emit()

                # Reclassification finished => set WoE coefficients
                self.woe[code] = (
                    wMap  # WoE for all factors and the transition code.
                )

                # Potentials are WoE map rescaled to 0--100 percents
                band = (sigmoid(wMap) * 100).astype(np.uint8)
                p = Raster()
                p.create([band], self.geodata)
                self.transitionPotentials[code] = p
                gc.collect()
        except MemoryError:
            self.errorReport.emit(
                "The system is out of memory during WoE training"
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during WoE training")
            )
            raise
        finally:
            self.processFinished.emit()

    def weightsToText(self):
        """Format self.weights as text report."""
        if self.weights == {}:
            return ""
        text = ""
        for code in self.codes:
            (initClass, finalClass) = self.analyst.decode(code)
            text = text + self.tr("Transition {} -> {}\n").format(
                int(initClass), int(finalClass)
            )
            try:
                factorW = self.weights[code]
                for factNum, factDict in factorW.items():
                    name = self.factors[factNum].getFileName()
                    name = basename(name)
                    text = text + self.tr("\t factor: {} \n").format(name)
                    for bandNum, bandWeights in factDict.items():
                        weights = [str(w) for w in bandWeights]
                        text = text + self.tr(
                            "\t\t Weights of band {}: {} \n"
                        ).format(bandNum, ", ".join(weights))
            except:
                text = text + self.tr(
                    "W for code {} ({} -> {}) causes error"
                ).format(code, initClass, finalClass)
                raise
        return text
