#!/usr/bin/env python


#   The module implements multiple resolution comparison
#   introdused in
#   article{pontius2004useful,
#       title={Useful techniques of validation for spatially explicit land-change models},
#       author={Pontius Jr, Robert Gilmore and Huffaker, Diana and Denman, Kevin},
#       journal={Ecological Modelling},
#       volume={179},
#       number={4},
#       pages={445--461},
#       year={2004},
#       publisher={Elsevier}
#   }

import numpy as np
from numpy import ma as ma
from qgis.PyQt.QtCore import *

from molusce.algorithms.utils import binaryzation, masks_identity


class EBError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


def weightedSum(arr, weights):
    """Returns weighted sum of the array's pixels.
    @param arr      The array.
    @param weights  The weights of the array.
    """
    s1 = np.sum(weights * arr)
    s2 = np.sum(weights)
    return s1 / s2


class EBudget(QObject):
    """Error Budget model"""

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    validationFinished = pyqtSignal(object)
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    def __init__(self, referenceMap, simulatedMap):
        """@param referenceMap     Reference raster
        @param simulatedMap     Simulated raster
        """
        QObject.__init__(self)

        if referenceMap.getBandsCount() + simulatedMap.getBandsCount() != 2:
            raise EBError(
                self.tr(
                    "The reference and simulated rasters must be 1-band rasters!"
                )
            )
        if not referenceMap.geoDataMatch(simulatedMap):
            raise EBError(
                self.tr(
                    "Geometries of the reference and simulated rasters are different!"
                )
            )

        self.categories = referenceMap.getBandGradation(1)
        for s in simulatedMap.getBandGradation(1):
            if s not in self.categories:
                raise EBError(
                    self.tr(
                        "Categories in the reference and simulated rasters are different!"
                    )
                )

        R = referenceMap.getBand(1)
        S = simulatedMap.getBand(1)
        self.shape = R.shape
        R, S = masks_identity(R, S, dtype=np.uint8)

        # Array for weight
        self.W = np.ones(self.shape)
        self.W = self.W - np.ma.getmask(R)

        R = np.ma.filled(R, 0)
        S = np.ma.filled(S, 0)

        # Proportion of category j in pixel n at the beginning resolution of the reference map
        self.Rj = {}
        for j in self.categories:
            self.Rj[j] = 1.0 * binaryzation(R, [j])
        # Proportion of category j in pixel n at the beginning resolution of the simulated map
        self.Sj = {}
        for j in self.categories:
            self.Sj[j] = 1.0 * binaryzation(S, [j])

    def coarse(self, scale):
        """Coarsen the scale of Rj and Sj.

        @param scale    An integer number is the number of merged raster cells.
        """
        rows, cols = self.shape
        if (rows < scale) or (cols < scale):  # Nothing to do
            return

        newRows, newCols = rows // scale, cols // scale
        scale2 = scale * scale

        newW = np.zeros((newRows, newCols))
        newSj, newRj = {}, {}
        for cat in self.categories:
            newSj[cat] = np.zeros((newRows, newCols))
            newRj[cat] = np.zeros((newRows, newCols))
        self.rangeChanged.emit(
            self.tr("An iteration of validation %p%"), newRows
        )
        r = 0
        while r // scale < newRows:
            c = 0
            while c // scale < newCols:
                w = self.W[r : r + scale, c : c + scale]
                sum_w = 1.0 * np.sum(w)
                newW[r // scale, c // scale] = 1.0 * sum_w / scale2
                for cat in self.categories:
                    if sum_w == 0:
                        newSj[cat][r // scale, c // scale] = 0
                        newRj[cat][r // scale, c // scale] = 0
                    else:
                        S = self.Sj[cat]
                        R = self.Rj[cat]
                        newSj[cat][r // scale, c // scale] = (
                            1.0
                            * np.sum(S[r : r + scale, c : c + scale] * w)
                            / sum_w
                        )
                        newRj[cat][r // scale, c // scale] = (
                            1.0
                            * np.sum(R[r : r + scale, c : c + scale] * w)
                            / sum_w
                        )
                c = c + scale
            r = r + scale
            self.updateProgress.emit()

        self.W = newW
        self.Rj = newRj
        self.Sj = newSj
        self.shape = (newRows, newCols)

    def getStat(self, nIter, scale=2):
        """Perform nIter iterations of error budget calculation and rescaling to coarse scale."""
        try:
            result = {}
            for i in range(nIter):
                result[i] = {
                    "NoNo": self.NoNo(),
                    "NoMed": self.NoMed(),
                    "MedMed": self.MedMed(),
                    "MedPer": self.MedPer(),
                    "PerPer": self.PerPer(),
                }
                self.coarse(scale)

        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during validation")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during validation")
            )
            raise
        finally:
            self.validationFinished.emit(result)
        return result

    # Proportion correct between the two
    # maps after the predicted map has been adjusted for
    # various levels of information of quantity and/or location.

    def NoNo(self):
        """No information about quantity, no information about location"""
        arr = np.ma.zeros(self.shape)
        size = len(self.categories)
        for j in self.categories:
            arr = arr + np.minimum(self.Rj[j], 1.0 / size)
        arr = self.W * arr
        return np.sum(arr) / np.sum(self.W)

    def NoMed(self):
        """No information about quantity, medium information about location"""
        arr = np.ma.zeros(self.shape)
        for j in self.categories:
            S = weightedSum(self.Sj[j], self.W)
            arr = arr + np.minimum(self.Rj[j], S)
        arr = self.W * arr
        return np.sum(arr) / np.sum(self.W)

    def MedMed(self):
        """Medium information about quantity, medium information about location"""
        arr = np.ma.zeros(self.shape)
        for j in self.categories:
            arr = arr + np.minimum(self.Rj[j], self.Sj[j])
        arr = self.W * arr
        return np.sum(arr) / np.sum(self.W)

    def MedPer(self):
        """Medium information about quantity, perfect information about location"""
        arr = 0
        for j in self.categories:
            S = weightedSum(self.Sj[j], self.W)
            R = weightedSum(self.Rj[j], self.W)
            arr = arr + np.minimum(R, S)
        return arr

    def PerPer(self):
        """Perfect information about quantity, perfect information about location"""
        return 1.0
