#!/usr/bin/env python

import math
from typing import Optional

import numpy as np
from numpy import ma as ma
from qgis.PyQt.QtCore import *

from molusce.algorithms.models.crosstabs.model import CrossTabError, CrossTable
from molusce.algorithms.utils import masks_identity


class CoeffError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class DependenceCoef(QObject):
    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    X: np.ndarray
    Y: np.ndarray
    expand: bool
    crosstable: Optional[CrossTable]

    def __init__(
        self, X: np.ndarray, Y: np.ndarray, expand: bool = False
    ) -> None:
        """@param band1    First band (numpy masked array)
        @param band2    Second band (numpy masked array)
        @param expand   If the param is True, use union of categories of the bands and compute NxN crosstable
        """
        QObject.__init__(self)

        self.X = X
        self.Y = Y
        self.expand = expand

        self.crosstable = None

    def getCrosstable(self) -> Optional[CrossTable]:
        if self.crosstable is None:
            self.calculateCrosstable()
        return self.crosstable

    def calculateCrosstable(self) -> None:
        try:
            self.rangeChanged.emit("Initialization...", 2)
            self.updateProgress.emit()
            self.crosstable = CrossTable(self.X, self.Y, expand=self.expand)
            self.updateProgress.emit()
            self.__propagateCrossTableSignals()
            self.crosstable.computeCrosstable()
        except CrossTabError as error:
            QMessageBox.warning(
                None, self.tr("Different geometry"), str(error)
            )
            return
        except MemoryError:
            self.errorReport.emit(
                self.tr(
                    "The system is out of memory during cross table calculation"
                )
            )
            raise
        except:
            self.errorReport.emit(
                self.tr(
                    "An unknown error occurs during cross table calculation"
                )
            )
            raise
        finally:
            self.processFinished.emit()

    def correlation(self):
        """Define correlation coefficient of the rasters."""
        x, y = masks_identity(self.X.flatten(), self.Y.flatten())
        x, y = np.ma.compressed(x), np.ma.compressed(y)
        R = np.corrcoef(x, y)
        del x
        del y
        # function np.corrcoef returns array of coefficients
        # R[0][0] = R[1][1] = 1.0 - correlation X--X and Y--Y
        # R[0][1] = R[1][0] - correlation X--Y and Y--X

        return R[0][1]

    def correctness(self, percent=True):
        """% (or count) of correct results"""
        table = self.getCrosstable()
        crosstable = table.getCrosstable()
        rows, cols = table.shape
        if rows != cols:
            raise CoeffError(
                self.tr("The method is applicable for NxN crosstable only!")
            )
        n = table.n
        s = 0.0
        for i in range(rows):
            s = s + crosstable[i][i]

        if percent:
            return 100.0 * s / n
        return s / n

    def cramer(self):
        """Define Cramer's relationship coefficient of the rasters for discrete values
        Coefficient change between [0, 1]
        0 - no dependence
        1 - full connection
        @param X    First raster's array
        @param Y    Second raster's array
        """
        table = self.getCrosstable()
        crosstable = table.getCrosstable()
        rows, cols = table.shape
        t_expect = table.getExpectedTable()

        # Mask T* to prevent division by zero
        t_expect = np.ma.array(t_expect, mask=(t_expect == 0))
        # chi-square coeff = sum((T-T*)^2/T*)
        x2 = np.sum(np.square(crosstable - t_expect) / t_expect)
        # CRAMER CONTINGENCY COEF. = sqrt(chi-square / (total * min(s-1,r-1)))
        # s, r - raster grauations
        Cramer = math.sqrt(x2 / (table.n * min(cols - 1, rows - 1)))

        return Cramer

    def jiu(self):
        """Define Joint Information Uncertainty coef., based on entropy., for discrete values
        Coefficient change between [0, 1]
        0 - no connection
        1 - full connection
        @param X    First raster's array
        @param Y    Second raster's array
        """
        # T, sum_r, sum_s, total, r, s = compute_table(X, Y)
        table = self.getCrosstable()
        T = table.getProbtable()  # Pij = Tij / total
        sum_rows = table.getProbRows()  # Pi. = Ti. / total  i=[0,(r-1)]
        sum_cols = table.getProbCols()  # P.j = T.j / total  j=[0,(s-1)]

        # to calculate the entropy we take the logarithm,
        # logarithm of zero does not exist, so we must mask zero values
        sum_rows = np.compress(sum_rows != 0, sum_rows)
        sum_cols = np.compress(sum_cols != 0, sum_cols)
        # Compute the entropy coeff. of two raster
        H_x = -np.sum(sum_rows * np.log(sum_rows))
        H_y = -np.sum(sum_cols * np.log(sum_cols))
        # Compute the joint entropy coeff.
        T = np.ma.array(T, mask=(T == 0))
        T = np.ma.compressed(T)
        H_xy = -np.sum(T * np.log(T))
        # Compute the Joint Information Uncertainty
        U = 2.0 * ((H_x + H_y - H_xy) / (H_x + H_y))

        return U

    def kappa(self, mode=None):
        """Kappa statistic
        @param X    Raster array.
        @param Y    Raster array.
        @param mode Kappa sttistic to compute:
            mode = None:    classic kappa
            mode = loc:     kappa location
            mode = histo    kappa histogram
        """
        table = self.getCrosstable()
        rows, cols = table.shape
        if rows != cols:
            raise CoeffError(
                self.tr("Kappa is applicable for NxN crosstable only!")
            )
        t_expect = table.getProbtable()
        pa = 0
        for i in range(rows):
            pa = pa + t_expect[i, i]
        prows = table.getProbRows()
        pcols = table.getProbCols()
        pexpect = sum(prows * pcols)
        pmax = sum(np.min([prows, pcols], axis=0))

        if mode is None:
            result = (pa - pexpect) / (1 - pexpect)
        elif mode == "loc":
            result = (pa - pexpect) / (pmax - pexpect)
        elif mode == "histo":
            result = (pmax - pexpect) / (1 - pexpect)
        elif mode == "all":
            result = {
                "loc": (pa - pexpect) / (pmax - pexpect),
                "histo": (pmax - pexpect) / (1 - pexpect),
                "overal": (pa - pexpect) / (1 - pexpect),
            }
        else:
            raise CoeffError(self.tr("Unknown mode of kappa statistics!"))

        return result

    def __propagateCrossTableSignals(self):
        self.crosstable.rangeChanged.connect(
            self.__crosstableProgressRangeChanged
        )
        self.crosstable.updateProgress.connect(
            self.__crosstableProgressChanged
        )
        self.crosstable.crossTableFinished.connect(self.__crosstableFinished)
        self.crosstable.errorReport.connect(self.__crosstableError)

    def __crosstableFinished(self):
        self.crosstable.rangeChanged.disconnect(
            self.__crosstableProgressRangeChanged
        )
        self.crosstable.updateProgress.disconnect(
            self.__crosstableProgressChanged
        )
        self.crosstable.crossTableFinished.disconnect(
            self.__crosstableFinished
        )

    def __crosstableProgressChanged(self):
        self.updateProgress.emit()

    def __crosstableProgressRangeChanged(self, message, maxValue):
        self.rangeChanged.emit(message, maxValue)

    def __crosstableError(self, message):
        self.errorReport.emit(message)
