#!/usr/bin/env python

import numpy as np
from qgis.PyQt.QtCore import *

from molusce.algorithms.utils import (
    get_gradations,
    masks_identity,
    sizes_equal,
)


class CrossTabError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class CrossTable(QObject):
    """Class for compute gradations, contingency (cross)table T"""

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    crossTableFinished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    def __init__(self, band1, band2, expand=False):
        """@param band1    First band (numpy masked array)
        @param band2    Second band (numpy masked array)
        @param expand   If the param is True, use union of categories of the bands and compute NxN crosstable
        """
        QObject.__init__(self)

        if not sizes_equal(band1, band2):
            raise CrossTabError("Sizes of rasters are not equal!")

        band1, band2 = masks_identity(band1, band2, dtype=np.uint8)

        self.X = np.ma.compressed(band1).flatten()
        self.Y = np.ma.compressed(band2).flatten()

        # Compute gradations of the bands
        self.graduation_x = get_gradations(self.X)
        self.graduation_y = get_gradations(self.Y)
        if expand:
            self.graduation_x = list(
                set(self.graduation_x + self.graduation_y)
            )
            self.graduation_x.sort()
            self.graduation_y = self.graduation_x

        rows, cols = len(self.graduation_x), len(self.graduation_y)
        self.shape = (rows, cols)

        self._T = None  # Crosstable
        self.n = None  # Count of elements in the crosstable

    def computeCrosstable(self):
        # Compute crosstable
        try:
            self.rangeChanged.emit(self.tr("Initializing Crosstable %p%"), 2)
            self.updateProgress.emit()
            rows, cols = self.shape
            self._T = np.zeros([rows, cols], dtype=int)
            self.n = len(
                self.X
            )  # Count of unmasked elements  (= sum of all elements of the table)
            self.updateProgress.emit()

            N = 1000
            self.rangeChanged.emit(
                self.tr("Computing Crosstable %p%"), self.n / N
            )  # 1/N to prevent too frequency updating
            k = 0
            for i in range(self.n):
                class_num_x = self.graduation_x.index(self.X[i])
                class_num_y = self.graduation_y.index(self.Y[i])
                self._T[class_num_x][class_num_y] += 1
                k = k + 1
                if k == N:
                    k = 0
                    self.updateProgress.emit()
        except MemoryError:
            self.errorReport.emit(
                "The system is out of memory during calculation of cross table"
            )
            raise
        except:
            self.errorReport.emit(
                self.tr(
                    "An unknown error occurs during calculation of cross table"
                )
            )
            raise
        finally:
            self.crossTableFinished.emit()

    def getCrosstable(self):
        if self._T is None:
            self.computeCrosstable()
        return self._T

    def getExpectedProbtable(self):
        """Return expected probabilities table. (if dependencies between X, Y are not present)."""
        t = self.getExpectedTable()
        return t / self.n

    def getExpectedTable(self):
        """Return expected crosstable. (if dependencies between X, Y are not present)."""
        # compute expected table T*
        # creation array : T*ij = (sum_r[i] * sum_s[j])/ total
        crostable = self.getCrosstable()
        rows, cols = crostable.shape
        sum_rows = self.getSumRows()
        sum_cols = self.getSumCols()
        sum_rows = np.tile(np.reshape(sum_rows, (rows, 1)), (1, cols))
        sum_cols = np.tile(sum_cols, (rows, 1))
        return 1.0 * sum_rows * sum_cols / self.n

    def getProbCols(self):
        return 1.0 * self.getSumCols() / self.n

    def getProbRows(self):
        return 1.0 * self.getSumRows() / self.n

    def getProbtable(self):
        """Return probability table of transitions"""
        return 1.0 * self.getCrosstable() / self.n

    def getSumRows(self):
        """This function returns sums in the rows (Ti.)"""
        crosstable = self.getCrosstable()
        return crosstable.sum(axis=1)

    def getSumCols(self):
        """This function returns sums in the cols (T.j)"""
        crosstable = self.getCrosstable()
        return crosstable.sum(axis=0)

    def getTransition(self, fromClass, toClass):
        """Return number of transitions from "fromClass" to "toClass" """
        i = self.graduation_x.index(fromClass)
        j = self.graduation_y.index(toClass)
        crosstable = self.getCrosstable()
        return crosstable[i, j]
