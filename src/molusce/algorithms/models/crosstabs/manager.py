#!/usr/bin/env python

import numpy as np
from qgis.PyQt.QtCore import *

from molusce.algorithms.models.crosstabs.model import CrossTable


class CrossTabManagerError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg

class CrossTableManager(QObject):
    """Provides statistic information about transitions InitState->FinalState.
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    crossTableFinished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    def __init__(self, initRaster, finalRaster):
        QObject.__init__(self)

        if not initRaster.geoDataMatch(finalRaster):
            raise CrossTabManagerError("Geometries of the raster maps are different!")

        if initRaster.getBandsCount() + finalRaster.getBandsCount() != 2:
            raise CrossTabManagerError("An input raster has more then one band. Use 1-band rasters!")

        self.pixelArea = initRaster.getPixelArea()

        self.crosstable = CrossTable(initRaster.getBand(1), finalRaster.getBand(1))

        self.crosstable.rangeChanged.connect(self.__crosstableProgressRangeChanged)
        self.crosstable.updateProgress.connect(self.__crosstableProgressChanged)
        self.crosstable.crossTableFinished.connect(self.__crosstableFinished)
        self.crosstable.errorReport.connect(self.__crosstableError)

    def __crosstableFinished(self):
        self.crosstable.rangeChanged.disconnect(self.__crosstableProgressRangeChanged)
        self.crosstable.updateProgress.disconnect(self.__crosstableProgressChanged)
        self.crosstable.crossTableFinished.disconnect(self.__crosstableFinished)
        self.crossTableFinished.emit()
    def __crosstableProgressChanged(self):
        self.updateProgress.emit()
    def __crosstableProgressRangeChanged(self, message, maxValue):
        self.rangeChanged.emit(message, maxValue)
    def __crosstableError(self, message):
        self.errorReport.emit(message)

    def computeCrosstable(self):
        try:
            self.crosstable.computeCrosstable()
        except MemoryError:
            self.errorReport.emit(self.tr("The system out of memory during calculation of cross table"))
            raise
        except:
            self.errorReport.emit(self.tr("An unknown error occurs during calculation of cross table"))
            raise

    def getCrosstable(self):
        return self.crosstable

    def getTransitionMatrix(self):
        tab = self.getCrosstable().getCrosstable()
        s = 1.0/np.sum(tab, axis=1)
        return tab*s[:,None]

    def getTransitionStat(self):
        pixelArea = self.pixelArea["area"]
        stat = {"unit": self.pixelArea["unit"]}
        tab = self.getCrosstable()

        initArea = tab.getSumRows()
        initArea = pixelArea * initArea
        initPerc = 100.0 * initArea / sum(initArea)
        stat["init"] = initArea
        stat["initPerc"] = initPerc

        finalArea = tab.getSumCols()
        finalArea = pixelArea * finalArea
        finalPerc = 100.0 * finalArea / sum(finalArea)
        stat["final"] = finalArea
        stat["finalPerc"] = finalPerc

        deltas = finalArea - initArea
        deltasPerc = finalPerc - initPerc
        stat["deltas"] = deltas
        stat["deltasPerc"] = deltasPerc

        return stat


