#!/usr/bin/env python
# QGIS MOLUSCE Plugin
# Copyright (C) 2026  NextGIS
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <https://www.gnu.org/licenses/>.


from pathlib import Path
from typing import Dict, Union

import numpy as np
from qgis.PyQt.QtCore import *

from molusce.algorithms.models.crosstabs.model import CrossTabError, CrossTable


class CrossTabManagerError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class CrossTableManager(QObject):
    """Provides statistic information about transitions InitState->FinalState."""

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    crossTableFinished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    def __init__(self, initRaster, finalRaster):
        QObject.__init__(self)

        if not initRaster.geoDataMatch(finalRaster):
            raise CrossTabManagerError(
                self.tr("Geometries of the raster maps are different!")
            )

        if initRaster.getBandsCount() + finalRaster.getBandsCount() != 2:
            raise CrossTabManagerError(
                self.tr(
                    "An input raster has more then one band. Use 1-band rasters!"
                )
            )

        self.pixelArea = initRaster.getPixelArea()

        try:
            self.crosstable = CrossTable(
                initRaster.getBand(1), finalRaster.getBand(1)
            )
        except CrossTabError as error:
            raise CrossTabManagerError(
                self.tr("Geometries of the input rasters are different!")
            ) from error

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
            self.errorReport.emit(
                self.tr(
                    "The system is out of memory during calculation of cross table"
                )
            )
            raise
        except:
            self.errorReport.emit(
                self.tr(
                    "An unknown error occurs during calculation of cross table"
                )
            )
            raise

    def getCrosstable(self):
        return self.crosstable

    def getTransitionMatrix(self):
        tab = self.getCrosstable().getCrosstable()
        s = 1.0 / np.sum(tab, axis=1)
        return tab * s[:, None]

    def getTransitionStat(self) -> Dict[str, Union[str, np.ndarray]]:
        """
        Calculate and return statistics about transitions between initial and
        final states based on the cross table.

        :returns: A dictionary containing the following keys:
            - "unit": The unit of measurement for the area (e.g., "sq. km.") as a string.
            - "init": A NumPy array of initial areas for each category.
            - "initPerc": A NumPy array of percentages of the initial areas.
            - "final": A NumPy array of final areas for each category.
            - "finalPerc": A NumPy array of percentages of the final areas.
            - "deltas": A NumPy array of differences between final and initial areas.
            - "deltasPerc": A NumPy array of percentage differences between final and initial areas.

        :raises CrossTabManagerError: If the input rasters contain different
            numbers of categories, making it impossible to compute statistics.
        """
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

        try:
            deltas = finalArea - initArea
            deltasPerc = finalPerc - initPerc
            stat["deltas"] = deltas
            stat["deltasPerc"] = deltasPerc
        except ValueError as error:
            quick_help_path = (
                Path(__file__).parents[3] / "doc" / "en" / "QuickHelp.pdf"
            )
            quick_help_url = QUrl.fromLocalFile(
                str(quick_help_path)
            ).toString()
            raise CrossTabManagerError(
                self.tr(
                    "The lists of categories in the input rasters do not match! "
                    "MOLUSCE cannot process rasters with different category lists yet.<br>"
                    "For more details, see the <a href={link}>documentation</a>"
                ).format(link=quick_help_url)
            ) from error

        return stat
