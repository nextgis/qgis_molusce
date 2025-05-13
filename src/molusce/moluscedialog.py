# ******************************************************************************
#
# MOLUSCE
# ---------------------------------------------------------
# Modules for Land Use Change Simulations
#
# Copyright (C) 2012-2013 NextGIS (info@nextgis.org)
#
# This source is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 2 of the License, or (at your option)
# any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# A copy of the GNU General Public License is available on the World Wide Web
# at <http://www.gnu.org/licenses/>. You can also obtain it by writing
# to the Free Software Foundation, 51 Franklin Street, Suite 500 Boston,
# MA 02110-1335 USA.
#
# ******************************************************************************

import datetime
import functools
import gc
import glob
import locale
import os.path
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from qgis.core import *
from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import *

try:
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QTAgg as NavigationToolbar,
    )
except ImportError:
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )

from importlib.util import find_spec

from matplotlib.figure import Figure
from qgis.utils import pluginMetadata

from . import (
    loadedmodelwidget,
    multicriteriaevaluationwidget,
    neuralnetworkwidget,
    weightofevidencewidget,
)
from . import molusceutils as utils
from .algorithms.dataprovider import ProviderError, Raster
from .algorithms.models.area_analysis.manager import (
    AreaAnalizerError,
    AreaAnalyst,
)
from .algorithms.models.correlation.model import CoeffError, DependenceCoef
from .algorithms.models.crosstabs.manager import (
    CrossTableManager,
    CrossTabManagerError,
)
from .algorithms.models.crosstabs.model import CrossTabError
from .algorithms.models.errorbudget.ebmodel import EBError, EBudget
from .algorithms.models.sampler.sampler import SamplerError
from .algorithms.models.serializer.serializer import (
    ModelParams,
    ModelParamsSerializer,
    SerializerError,
)
from .algorithms.models.simulator.sim import Simulator
from .ui.ui_moluscedialogbase import Ui_MolusceDialogBase

scipyMissed = False
if find_spec("scipy") is not None:
    from . import logisticregressionwidget
else:
    scipyMissed = True

QGIS_3_38 = 33800


class MolusceDialog(QDialog, Ui_MolusceDialogBase):
    def __init__(self, iface):
        QDialog.__init__(self)
        self.setupUi(self)

        self.setWindowFlags(
            Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self.iface = iface
        self.modelWidget = None
        self.workThread = QThread()

        # Here we'll store all input rasters and then use this dictionary instead of
        # creating Raster objects each time when we need it. Be careful when processing
        # large rasters, you can out of memory!
        # Dictionary has next struct:
        # {"initial" : Raster(),
        #  "final" : Raster(),
        #  "factors" : {"layerId_1" : Raster(),
        #               "layerId_2" : Raster(),
        #               ...
        #               "layerId_N" : Raster()
        #              },
        #  "bandCount" : 0,
        #  "crosstab" : list,
        #  "model" : object
        #   ....
        # }
        # Layer ids are necessary to handle factors changes (e.g. adding new or removing
        # existing factor)
        self.inputs = dict()

        self.settings = QgsSettings("NextGIS", "MOLUSCE")

        self.grpSampling.setSettings(self.settings)

        self._geometry_matched = False
        self.__updateAnalyticTabs(self.geometry_matched)

        # connect signals and slots
        self.btnSetInitialRaster.clicked.connect(self.setInitialRaster)
        self.btnSetFinalRaster.clicked.connect(self.setFinalRaster)

        self.btnAddFactor.clicked.connect(self.addFactor)
        self.btnAddFactorSeparateVars.clicked.connect(
            self.addFactorSeparateVars
        )

        self.btnRemoveFactor.clicked.connect(self.removeFactor)
        self.btnRemoveFactorSeparateVars.clicked.connect(
            self.removeFactorSeparateVars
        )

        self.btnRemoveAllFactors.clicked.connect(self.removeAllFactors)
        self.btnRemoveAllFactorsSeparateVars.clicked.connect(
            self.removeAllFactorsSeparateVars
        )

        self.btnCheckGeometry.clicked.connect(self.checkGeometry)
        self.btnCheckConsistencySeparateVars.clicked.connect(
            self.checkConsistencySeparateVars
        )

        self.chkAllCorr.stateChanged.connect(self.__toggleCorrLayers)
        self.btnStartCorrChecking.clicked.connect(self.correlationChecking)

        self.btnUpdateStatistics.clicked.connect(
            self.startUpdateStatisticsTable
        )
        self.cmbUnits.currentIndexChanged.connect(self.__drawTransitionStat)
        self.btnCreateChangeMap.clicked.connect(self.createChangeMap)

        self.cmbSamplingMode.currentIndexChanged.connect(self.__modeChanged)
        self.cmbSimulationMethod.currentIndexChanged.connect(
            self.__modelChanged
        )

        self.btnSelectSamples.clicked.connect(self.__selectSamplesOutput)

        self.chkRiskFunction.toggled.connect(self.__toggleLineEdit)
        self.chkRiskValidation.toggled.connect(self.__toggleLineEdit)
        self.chkMonteCarlo.toggled.connect(self.__toggleLineEdit)
        self.chkTransitionPotentials.toggled.connect(self.__toggleLineEdit)

        self.btnSelectRiskFunction.clicked.connect(
            self.__selectSimulationOutput
        )
        self.btnSelectTransitionPrefix.clicked.connect(
            self.__selectSimulationOutput
        )
        self.btnSelectMonteCarlo.clicked.connect(self.__selectSimulationOutput)
        self.btnSelectRiskValidation.clicked.connect(self.createValidationMap)

        self.btnStartSimulation.clicked.connect(self.startSimulation)

        self.btnSelectSimulatedMap.clicked.connect(self.__selectValidationMap)
        self.btnSelectReferenceMap.clicked.connect(self.__selectValidationMap)

        self.btnStartValidation.clicked.connect(self.startValidation)

        self.btnKappaCalc.clicked.connect(self.startKappaValidation)

        self.tabWidget.currentChanged.connect(self.tabChanged)

        self.saveModelBtn.clicked.connect(self.saveModel)
        self.loadModelBtn.clicked.connect(self.loadModel)

        self.manageGui()
        self.logMessage(self.tr("Start logging"))

    @property
    def geometry_matched(self):
        return self._geometry_matched

    @geometry_matched.setter
    def geometry_matched(self, value):
        if value not in [True, False]:
            raise ValueError('"geometry_matched" property must be Boolean!')

        self._geometry_matched = value

    def manageGui(self):
        try:
            self.restoreGeometry(self.settings.value("/ui/geometry"))
        except Exception:
            pass

        self.tabWidget.setCurrentIndex(0)

        self.__populateLayers()
        self.__populateCorrCheckingMet()
        self.__populateUnits()
        self.__populateSamplingModes()
        self.__populateSimulationMethods()
        self.__populateRasterNames()
        self.__populateValidationPlot()

        self.__readSettings()

    def closeEvent(self, e):
        self.settings.setValue("/ui/geometry", self.saveGeometry())

        self.__writeSettings()
        self.inputs = None
        gc.collect()

        QDialog.closeEvent(self, e)

    def setInitialRaster(self):
        try:
            layerName = self.lstLayers.selectedItems()[0].text()
            self.initRasterId = self.lstLayers.selectedItems()[0].data(
                Qt.UserRole
            )
            self.leInitRasterName.setText(layerName)
        except IndexError:
            QMessageBox.warning(
                self,
                self.tr("Missed selected row"),
                self.tr(
                    "Initial raster is not selected. Please specify input data and try again"
                ),
            )
            return
        rx = QRegExp(r"(19|2\d)\d\d")
        rx.indexIn(layerName)
        year = rx.cap()
        self.leInitYear.setText(year)

        layer = utils.getLayerById(self.initRasterId)
        try:
            initRaster = Raster(
                str(layer.source()),
                maskVals=utils.getLayerMaskById(self.initRasterId),
            )
            self.logMessage(self.tr("Set initial layer to %s") % (layerName))
        except MemoryError:
            self.logErrorReport(
                self.tr(
                    "Memory Error occurred (loading raster %s). Perhaps the system is low on memory."
                )
                % (layerName)
            )
            raise

        if initRaster.isCountinues(1):
            QMessageBox.warning(
                self,
                self.tr("Raster must store codes of a nominal variable"),
                self.tr(
                    "The raster has a lot of different values. Does the raster store a nominal variable?"
                ),
            )
            initRaster = None
            self.leInitRasterName.setText("")
            gc.collect()
            return
        self.geometry_matched = False
        self.__updateAnalyticTabs(self.geometry_matched)
        self.inputs["initial"] = initRaster

    def setFinalRaster(self):
        try:
            layerName = self.lstLayers.selectedItems()[0].text()
            self.finalRasterId = self.lstLayers.selectedItems()[0].data(
                Qt.UserRole
            )
            self.leFinalRasterName.setText(layerName)
            # self.leReferenceMapPath.setText(unicode(utils.getLayerById(self.finalRasterId).source()))
        except IndexError:
            QMessageBox.warning(
                self,
                self.tr("Missed selected row"),
                self.tr(
                    "Final raster is not selected. Please specify input data and try again"
                ),
            )
            return
        rx = QRegExp(r"(19|2\d)\d\d")
        rx.indexIn(layerName)
        year = rx.cap()
        self.leFinalYear.setText(year)

        try:
            finalRaster = Raster(
                str(utils.getLayerById(self.finalRasterId).source()),
                utils.getLayerMaskById(self.finalRasterId),
            )
            self.logMessage(self.tr("Set final layer to %s") % (layerName))
        except MemoryError:
            self.logErrorReport(
                self.tr(
                    "Memory Error occurred (loading raster %s). Perhaps the system is low on memory."
                )
                % (layerName)
            )
            raise

        if finalRaster.isCountinues(1):
            QMessageBox.warning(
                self,
                self.tr("Raster must store codes of a nominal variable"),
                self.tr(
                    "The raster has a lot of different values. Does the raster store a nominal variable?"
                ),
            )
            finalRaster = None
            self.leFinalRasterName.setText("")
            gc.collect()
            return
        self.inputs["final"] = finalRaster
        self.geometry_matched = False
        self.__updateAnalyticTabs(self.geometry_matched)

    def addFactor(self):
        layerNames = self.lstLayers.selectedItems()

        if len(layerNames) <= 0:
            QMessageBox.warning(
                self,
                self.tr("Missed selected row"),
                self.tr(
                    "Factor raster is not selected. Please specify input data and try again"
                ),
            )
            return

        for i in layerNames:
            layerName = i.text()

            if len(self.lstFactors.findItems(layerName, Qt.MatchExactly)) > 0:
                return

            item = QListWidgetItem(i)
            layerId = str(item.data(Qt.UserRole))
            self.lstFactors.insertItem(self.lstFactors.count() + 1, item)

            try:
                if "factors" in self.inputs:
                    self.inputs["factors"][layerId] = Raster(
                        str(utils.getLayerById(layerId).source()),
                        utils.getLayerMaskById(layerId),
                    )
                else:
                    d = OrderedDict()
                    d[layerId] = Raster(
                        str(utils.getLayerById(layerId).source()),
                        utils.getLayerMaskById(layerId),
                    )
                    self.inputs["factors"] = d

                self.inputs["bandCount"] = self.__bandCount()

                self.logMessage(self.tr("Added factor layer %s") % (layerName))
            except MemoryError:
                self.logErrorReport(
                    self.tr(
                        "Memory Error occurred (loading raster %s). Perhaps the system is low on memory."
                    )
                    % (layerName)
                )
                QMessageBox.warning(
                    self,
                    self.tr("Memory error"),
                    self.tr(
                        "Memory error occurred. Perhaps the system is low on memory."
                    ),
                )
                raise

        self.geometry_matched = False
        self.__updateAnalyticTabs(self.geometry_matched)
        gc.collect()

    def addFactorSeparateVars(self) -> None:
        layer_names = self.lstLayersSeparateVars.selectedItems()

        if len(layer_names) == 0:
            QMessageBox.warning(
                self,
                self.tr("Missed selected row"),
                self.tr(
                    "Factor raster is not selected. Please specify input data and try again"
                ),
            )
            return

        for layer_name_record in layer_names:
            layer_name = layer_name_record.text()

            if (
                len(
                    self.lstFactorsSeparateVars.findItems(
                        layer_name, Qt.MatchFlag.MatchExactly
                    )
                )
                > 0
            ):
                return

            item = QListWidgetItem(layer_name_record)
            layer_id = str(item.data(Qt.ItemDataRole.UserRole))
            self.lstFactorsSeparateVars.insertItem(
                self.lstFactors.count() + 1, item
            )

            try:
                if "factors_sim" in self.inputs:
                    self.inputs["factors_sim"][layer_id] = Raster(
                        str(utils.getLayerById(layer_id).source()),
                        utils.getLayerMaskById(layer_id),
                    )
                else:
                    factors_sim = OrderedDict()
                    factors_sim[layer_id] = Raster(
                        str(utils.getLayerById(layer_id).source()),
                        utils.getLayerMaskById(layer_id),
                    )
                    self.inputs["factors_sim"] = factors_sim

                self.inputs["bandCount_sim"] = self.__bandCount(sim=True)

                self.logMessage(
                    self.tr("Added factor (sim) layer %s") % (layer_name)
                )
            except MemoryError:
                self.logErrorReport(
                    self.tr(
                        "Memory Error occurred (loading raster %s). Perhaps the system is low on memory."
                    )
                    % (layer_name)
                )
                QMessageBox.warning(
                    self,
                    self.tr("Memory error"),
                    self.tr(
                        "Memory error occurred. Perhaps the system is low on memory."
                    ),
                )
                self.consistency_sim_checked = False
                return

        self.consistency_sim_checked = False

    def removeFactor(self):
        layerNames = self.lstFactors.selectedItems()

        if len(layerNames) <= 0:
            QMessageBox.warning(
                self,
                self.tr("Missed selected row"),
                self.tr(
                    "Factor raster is not selected. Please specify it and try again"
                ),
            )
            return

        for i in layerNames:
            layerId = str(i.data(Qt.UserRole))
            layerName = i.text()

            self.lstFactors.takeItem(self.lstFactors.row(i))

            del self.inputs["factors"][layerId]
            gc.collect()

            if self.inputs["factors"] == {}:
                del self.inputs["factors"]
                del self.inputs["bandCount"]

                self.geometry_matched = False
                self.__updateAnalyticTabs(self.geometry_matched)
            else:
                self.inputs["bandCount"] = self.__bandCount()

            self.logMessage(self.tr("Removed factor layer %s") % (layerName))
            gc.collect()

    def removeFactorSeparateVars(self) -> None:
        layer_names = self.lstFactorsSeparateVars.selectedItems()

        if len(layer_names) == 0:
            QMessageBox.warning(
                self,
                self.tr("Missed selected row"),
                self.tr(
                    "Factor raster is not selected. Please specify it and try again"
                ),
            )
            return

        for layer_name_record in layer_names:
            layer_id = str(layer_name_record.data(Qt.ItemDataRole.UserRole))
            layer_name = layer_name_record.text()

            self.lstFactorsSeparateVars.takeItem(
                self.lstFactorsSeparateVars.row(layer_name_record)
            )

            del self.inputs["factors_sim"][layer_id]
            gc.collect()

            if self.inputs["factors_sim"] == {}:
                del self.inputs["factors_sim"]
                del self.inputs["bandCount_sim"]

                self.consistency_sim_checked = False
            else:
                self.inputs["bandCount_sim"] = self.__bandCount(sim=True)

            self.logMessage(
                self.tr("Removed factor (sim) layer %s") % (layer_name)
            )

    def removeAllFactors(self):
        self.lstFactors.clear()
        try:
            del self.inputs["factors"]
            del self.inputs["bandCount"]
            gc.collect()
        except KeyError:
            pass

        self.geometry_matched = False
        self.__updateAnalyticTabs(self.geometry_matched)
        self.logMessage(self.tr("Factors list cleared"))

    def removeAllFactorsSeparateVars(self) -> None:
        self.lstFactorsSeparateVars.clear()
        try:
            del self.inputs["factors_sim"]
            del self.inputs["bandCount_sim"]
        except KeyError:
            pass

        self.consistency_sim_checked = False
        self.logMessage(self.tr("Factors list (sim) cleared"))

    def checkGeometry(self):
        if not utils.checkFactors(self.inputs):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Factors rasters is not set. Please specify them and try again"
                ),
            )
            return
        if not utils.checkInputRasters(self.inputs):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Initial or final raster is not set. Please specify input data and try again"
                ),
            )
            return

        initRaster = self.inputs["initial"]
        for _k, v in self.inputs["factors"].items():
            if not initRaster.geoDataMatch(v):
                QMessageBox.warning(
                    self,
                    self.tr("Different geometry"),
                    self.tr(
                        "Geometries of the initial raster and raster '{}' are different!"
                    ).format(v.getFileName()),
                )
                return
        if not initRaster.geoDataMatch(self.inputs["final"]):
            QMessageBox.warning(
                self,
                self.tr("Different geometry"),
                self.tr(
                    "Geometries of the initial raster and final raster are different!"
                ),
            )
            return
        QMessageBox.warning(
            self,
            self.tr("Geometry is matched"),
            self.tr("Geometries of the rasters are matched!"),
        )
        self.geometry_matched = True
        self.__updateAnalyticTabs(self.geometry_matched)

    def checkConsistencyLoadedModel(self, model_params: ModelParams) -> bool:
        assert self.inputs is not None

        if not utils.checkFactors(self.inputs):
            return False

        return model_params.is_consistent_with(
            self.inputs["initial"],
            self.inputs["factors"],
        )

    def checkConsistencySeparateVars(self) -> None:
        # separate spatial variables for simulations should be:
        # - same number of variables as training ones
        # - same number of bands for each matching variable
        # - match geometry of initial rasters and training variables
        if not utils.checkFactors(self.inputs, sim=True):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Factors rasters is not set. Please specify them and try again"
                ),
            )
            return

        if len(self.inputs["factors"]) != len(self.inputs["factors_sim"]):
            QMessageBox.warning(
                self,
                self.tr("Different number of variables"),
                self.tr(
                    "Model is trained using {} variables, and simulation was set up with {} variables"
                ).format(
                    len(self.inputs["factors"]),
                    len(self.inputs["factors_sim"]),
                ),
            )
            return

        init_raster = self.inputs["initial"]
        for v, v2 in zip(
            self.inputs["factors_sim"].values(),
            self.inputs["factors"].values(),
        ):
            if not init_raster.geoDataMatch(v):
                QMessageBox.warning(
                    self,
                    self.tr("Different geometry"),
                    self.tr(
                        "Geometries of the initial raster and raster '{}' are different!"
                    ).format(v.getFileName()),
                )
                return

            if v.bandcount != v2.bandcount:
                QMessageBox.warning(
                    self,
                    self.tr("Different number of bands"),
                    self.tr(
                        "Training variable {} and simulation variable {} have different number of bands, {} and {} respectively"
                    ).format(
                        v.getFileName(),
                        v2.getFileName(),
                        v.bandcount,
                        v2.bandcount,
                    ),
                )
                return

        QMessageBox.warning(
            self,
            self.tr("Consistancy is checked"),
            self.tr(
                "Training variables and Simulation variables are matched!"
            ),
        )

        self.consistency_sim_checked = True

    def correlationChecking(self) -> None:
        if not utils.checkFactors(self.inputs):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Factors rasters is not set. Please specify them and try again"
                ),
            )
            return
        try:
            if self.chkAllCorr.isChecked():
                self.__checkAllCorr()
            else:
                self.__checkTwoCorr()
        except MemoryError:
            self.logErrorReport(
                self.tr(
                    "Memory Error occurred (correlation checking). Perhaps the system is low on memory."
                )
            )
            raise

        self.tblCorrelation.resizeRowsToContents()
        self.tblCorrelation.resizeColumnsToContents()
        gc.collect()

    def startUpdateStatisticsTable(self):
        if not utils.checkInputRasters(self.inputs):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Initial or final raster is not set. Please specify input data and try again"
                ),
            )
            return

        try:
            crossTabMan = CrossTableManager(
                self.inputs["initial"], self.inputs["final"]
            )
        except CrossTabManagerError as error:
            QMessageBox.warning(
                self,
                self.tr("Invalid input data"),
                str(error),
            )
            return

        self.inputs["crosstab"] = crossTabMan

        # class statistics
        crossTabMan.moveToThread(self.workThread)

        self.workThread.started.connect(crossTabMan.computeCrosstable)
        crossTabMan.rangeChanged.connect(self.setProgressRange)
        crossTabMan.errorReport.connect(self.logErrorReport)
        crossTabMan.updateProgress.connect(self.showProgress)
        crossTabMan.crossTableFinished.connect(self.updateStatisticsTableDone)
        self.workThread.start()

    def updateStatisticsTableDone(self):
        crossTabMan = self.inputs["crosstab"]
        self.workThread.started.disconnect(crossTabMan.computeCrosstable)
        crossTabMan.errorReport.disconnect(self.logErrorReport)
        crossTabMan.rangeChanged.disconnect(self.setProgressRange)
        crossTabMan.updateProgress.disconnect(self.showProgress)
        crossTabMan.crossTableFinished.disconnect(
            self.updateStatisticsTableDone
        )
        self.workThread.quit()
        self.restoreProgressState()
        self.__drawTransitionStat()
        self.logMessage(
            self.tr("Class statistics and transition matrix are updated")
        )

    @pyqtSlot()
    def createChangeMap(self) -> None:
        """
        Validates input rasters and initiates the creation of a land cover change map.

        This method performs the following:
        - Checks that both the initial and final rasters are provided.
        - Initializes the AreaAnalyst to compare the rasters.
        - Prompts the user to select a file path to save the resulting change map.
        - Deletes the file at the selected path if it already exists (after user confirmation).
        - Moves the `AreaAnalyst` instance to a background thread for asynchronous processing.
        - Connects signal-slot mechanisms for progress reporting, error handling, and completion.
        - Starts the background thread to compute the change map.
        """
        if not utils.checkInputRasters(self.inputs):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Initial or final raster is not set. Please specify input data and try again"
                ),
            )
            return

        try:
            self.analyst = AreaAnalyst(
                self.inputs["initial"], self.inputs["final"]
            )
        except AreaAnalizerError as error:
            QMessageBox.warning(
                self,
                self.tr("Invalid input rasters"),
                str(error),
            )
            return

        fileName = utils.saveRasterDialog(
            self,
            self.settings,
            self.tr("Save change map"),
            self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)"),
        )

        change_map_path = Path(fileName)
        is_change_map_path_unlinking = self.__checking_file_saving(
            change_map_path
        )
        if is_change_map_path_unlinking is None:
            return
        if self.__checking_file_saving(change_map_path):
            change_map_path.unlink()

        self.inputs["changeMapName"] = fileName

        self.analyst.moveToThread(self.workThread)
        self.workThread.started.connect(self.analyst.getChangeMap)
        self.analyst.rangeChanged.connect(self.setProgressRange)
        self.analyst.updateProgress.connect(self.showProgress)
        self.analyst.error_occurred.connect(self.show_error)
        self.analyst.errorReport.connect(self.logErrorReport)
        self.analyst.processFinished.connect(self.changeMapDone)
        self.analyst.processFinished.connect(self.workThread.quit)
        self.workThread.start()

    def changeMapDone(self, raster: Raster) -> None:
        """
        Handles the completion of the change map creation process.

        This method performs the following actions:
        - Stores the resulting raster in the inputs dictionary.
        - Saves the raster to the path specified in 'changeMapName'.
        - Adds the raster to the QGIS canvas.
        - Applies a color ramp style to the raster layer.
        - Disconnects all signals connected to the change map processing.
        - Restores the progress bar to its previous state.
        - Logs the completion message and performs garbage collection.

        param raster: The generated raster layer representing the change map.
        """
        self.inputs["changeMap"] = raster
        self.inputs["changeMap"].save(self.inputs["changeMapName"])
        self.__addRasterToCanvas(self.inputs["changeMapName"])
        layer = utils.getLayerByName(
            QFileInfo(self.inputs["changeMapName"]).baseName()
        )
        colorRamp = self.calcChangeMapColorRamp(
            layer, self.analyst, False, False
        )
        self.applyStyle(layer, colorRamp)
        del self.inputs["changeMapName"]
        self.workThread.started.disconnect(self.analyst.getChangeMap)
        self.analyst.rangeChanged.disconnect(self.setProgressRange)
        self.analyst.updateProgress.disconnect(self.showProgress)
        self.analyst.error_occurred.disconnect(self.show_error)
        self.analyst.errorReport.disconnect(self.logErrorReport)
        self.analyst.processFinished.disconnect(self.changeMapDone)
        self.analyst.processFinished.disconnect(self.workThread.quit)
        self.restoreProgressState()
        self.logMessage(self.tr("Change Map is created"))
        gc.collect()

    @pyqtSlot()
    def startSimulation(self) -> None:
        """
        Initializes and starts the land-use simulation process.

        This method performs a comprehensive validation of required input data such as:
        - Factor rasters
        - Trained model
        - Transition matrix
        - Initial raster
        """
        if not utils.checkFactors(self.inputs):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Factors rasters is not set. Please specify them and try again"
                ),
            )
            return

        if "model" not in self.inputs:
            QMessageBox.warning(
                self,
                self.tr("Missed model"),
                self.tr("Model not selected. Please select and train model."),
            )
            return

        if "crosstab" not in self.inputs:
            QMessageBox.warning(
                self,
                self.tr("Missed transition matrix"),
                self.tr("Please calculate transition matrix and try again"),
            )
            return

        if not utils.checkInputRasters(self.inputs):
            QMessageBox.warning(
                self,
                self.tr("Missed input data"),
                self.tr(
                    "Initial raster is not set. Please specify it and try again"
                ),
            )
            return

        calcTransitions = False
        if self.chkTransitionPotentials.isChecked():
            if self.leTransitionPotentialPrefix.text() == "":
                QMessageBox.warning(
                    self,
                    self.tr("Can't save file"),
                    self.tr(
                        "Prefix of transition potential maps is not set. Please specify it and try again"
                    ),
                )
                return

            if self.leTransitionPotentialDirectory.text() == "":
                QMessageBox.warning(
                    self,
                    self.tr("Can't save file"),
                    self.tr(
                        "Directory of transition potential maps is not set. Please specify it and try again"
                    ),
                )
                return

            paths_list = glob.glob(
                self.leTransitionPotentialDirectory.text()
                + "/"
                + self.leTransitionPotentialPrefix.text()
                + "*"
            )
            if len(paths_list) > 0:
                reply = QMessageBox.question(
                    self,
                    self.tr("Overwrite file"),
                    self.tr(
                        "Files with the specified prefix already exist in this directory. This may cause the files to be overwritten. Are you sure you want to continue?"
                    ),
                    QMessageBox.StandardButton.Yes,
                    QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.No:
                    return

            calcTransitions = True

        is_risk_function_path_unlinking = False
        is_monte_carlo_path_unlinking = False

        risk_function_path = Path(
            QgsFileUtils.ensureFileNameHasExtension(
                self.leRiskFunctionPath.text(), ["tif"]
            )
        )

        monte_carlo_path = Path(
            QgsFileUtils.ensureFileNameHasExtension(
                self.leMonteCarloPath.text(), ["tif"]
            )
        )

        if (
            self.chkRiskFunction.isChecked()
            and self.leRiskFunctionPath.text() != ""
        ):
            is_risk_function_path_unlinking = self.__checking_file_saving(
                risk_function_path
            )
            if is_risk_function_path_unlinking is None:
                return

        self.leRiskFunctionPath.setText(
            QgsFileUtils.ensureFileNameHasExtension(
                self.leRiskFunctionPath.text(), ["tif"]
            )
        )

        if (
            self.chkMonteCarlo.isChecked()
            and self.leMonteCarloPath.text() != ""
        ):
            is_monte_carlo_path_unlinking = self.__checking_file_saving(
                monte_carlo_path
            )
            if is_monte_carlo_path_unlinking is None:
                return

        self.leMonteCarloPath.setText(
            QgsFileUtils.ensureFileNameHasExtension(
                self.leMonteCarloPath.text(), ["tif"]
            )
        )

        if self.chkRiskFunction.isChecked() and self.chkMonteCarlo.isChecked():
            if risk_function_path == monte_carlo_path:
                QMessageBox.warning(
                    self,
                    self.tr("Can't save file"),
                    self.tr(
                        "Can't save files with the same output path '{}'."
                        " Please specify different output paths and try again"
                    ).format(self.leMonteCarloPath.text()),
                )
                return

        if self.chkSeparateVars.isChecked():
            if not utils.checkFactors(self.inputs, sim=True):
                QMessageBox.warning(
                    self,
                    self.tr("Missed input data"),
                    self.tr(
                        "Factors rasters for simulation are not set. Please specify them and try again, or disable their usage"
                    ),
                )
                return

            if not self.consistency_sim_checked:
                QMessageBox.warning(
                    self,
                    self.tr("Missed input data"),
                    self.tr(
                        "Separate variables version consistency is not checked"
                    ),
                )
                return

            factors_values = list(self.inputs["factors_sim"].values())

        else:
            if not utils.checkFactors(self.inputs):
                QMessageBox.warning(
                    self,
                    self.tr("Missed input data"),
                    self.tr(
                        "Factors rasters is not set. Please specify them and try again"
                    ),
                )
                return

            factors_values = list(self.inputs["factors"].values())

        if "model" not in self.inputs:
            QMessageBox.warning(
                self,
                self.tr("Missed model"),
                self.tr("Model not selected. Please select and train model."),
            )

        if is_risk_function_path_unlinking:
            risk_function_path.unlink()
        if is_monte_carlo_path_unlinking:
            monte_carlo_path.unlink()

        self.simulator = Simulator(
            self.inputs["final"],
            factors_values,
            self.inputs["model"],
            self.inputs["crosstab"],
        )

        self.simulator.setIterationCount(self.spnIterations.value())
        self.simulator.setCalcTransitions(calcTransitions)
        self.simulator.moveToThread(self.workThread)

        self.btnStartSimulation.setEnabled(False)

        self.workThread.started.connect(self.simulator.simN)
        self.simulator.rangeChanged.connect(self.setProgressRange)
        self.simulator.updateProgress.connect(self.showProgress)
        self.simulator.error_occurred.connect(self.show_error)
        self.simulator.errorReport.connect(self.logErrorReport)
        self.simulator.simFinished.connect(self.simulationDone)
        self.workThread.start()
        self.logMessage(self.tr("Simulation process is started"))

    @pyqtSlot()
    def simulationDone(self) -> None:
        """
        Handles the completion of the simulation process.

        This method performs the following:
        - Saves and displays the risk function raster if enabled.
        - Saves and displays the Monte Carlo simulated raster if enabled.
        - Saves transition potential maps if enabled.
        - Disconnects signals, cleans up, and logs the simulation status.
        """
        self.btnStartSimulation.setEnabled(True)
        if self.chkRiskFunction.isChecked():
            if self.leRiskFunctionPath.text() != "":
                res = self.simulator.getConfidence()
                grad = res.getBandGradation(1)
                saved = False
                # Try to use some Values as No-data Value
                maxVal = res.getGDALMaxVal()
                for noData in [0, maxVal]:
                    if noData not in grad:
                        res.save(
                            QgsFileUtils.ensureFileNameHasExtension(
                                self.leRiskFunctionPath.text(), ["tif"]
                            ),
                            nodata=noData,
                        )
                        saved = True
                        break
                if not saved:
                    res.save(
                        QgsFileUtils.ensureFileNameHasExtension(
                            self.leRiskFunctionPath.text(), ["tif"]
                        ),
                        nodata=maxVal - 1,
                    )
                del res
                self.__addRasterToCanvas(self.leRiskFunctionPath.text())
                layer = utils.getLayerByName(
                    QFileInfo(self.leRiskFunctionPath.text()).baseName()
                )
                colorRamp = self.calcCertancyColorRamp(layer)
                self.applyStyle(layer, colorRamp)
            else:
                self.logMessage(
                    self.tr(
                        "Output path for risk function map is not set. Skipping this step"
                    )
                )

        if self.chkMonteCarlo.isChecked():
            if self.leMonteCarloPath.text() != "":
                self.leSimulatedMapPath.setText(self.leMonteCarloPath.text())
                res = self.simulator.getState()
                grad = res.getBandGradation(1)
                saved = False
                # Try to use some Values as No-data Value
                maxVal = res.getGDALMaxVal()
                for noData in [0, maxVal]:
                    if noData not in grad:
                        res.save(
                            QgsFileUtils.ensureFileNameHasExtension(
                                self.leMonteCarloPath.text(), ["tif"]
                            ),
                            nodata=noData,
                        )
                        saved = True
                        break
                if not saved:
                    res.save(
                        QgsFileUtils.ensureFileNameHasExtension(
                            self.leMonteCarloPath.text(), ["tif"]
                        ),
                        nodata=maxVal - 1,
                    )
                del res
                self.__addRasterToCanvas(self.leMonteCarloPath.text())
                if utils.copySymbology(
                    utils.getLayerByName(self.leInitRasterName.text()),
                    utils.getLayerByName(
                        QFileInfo(self.leMonteCarloPath.text()).baseName()
                    ),
                ):
                    layer = utils.getLayerByName(
                        QFileInfo(self.leMonteCarloPath.text()).baseName()
                    )
                    # layer.setCacheImage(None)
                    layer.triggerRepaint()
                    self.iface.layerTreeView().refreshLayerSymbology(
                        layer.id()
                    )
                    self.iface.mapCanvas().refresh()
                    QgsProject.instance().setDirty(True)
            else:
                self.logMessage(
                    self.tr(
                        "Output path for simulated risk map is not set. Skipping this step"
                    )
                )

        if self.chkTransitionPotentials.isChecked():
            potentials = self.simulator.getTransitionPotentials()
            directory_path = Path(self.leTransitionPotentialDirectory.text())
            prefix = self.leTransitionPotentialPrefix.text()
            if not hasattr(self, "analyst"):
                try:
                    self.analyst = AreaAnalyst(
                        self.inputs["initial"], self.inputs["final"]
                    )
                except AreaAnalizerError as error:
                    QMessageBox.warning(
                        self,
                        self.tr("Invalid input rasters"),
                        str(error),
                    )
                    return
            if potentials is not None:
                for k, v in potentials.items():
                    try:
                        initcat, finalcat = map(
                            lambda category: str(category).replace(".", "_"),
                            self.analyst.decode(int(k)),
                        )
                        file_name = (
                            f"{prefix}_from_{initcat}_to_{finalcat}.tif"
                        )
                        v.save(str(directory_path / file_name))
                    except AreaAnalizerError as error:
                        QMessageBox.warning(
                            self,
                            self.tr("Invalid input rasters"),
                            str(error),
                        )
                        return
            else:
                QMessageBox.warning(
                    self,
                    self.tr("Not implemented yet"),
                    self.tr(
                        "Transition potentials not implemented yet for the model."
                    ),
                )

        self.workThread.started.disconnect(self.simulator.simN)
        self.simulator.rangeChanged.disconnect(self.setProgressRange)
        self.simulator.updateProgress.disconnect(self.showProgress)
        self.simulator.error_occurred.disconnect(self.show_error)
        self.simulator.errorReport.disconnect(self.logErrorReport)
        self.simulator.simFinished.disconnect(self.simulationDone)
        self.workThread.quit()
        self.simulator = None
        gc.collect()
        self.restoreProgressState()
        self.logMessage(self.tr("Simulation process is finished"))

    def startValidation(self):
        try:
            reference = Raster(str(self.leReferenceMapPath.text()))
        except ProviderError:
            QMessageBox.warning(
                self,
                self.tr("Can't read file"),
                self.tr("Can't read file: '{}'").format(
                    self.leReferenceMapPath.text()
                ),
            )
            return
        try:
            simulated = Raster(str(self.leSimulatedMapPath.text()))
        except ProviderError:
            QMessageBox.warning(
                self,
                self.tr("Can't read file"),
                self.tr("Can't read file: '{}'").format(
                    self.leSimulatedMapPath.text()
                ),
            )
            return

        try:
            self.eb = EBudget(reference, simulated)
        except EBError as error:
            QMessageBox.warning(
                self,
                self.tr("Invalid rasters"),
                str(error),
            )
            return
        except MemoryError:
            self.logErrorReport(
                self.tr(
                    "The system is out of memory during validation procedure"
                )
            )
            raise

        self.eb.moveToThread(self.workThread)

        self.workThread.started.connect(self.validate)
        self.eb.rangeChanged.connect(self.setProgressRange)
        self.eb.updateProgress.connect(self.showProgress)
        self.eb.validationFinished.connect(self.validationDone)
        self.workThread.start()
        self.logMessage(self.tr("Validation process is started"))

    def validationDone(self, stat):
        self.workThread.started.disconnect(self.validate)
        self.eb.rangeChanged.disconnect(self.setProgressRange)
        self.eb.updateProgress.disconnect(self.showProgress)
        self.eb.validationFinished.disconnect(self.validationDone)
        self.workThread.quit()
        self.eb = None
        gc.collect()
        self.restoreProgressState()

        self.scaleData = list(stat.keys())
        (
            self.noNoData,
            self.noMedData,
            self.medMedData,
            self.medPerData,
            self.perPerData,
        ) = [], [], [], [], []
        for k in list(stat.keys()):
            self.noNoData.append(stat[k]["NoNo"])
            self.noMedData.append(stat[k]["NoMed"])
            self.medMedData.append(stat[k]["MedMed"])
            self.medPerData.append(stat[k]["MedPer"])
            self.perPerData.append(stat[k]["PerPer"])

        self.valAxes.set_xbound(lower=0, upper=len(self.scaleData) - 1)
        self.valAxes.set_ybound(lower=0, upper=1)

        self.noNo.set_xdata(numpy.array(self.scaleData))
        self.noNo.set_ydata(numpy.array(self.noNoData))
        self.noMed.set_xdata(numpy.array(self.scaleData))
        self.noMed.set_ydata(numpy.array(self.noMedData))
        self.medMed.set_xdata(numpy.array(self.scaleData))
        self.medMed.set_ydata(numpy.array(self.medMedData))
        self.medPer.set_xdata(numpy.array(self.scaleData))
        self.medPer.set_ydata(numpy.array(self.medPerData))
        self.perPer.set_xdata(numpy.array(self.scaleData))
        self.perPer.set_ydata(numpy.array(self.perPerData))

        self.valCanvas.draw()
        self.logMessage(self.tr("Validation process is finished"))
        gc.collect()

    def startKappaValidation(self):
        try:
            reference = Raster(str(self.leReferenceMapPath.text()))
        except ProviderError:
            QMessageBox.warning(
                self,
                self.tr("Can't read file"),
                self.tr("Can't read file: '{}'").format(
                    self.leReferenceMapPath.text()
                ),
            )
            return
        try:
            simulated = Raster(str(self.leSimulatedMapPath.text()))
        except ProviderError:
            QMessageBox.warning(
                self,
                self.tr("Can't read file"),
                self.tr("Can't read file: '{}'").format(
                    self.leSimulatedMapPath.text()
                ),
            )
            return
        for raster in [reference, simulated]:
            if raster.isCountinues(bandNo=1):
                QMessageBox.warning(
                    self,
                    self.tr("Kappa is not applicable"),
                    self.tr(
                        "Kappa is not applicable to the file: '{}' because it contains continuous value"
                    ).format(raster.getFileName()),
                )
                return

        # Kappa
        self.depCoef = DependenceCoef(
            reference.getBand(1), simulated.getBand(1), expand=True
        )

        try:
            self.depCoef.calculateCrosstable()
        except CrossTabError:
            QMessageBox.warning(
                self,
                self.tr("Different geometry"),
                self.tr(
                    "Geometries of the reference and simulated rasters are different!"
                ),
            )
            return

        self.depCoef.moveToThread(self.workThread)

        self.workThread.started.connect(self.depCoef.calculateCrosstable)
        self.depCoef.rangeChanged.connect(self.setProgressRange)
        self.depCoef.updateProgress.connect(self.showProgress)
        self.depCoef.errorReport.connect(self.logErrorReport)
        self.depCoef.processFinished.connect(self.kappaValDone)
        self.workThread.start()
        self.logMessage(self.tr("Kappa validation process is started"))

    def kappaValDone(self):
        self.workThread.started.disconnect(self.depCoef.calculateCrosstable)
        self.depCoef.rangeChanged.disconnect(self.setProgressRange)
        self.depCoef.errorReport.disconnect(self.logErrorReport)
        self.depCoef.updateProgress.disconnect(self.showProgress)
        self.depCoef.processFinished.disconnect(self.kappaValDone)
        self.workThread.quit()
        self.restoreProgressState()

        try:
            kappas = self.depCoef.kappa(mode="all")
            self.leKappaOveral.setText("%6.5f" % (kappas["overal"]))
            self.leKappaHisto.setText("%6.5f" % (kappas["histo"]))
            self.leKappaLoc.setText("%6.5f" % (kappas["loc"]))
            # % of Correctness
            percent = self.depCoef.correctness()
            self.leKappaCorrectness.setText("%6.5f" % (percent))
            del self.depCoef
            self.logMessage(self.tr("Kappa validation process is finished"))
            gc.collect()
        except CoeffError as error:
            QMessageBox.warning(
                self,
                self.tr("Model training failed"),
                str(error),
            )

    @pyqtSlot()
    def createValidationMap(self) -> None:
        """
        Starts the process of creating a validation map.

        This method:
        - Loads the reference and simulated rasters.
        - Validates their compatibility.
        - Optionally sets the initial raster if persistent classes are checked.
        - Prompts the user to save the output file.
        - Sets up and starts the worker thread for validation map generation.
        """
        try:
            layerSource = str(self.leReferenceMapPath.text())
            reference = Raster(
                layerSource, utils.getLayerMaskBySource(layerSource)
            )
        except ProviderError:
            QMessageBox.warning(
                self,
                self.tr("Can't read file"),
                self.tr("Can't read file: '{}'").format(
                    self.leReferenceMapPath.text()
                ),
            )
            return
        try:
            layerSource = str(self.leSimulatedMapPath.text())
            simulated = Raster(
                layerSource, utils.getLayerMaskBySource(layerSource)
            )
        except ProviderError:
            QMessageBox.warning(
                self,
                self.tr("Can't read file"),
                self.tr("Can't read file: '{}'").format(
                    self.leSimulatedMapPath.text()
                ),
            )
            return

        try:
            self.analystVM = AreaAnalyst(reference, simulated)
        except AreaAnalizerError:
            QMessageBox.warning(
                self,
                self.tr("Different characteristics of rasters"),
                self.tr(
                    "Characteristics of the reference and simulated rasters are different!"
                ),
            )
            return

        if self.chkCheckPersistentClasses.isChecked():
            if not utils.checkInputRasters(self.inputs):
                QMessageBox.warning(
                    self,
                    self.tr("Missed input data"),
                    self.tr(
                        "Initial raster is not set. Please specify it and try again"
                    ),
                )
                return
            self.analystVM.setInitialRaster(self.inputs["initial"])

        fileName = utils.saveRasterDialog(
            self,
            self.settings,
            self.tr("Save validation map"),
            self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)"),
        )

        validation_map_path = Path(fileName)
        is_validation_map_path_unlinking = self.__checking_file_saving(
            validation_map_path
        )
        if is_validation_map_path_unlinking is None:
            return
        if self.__checking_file_saving(validation_map_path):
            validation_map_path.unlink()

        self.inputs["valMapName"] = str(fileName)

        self.analystVM.moveToThread(self.workThread)
        self.workThread.started.connect(self.analystVM.getChangeMap)
        self.analystVM.rangeChanged.connect(self.setProgressRange)
        self.analystVM.updateProgress.connect(self.showProgress)
        self.analystVM.error_occurred.connect(self.show_error)
        self.analystVM.errorReport.connect(self.logErrorReport)
        self.analystVM.processFinished.connect(self.validationMapDone)
        self.analystVM.processFinished.connect(self.workThread.quit)
        self.workThread.start()
        self.logMessage(
            self.tr("Process of Validation Map creating is started")
        )

    def validationMapDone(self, raster: Raster) -> None:
        """
        Handles the completion of the validation map creation process.

        This method:
        - Saves the resulting raster.
        - Adds it to the QGIS canvas.
        - Applies a color ramp style based on the analysis.
        - Disconnects all signals and performs cleanup.
        - Logs the success message.

        :param raster: The generated validation raster layer.
        """
        validationMap = raster
        validationMap.save(self.inputs["valMapName"])
        self.__addRasterToCanvas(self.inputs["valMapName"])
        layer = utils.getLayerByName(
            QFileInfo(self.inputs["valMapName"]).baseName()
        )
        colorRamp = self.calcChangeMapColorRamp(
            layer,
            self.analystVM,
            True,
            self.chkCheckPersistentClasses.isChecked(),
        )
        self.applyStyle(layer, colorRamp)
        del self.inputs["valMapName"]
        self.workThread.started.disconnect(self.analystVM.getChangeMap)
        self.analystVM.rangeChanged.disconnect(self.setProgressRange)
        self.analystVM.updateProgress.disconnect(self.showProgress)
        self.analystVM.error_occurred.disconnect(self.show_error)
        self.analystVM.errorReport.disconnect(self.logErrorReport)
        self.analystVM.processFinished.disconnect(self.validationMapDone)
        self.analystVM.processFinished.disconnect(self.workThread.quit)
        del self.analystVM
        gc.collect()
        self.restoreProgressState()
        self.logMessage(
            self.tr("Process of Validation Map creating is finished")
        )

    def tabChanged(self, index):
        gc.collect()
        # if self.tabWidget.currentWidget() == self.tabModel:
        #  self.__modelChanged()
        if index == 1:  # tabCorrelationChecking
            self.__populateRasterNames()

    # ******************************************************************************

    def __populateLayers(self):
        layers = utils.getRasterLayers()
        # ~ relations = self.iface.legendInterface().groupLayerRelationship()
        for layer in sorted(
            layers.items(),
            key=functools.cmp_to_key(
                lambda lhs, rhs: locale.strcoll(lhs[1], rhs[1])
            ),
        ):
            # ~ groupName = utils.getLayerGroup(relations, layer[0])
            groupName = ""
            item = QListWidgetItem()
            if groupName == "":
                item.setText(layer[1])
                item.setData(Qt.UserRole, layer[0])
            else:
                item.setText(f"{layer[1]} - {groupName}")
                item.setData(Qt.UserRole, layer[0])

            self.lstLayers.addItem(item)
            self.lstLayersSeparateVars.addItem(item.clone())

    def __populateRasterNames(self):
        self.cmbFirstRaster.clear()
        self.cmbSecondRaster.clear()
        for index in range(self.lstFactors.count()):
            item = self.lstFactors.item(index)
            self.cmbFirstRaster.addItem(item.text(), item.data(Qt.UserRole))
            self.cmbSecondRaster.addItem(item.text(), item.data(Qt.UserRole))

    def __populateCorrCheckingMet(self):
        self.cmbCorrCheckMethod.addItems(
            [
                self.tr("Pearson's Correlation"),
                self.tr("Cramer's Coefficient"),
                self.tr("Joint Information Uncertainty"),
            ]
        )

    def __populateUnits(self):
        self.cmbUnits.addItems(
            [self.tr("raster units"), self.tr("sq. km."), self.tr("ha")]
        )

    def __populateSimulationMethods(self):
        self.cmbSimulationMethod.addItems(
            [
                self.tr("Artificial Neural Network (Multi-layer Perceptron)"),
                self.tr("Weights of Evidence"),
                self.tr("Multi Criteria Evaluation"),
            ]
        )
        if not scipyMissed:
            self.cmbSimulationMethod.addItem(self.tr("Logistic Regression"))

        self.cmbSimulationMethod.addItems([self.tr("Model from file")])

    def __populateSamplingModes(self):
        self.cmbSamplingMode.addItem(self.tr("All"), 0)
        self.cmbSamplingMode.addItem(self.tr("Random"), 1)
        self.cmbSamplingMode.addItem(self.tr("Stratified"), 2)
        self.cmbSamplingMode.setCurrentIndex(1)

    def __populateValidationPlot(self):
        # init plot for validation curve
        self.valFigure = Figure()
        self.valAxes = self.valFigure.add_subplot(111)
        self.valAxes.grid(True)
        self.valFigure.suptitle(self.tr("Multiple-resolution budget"))
        self.valCanvas = FigureCanvas(self.valFigure)
        self.valtoolbar = NavigationToolbar(self.valCanvas, None)
        lstActions = self.valtoolbar.actions()
        self.valtoolbar.removeAction(lstActions[7])
        self.layoutValPlot.addWidget(self.valCanvas)
        self.layoutValPlot.addWidget(self.valtoolbar)

        self.scaleData = []
        self.noNoData = []
        self.noNo = self.valAxes.plot(
            self.noNoData,
            linewidth=1,
            color="green",
            linestyle="dashed",
            marker="o",
        )[0]
        self.noMedData = []
        self.noMed = self.valAxes.plot(
            self.noMedData,
            linewidth=1,
            color="red",
            marker="o",
        )[0]
        self.medMedData = []
        self.medMed = self.valAxes.plot(
            self.medMedData,
            linewidth=1,
            color="purple",
            linestyle="dashed",
            marker="v",
        )[0]
        self.medPerData = []
        self.medPer = self.valAxes.plot(
            self.medPerData,
            linewidth=1,
            color="black",
            linestyle="dashed",
            marker="+",
        )[0]
        self.perPerData = []
        self.perPer = self.valAxes.plot(
            self.perPerData,
            linewidth=1,
            color="yellow",
            marker="*",
        )[0]
        box = self.valAxes.get_position()
        self.valAxes.set_position(
            (box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8)
        )
        leg = self.valAxes.legend(
            (
                self.tr("No location,\nno quantity inform."),
                self.tr("No location,\nmedium quantity inform."),
                self.tr("Medium location,\nmedium quantity inform."),
                self.tr("Perfect location,\nmedium quantity inform."),
                self.tr("Perfect location,\nperfect quantity inform."),
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.09),
            fancybox=True,
            ncol=3,
            shadow=False,
            fontsize=8,
        )
        for t in leg.get_texts():
            t.set_fontsize("small")

    def __checkAllCorr(self):
        dim = self.__bandCount()
        self.tblCorrelation.clear()
        self.tblCorrelation.setRowCount(dim)
        self.tblCorrelation.setColumnCount(dim)

        labels = []
        mapping, labNo = {}, 0  # Maping between raster ID and label number
        for k, v in self.inputs["factors"].items():
            mapping[k] = {}
            for b in range(v.getBandsCount()):
                if v.getBandsCount() > 1:
                    name = (
                        f"{utils.getLayerById(k).name()} (band {str(b + 1)})"
                    )
                else:
                    name = utils.getLayerById(k).name()
                mapping[k][b] = labNo
                labNo = labNo + 1
                labels.append(name)
        self.tblCorrelation.setVerticalHeaderLabels(labels)
        self.tblCorrelation.setHorizontalHeaderLabels(labels)

        method = self.cmbCorrCheckMethod.currentText()
        discreteMethods = [  # The methods need categorial values
            self.tr("Cramer's Coefficient"),
            self.tr("Joint Information Uncertainty"),
        ]
        # Loop over all rasters and all bands
        self.setProgressRange(
            self.tr("Correlation checking"), dim * (dim - 1) / 2
        )
        for i, fact1 in self.inputs["factors"].items():
            for b1 in range(fact1.getBandsCount()):
                labNo1 = mapping[i][b1]
                for j, fact2 in self.inputs["factors"].items():
                    for b2 in range(fact2.getBandsCount()):
                        labNo2 = mapping[j][b2]
                        if labNo2 < labNo1:
                            continue
                        if labNo2 == labNo1:
                            item = QTableWidgetItem("--")
                        # Check if method is applicable to the bands
                        elif (
                            fact1.isCountinues(b1 + 1)
                            or fact2.isCountinues(b2 + 1)
                        ) and method in discreteMethods:
                            item = QTableWidgetItem(self.tr("Not applicable"))
                        else:
                            depCoef = DependenceCoef(
                                fact1.getBand(b1 + 1), fact2.getBand(b2 + 1)
                            )
                            if method == self.tr("Pearson's Correlation"):
                                coef = depCoef.correlation()
                            elif method == self.tr(
                                "Joint Information Uncertainty"
                            ):
                                coef = depCoef.jiu()
                            elif method == self.tr("Cramer's Coefficient"):
                                coef = depCoef.cramer()
                            item = QTableWidgetItem(str(coef))
                        self.tblCorrelation.setItem(labNo1, labNo2, item)
                        self.showProgress()
        try:
            del depCoef
        except NameError:
            pass
        self.restoreProgressState()

    def __checkTwoCorr(self):
        index = self.cmbFirstRaster.currentIndex()
        layerId = str(self.cmbFirstRaster.itemData(index, Qt.UserRole))
        first = {
            "Raster": self.inputs["factors"][layerId],
            "Name": self.cmbFirstRaster.currentText(),
        }
        index = self.cmbSecondRaster.currentIndex()
        layerId = str(self.cmbSecondRaster.itemData(index, Qt.UserRole))
        second = {
            "Raster": self.inputs["factors"][layerId],
            "Name": self.cmbSecondRaster.currentText(),
        }

        dimensions = (
            first["Raster"].getBandsCount(),
            second["Raster"].getBandsCount(),
        )
        self.tblCorrelation.setRowCount(dimensions[0])
        self.tblCorrelation.setColumnCount(dimensions[1])
        labels = []
        for i in range(dimensions[0]):
            raster = first["Raster"]
            if raster.getBandsCount() > 1:
                name = f"{first['Name']} (band {str(i + 1)}"
            else:
                name = str(first["Name"])
            labels.append(name)
        self.tblCorrelation.setVerticalHeaderLabels(labels)
        labels = []
        for i in range(dimensions[1]):
            raster = second["Raster"]
            if raster.getBandsCount() > 1:
                name = f"{second['Name']} (band {str(i + 1)})"
            else:
                name = str(second["Name"])
            labels.append(name)
        self.tblCorrelation.setHorizontalHeaderLabels(labels)

        method = self.cmbCorrCheckMethod.currentText()
        if method == self.tr("Pearson's Correlation"):
            for col in range(dimensions[1]):
                for row in range(dimensions[0]):
                    depCoef = DependenceCoef(
                        first["Raster"].getBand(row + 1),
                        second["Raster"].getBand(col + 1),
                    )
                    corr = depCoef.correlation()
                    item = QTableWidgetItem(str(corr))
                    self.tblCorrelation.setItem(row, col, item)
        elif method == self.tr("Cramer's Coefficient"):
            for col in range(dimensions[1]):
                for row in range(dimensions[0]):
                    depCoef = DependenceCoef(
                        first["Raster"].getBand(row + 1),
                        second["Raster"].getBand(col + 1),
                    )
                    if first["Raster"].isCountinues(row + 1) or second[
                        "Raster"
                    ].isCountinues(col + 1):
                        item = QTableWidgetItem(str(self.tr("Not applicable")))
                    else:
                        corr = depCoef.cramer()
                        item = QTableWidgetItem(str(corr))
                    self.tblCorrelation.setItem(row, col, item)
        elif method == self.tr("Joint Information Uncertainty"):
            for col in range(dimensions[1]):
                for row in range(dimensions[0]):
                    depCoef = DependenceCoef(
                        first["Raster"].getBand(row + 1),
                        second["Raster"].getBand(col + 1),
                    )
                    if first["Raster"].isCountinues(row + 1) or second[
                        "Raster"
                    ].isCountinues(col + 1):
                        item = QTableWidgetItem(str(self.tr("Not applicable")))
                    else:
                        corr = depCoef.jiu()
                        item = QTableWidgetItem(str(corr))
                    self.tblCorrelation.setItem(row, col, item)
        try:
            del depCoef
        except NameError:
            pass

    def __drawTransitionStat(self):
        if "crosstab" not in self.inputs:
            return

        try:
            stat = self.inputs["crosstab"].getTransitionStat()
        except CrossTabManagerError as error:
            QMessageBox.warning(
                self,
                self.tr("Invalid input rasters"),
                str(error),
            )
            return

        dimensions = len(stat["init"])

        units_translations = {
            "metre": self.tr("metre"),
            "meter": self.tr("meter"),
            "meters": self.tr("meters"),
            "metres": self.tr("metres"),
            "unknown": self.tr("unknown"),
        }
        units = stat["unit"].lower()
        displayUnits = self.cmbUnits.currentText()
        if displayUnits == self.tr("sq. km."):
            denominator = 1000000
        elif displayUnits == self.tr("ha"):
            denominator = 10000
        else:
            denominator = 1.0
            displayUnits = self.tr("sq. ") + units_translations.get(
                units, units
            )

        if units not in ["metre", "meter", "meters", "metres"]:
            denominator = 1.0

        self.tblStatistics.clear()
        self.tblStatistics.setRowCount(dimensions)
        self.tblStatistics.setColumnCount(7)

        labels = []
        colors = []
        layer = utils.getLayerById(self.initRasterId)
        rows = layer.height()
        cols = layer.width()
        data_provider = layer.dataProvider()
        block = data_provider.block(1, data_provider.extent(), cols, rows)
        unique_values = list(
            set([block.value(r, c) for r in range(rows) for c in range(cols)])
        )
        if layer.renderer().type().lower() in (
            "singlebandpseudocolor",
            "paletted",
        ):
            if "paletted" == layer.renderer().type().lower():
                legend = layer.renderer().classes()
            if "singlebandpseudocolor" == layer.renderer().type().lower():
                legend = (
                    layer.renderer()
                    .shader()
                    .rasterShaderFunction()
                    .colorRampItemList()
                )
            for i in legend:
                if i.value in unique_values:
                    labels.append(i.label)
                    colors.append(i.color)
        else:
            labels = [str(i) for i in range(1, 7)]

        self.tblStatistics.setVerticalHeaderLabels(labels)

        labels = [
            self.tr("Class color"),
            self.leInitYear.text(),
            self.leFinalYear.text(),
            "Δ",
            self.leInitYear.text() + " %",
            self.leFinalYear.text() + " %",
            "Δ %",
        ]
        self.tblStatistics.setHorizontalHeaderLabels(labels)

        # legend colors
        d = len(colors)
        if d > 0:
            for i in range(d):
                item = QTableWidgetItem("")
                item.setBackground(QBrush(colors[i]))
                self.tblStatistics.setItem(i, 0, item)

        self.__addTableColumn(
            1,
            ["%0.2f" % (a,) for a in stat["init"] / denominator],
            displayUnits,
        )
        self.__addTableColumn(
            2,
            ["%0.2f" % (a,) for a in stat["final"] / denominator],
            displayUnits,
        )
        self.__addTableColumn(
            3,
            ["%0.2f" % (a,) for a in stat["deltas"] / denominator],
            displayUnits,
        )
        self.__addTableColumn(4, stat["initPerc"])
        self.__addTableColumn(5, stat["finalPerc"])
        self.__addTableColumn(6, stat["deltasPerc"])

        # self.tblStatistics.resizeRowsToContents()
        self.tblStatistics.resizeColumnsToContents()

        # transitional matrix
        transition = self.inputs["crosstab"].getTransitionMatrix()
        dimensions = len(transition)

        self.tblTransMatrix.clear()
        self.tblTransMatrix.setRowCount(dimensions)
        self.tblTransMatrix.setColumnCount(dimensions)

        labels = []
        layer = utils.getLayerById(self.initRasterId)
        if layer.renderer().type().lower() in (
            "singlebandpseudocolor",
            "paletted",
        ):
            if "paletted" == layer.renderer().type().lower():
                legend = layer.renderer().classes()
            if "singlebandpseudocolor" == layer.renderer().type().lower():
                legend = (
                    layer.renderer()
                    .shader()
                    .rasterShaderFunction()
                    .colorRampItemList()
                )
            for i in legend:
                if i.value in unique_values:
                    labels.append(i.label)
        else:
            labels = [str(i) for i in range(1, dimensions + 1)]

        self.tblTransMatrix.setVerticalHeaderLabels(labels)
        self.tblTransMatrix.setHorizontalHeaderLabels(labels)

        for row in range(dimensions):
            for col in range(dimensions):
                item = QTableWidgetItem(str(transition[row, col]))
                self.tblTransMatrix.setItem(row, col, item)

        self.tblTransMatrix.resizeRowsToContents()
        self.tblTransMatrix.resizeColumnsToContents()

    def __modeChanged(self, index):
        mode = self.cmbSamplingMode.itemData(index)
        if mode == 0:
            self.inputs["samplingMode"] = "All"
        elif mode == 1:
            self.inputs["samplingMode"] = "Random"
        elif mode == 2:
            self.inputs["samplingMode"] = "Stratified"

    def __modelChanged(self):
        if self.modelWidget is not None:
            self.widgetStackMethods.removeWidget(self.modelWidget)

            self.modelWidget = None
            del self.modelWidget

        modelName = self.cmbSimulationMethod.currentText()

        if modelName == self.tr("Logistic Regression"):
            self.modelWidget = (
                logisticregressionwidget.LogisticRegressionWidget(self)
            )
            self.grpSampling.show()
        elif modelName == self.tr(
            "Artificial Neural Network (Multi-layer Perceptron)"
        ):
            self.modelWidget = neuralnetworkwidget.NeuralNetworkWidget(self)
            self.grpSampling.show()
        elif modelName == self.tr("Weights of Evidence"):
            self.modelWidget = weightofevidencewidget.WeightOfEvidenceWidget(
                self
            )
            self.grpSampling.hide()
        elif modelName == self.tr("Multi Criteria Evaluation"):
            self.modelWidget = (
                multicriteriaevaluationwidget.MultiCriteriaEvaluationWidget(
                    self
                )
            )
            self.grpSampling.hide()
        elif modelName == self.tr("Model from file"):
            self.modelWidget = loadedmodelwidget.LoadedModelWidget(self)
            self.grpSampling.hide()

        self.widgetStackMethods.addWidget(self.modelWidget)
        self.widgetStackMethods.setCurrentWidget(self.modelWidget)

    def __toggleLineEdit(self, checked):
        senderName = self.sender().objectName()
        if senderName == "chkRiskFunction":
            self.leRiskFunctionPath.setEnabled(checked)
            self.btnSelectRiskFunction.setEnabled(checked)
        elif senderName == "chkRiskValidation":
            self.btnSelectRiskValidation.setEnabled(checked)
            self.chkCheckPersistentClasses.setEnabled(checked)
        elif senderName == "chkMonteCarlo":
            self.leMonteCarloPath.setEnabled(checked)
            self.btnSelectMonteCarlo.setEnabled(checked)
            self.lblIterations.setEnabled(checked)
            self.spnIterations.setEnabled(checked)
        elif senderName == "chkTransitionPotentials":
            self.leTransitionPotentialPrefix.setEnabled(checked)
            self.leTransitionPotentialDirectory.setEnabled(checked)
            self.btnSelectTransitionPrefix.setEnabled(checked)

    def __selectSamplesOutput(self):
        if "model" not in self.inputs:
            QMessageBox.warning(
                self,
                self.tr("Missed model"),
                self.tr(
                    "Nothing to save, samples were not yet generated as the model was not trained. Train the model first."
                ),
            )
            return
        model = self.inputs["model"]
        if not hasattr(model, "saveSamples"):
            QMessageBox.warning(
                self,
                self.tr("Missed samples"),
                self.tr("Selected model doesn't use samples"),
            )
            return

        fileName = utils.saveVectorDialog(
            self,
            self.settings,
            self.tr("Save file"),
            self.tr("Shape files (*.shp *.SHP *.Shp)"),
        )
        if fileName == "":
            return

        try:
            model.saveSamples(str(fileName))
        except SamplerError:
            QMessageBox.warning(
                self,
                self.tr("Can't save file"),
                self.tr("Can't save file: '{}'").format(fileName),
            )
            return

        samples_path = Path(fileName)
        if samples_path.exists():
            if utils.is_file_used_by_project(samples_path):
                QMessageBox.warning(
                    self,
                    self.tr("Can't rewrite file"),
                    self.tr(
                        "File '{}' is used in the QGIS project. It is not possible to overwrite the file, specify a different file name and try again"
                    ).format(fileName),
                )
                return

            samples_path.unlink()
            samples_path.with_suffix(".prj").unlink(missing_ok=True)
            samples_path.with_suffix(".dbf").unlink(missing_ok=True)

        if self.chkLoadSamples.isChecked():
            newLayer = QgsVectorLayer(
                fileName, QFileInfo(fileName).baseName(), "ogr"
            )

            if newLayer.isValid():
                QgsProject.instance().addMapLayer(newLayer)
            else:
                QMessageBox.warning(
                    self,
                    self.tr("Can't open file"),
                    self.tr("Error loading output shapefile:\n{}").format(
                        fileName
                    ),
                )

    def __selectSimulationOutput(self):
        senderName = self.sender().objectName()

        if senderName == "btnSelectTransitionPrefix":
            dirname = utils.openDirectoryDialog(
                self, self.settings, self.tr("Select Directory name")
            )
            self.leTransitionPotentialDirectory.setText(dirname)
            return

        fileName = utils.saveRasterDialog(
            self,
            self.settings,
            self.tr("Save file"),
            self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)"),
        )
        if fileName == "":
            return
        dirname = os.path.dirname(fileName)

        if senderName == "btnSelectRiskFunction":
            self.leRiskFunctionPath.setText(fileName)
        elif senderName == "btnSelectMonteCarlo":
            self.leMonteCarloPath.setText(fileName)

    def __toggleCorrLayers(self, state):
        if state == Qt.Checked:
            self.cmbFirstRaster.setEnabled(False)
            self.cmbSecondRaster.setEnabled(False)
        else:
            self.cmbFirstRaster.setEnabled(True)
            self.cmbSecondRaster.setEnabled(True)

    def __selectValidationMap(self):
        senderName = self.sender().objectName()

        fileName = utils.openRasterDialog(
            self,
            self.settings,
            self.tr("Open file"),
            self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)"),
        )
        if fileName == "":
            return

        if senderName == "btnSelectReferenceMap":
            self.leReferenceMapPath.setText(fileName)
        elif senderName == "btnSelectSimulatedMap":
            self.leSimulatedMapPath.setText(fileName)

    def validate(self):
        nIter = self.spnValIterCount.value()
        try:
            self.eb.getStat(nIter)
        except MemoryError:
            self.logErrorReport(
                self.tr(
                    "The system is out of memory during validation procedure"
                )
            )
            raise
        except:
            self.logErrorReport(
                self.tr("An unknown error occurs during validation procedure")
            )
            raise

    def logMessage(self, message):
        self.txtMessages.append(
            f"[{datetime.datetime.now().strftime(b'%a %b %d %Y %H:%M:%S'.decode('utf-8'))}] {message}"
        )

    def logErrorReport(self, message):
        self.logMessage("ERROR: " + message)

    def __addTableColumn(self, col, values, units=""):
        dimensions = len(values)
        for r in range(dimensions):
            if units == "":
                item = QTableWidgetItem(str(values[r]))
            else:
                item = QTableWidgetItem(str(values[r]) + " " + units)
            self.tblStatistics.setItem(r, col, item)

    def __addRasterToCanvas(self, filePath):
        layer = QgsRasterLayer(filePath, QFileInfo(filePath).baseName())
        if layer.isValid():
            QgsProject.instance().addMapLayers([layer])
        else:
            self.logMessage(self.tr("Can't load raster %s") % (filePath))

    def __bandCount(self, sim: bool = False) -> int:
        bands = 0
        factors_key = "factors_sim" if sim else "factors"
        for v in self.inputs.get(factors_key, {}).values():
            bands += v.getBandsCount()
        return bands

    def setProgressRange(self, message, maxValue):
        self.progressBar.setFormat(message)
        self.progressBar.setRange(0, int(maxValue))

    def showProgress(self):
        self.progressBar.setValue(self.progressBar.value() + 1)

    def restoreProgressState(self):
        self.progressBar.setFormat("%p%")
        self.progressBar.setRange(0, 1)
        self.progressBar.setValue(0)

    def __writeSettings(self):
        # samples and model tab
        self.settings.setValue(
            "ui/samplingMode",
            self.cmbSamplingMode.itemData(self.cmbSamplingMode.currentIndex()),
        )
        self.settings.setValue("ui/samplesCount", self.spnSamplesCount.value())
        self.settings.setValue(
            "ui/loadSamples", self.chkLoadSamples.isChecked()
        )

        # simulation tab
        self.settings.setValue(
            "ui/createRiskFunction", self.chkRiskFunction.isChecked()
        )
        self.settings.setValue(
            "ui/createRiskValidation", self.chkRiskValidation.isChecked()
        )
        self.settings.setValue(
            "ui/createMonteCarlo", self.chkMonteCarlo.isChecked()
        )
        self.settings.setValue(
            "ui/monteCarloIterations", self.spnIterations.value()
        )

        # correlation tab
        self.settings.setValue(
            "ui/checkAllRasters", self.chkAllCorr.isChecked()
        )

    def __readSettings(self):
        # samples and model tab
        samplingMode = int(self.settings.value("ui/samplingMode", 1))
        self.cmbSamplingMode.setCurrentIndex(
            self.cmbSamplingMode.findData(samplingMode)
        )
        self.spnSamplesCount.setValue(
            int(self.settings.value("ui/samplesCount", 1000))
        )
        self.chkLoadSamples.setChecked(
            bool(self.settings.value("ui/loadSamples", False))
        )

        # simulation tab
        self.chkRiskFunction.setChecked(
            bool(self.settings.value("ui/createRiskFunction", False))
        )
        self.chkRiskValidation.setChecked(
            bool(self.settings.value("ui/createRiskValidation", False))
        )
        self.chkMonteCarlo.setChecked(
            bool(self.settings.value("ui/createMonteCarlo", False))
        )
        self.spnIterations.setValue(
            int(self.settings.value("ui/monteCarloIterations", 1))
        )

        # correlation tab
        self.chkAllCorr.setChecked(
            bool(self.settings.value("ui/checkAllRasters", False))
        )

    def calcCertancyColorRamp(self, layer):
        r = Raster(str(layer.source()))
        _stat = r.getBandStat(1)
        minVal = 0.0
        maxVal = 100.0
        numberOfEntries = 11

        entryValues = []
        entryColors = []

        colorRamp = QgsStyle().defaultStyle().colorRamp("Spectral")
        currentValue = float(minVal)
        intervalDiff = float(maxVal - minVal) / float(numberOfEntries - 1)

        for i in range(numberOfEntries):
            entryValues.append(currentValue)
            currentValue += intervalDiff
            entryColors.append(
                colorRamp.color(float(i) / float(numberOfEntries))
            )

        colorRampItems = []
        for i in range(len(entryValues)):
            item = QgsColorRampShader.ColorRampItem()

            item.value = entryValues[i]
            item.color = entryColors[i]
            item.label = str(entryValues[i])
            colorRampItems.append(item)

        return colorRampItems

    def calcChangeMapColorRamp(
        self, layer, analyst, validationMode, usePercistentClass
    ):
        l = utils.getLayerByName(self.leInitRasterName.text())  # noqa: E741
        mode = l.renderer().type().lower()
        if mode not in ("singlebandpseudocolor", "paletted"):
            self.logMessage(
                self.tr(
                    "Init raster should be in PseudoColor or Paletted mode. Style not applied."
                )
            )
            return None

        r = Raster(str(layer.source()))
        stat = r.getBandStat(1)
        minVal = float(stat["min"])
        maxVal = float(stat["max"])
        numberOfEntries = int(maxVal - minVal + 1)

        if usePercistentClass:
            persistentCategoryCode = analyst.persistentCategoryCode

        entryValues = []
        entryColors = []

        colorRamp = QgsStyle().defaultStyle().colorRamp("Spectral")
        currentValue = float(minVal)
        intervalDiff = float(maxVal - minVal) / float(numberOfEntries - 1)

        for i in range(numberOfEntries):
            entryValues.append(currentValue)
            currentValue += intervalDiff
            entryColors.append(
                colorRamp.color(float(i) / float(numberOfEntries))
            )

        if "singlebandpseudocolor" == l.renderer().type().lower():
            cr = (
                l.renderer()
                .shader()
                .rasterShaderFunction()
                .colorRampItemList()
            )
        if "paletted" == l.renderer().type().lower():
            cr = l.renderer().classes()

        colorRampItems = []
        for i in range(len(entryValues)):
            item = QgsColorRampShader.ColorRampItem()

            item.value = entryValues[i]
            item.color = entryColors[i]

            if usePercistentClass and item.value == persistentCategoryCode:
                item.label = self.tr("Persistent")
            else:
                ic, fc = analyst.decode(int(entryValues[i]))
                item.label = str(self.fl(cr, ic) + " → " + self.fl(cr, fc))
                if ic == fc and validationMode:
                    item.color = QColor(255, 255, 255, 0)

            colorRampItems.append(item)

        return colorRampItems

    def applyStyle(self, layer, colorRampItems):
        if colorRampItems is None:
            return
        rasterShader = QgsRasterShader()
        colorRampShader = QgsColorRampShader()

        initial_layer = utils.getLayerByName(self.leInitRasterName.text())

        if (
            "singlebandpseudocolor" == initial_layer.renderer().type().lower()
            or layer.source() == self.leRiskFunctionPath.text()
        ):
            colorRampShader.setColorRampItemList(colorRampItems)
            if Qgis.versionInt() >= QGIS_3_38:
                colorRampShader.setColorRampType(
                    Qgis.ShaderInterpolationMethod.Linear
                )
            else:
                colorRampShader.setColorRampType(
                    QgsColorRampShader.Type.Interpolated
                )
            rasterShader.setRasterShaderFunction(colorRampShader)

            renderer = QgsSingleBandPseudoColorRenderer(
                layer.dataProvider(), 1, rasterShader
            )

            minVal = colorRampItems[0].value
            maxVal = colorRampItems[-1].value
            renderer.setClassificationMin(minVal)
            renderer.setClassificationMax(maxVal)
            min_max_origin = renderer.minMaxOrigin()
            min_max_origin.setExtent(QgsRasterMinMaxOrigin.Extent.WholeRaster)
            renderer.setMinMaxOrigin(min_max_origin)

        if (
            "paletted" == initial_layer.renderer().type().lower()
            and layer.source() != self.leRiskFunctionPath.text()
        ):
            colorRampShader.setColorRampItemList(colorRampItems)
            if Qgis.versionInt() >= QGIS_3_38:
                colorRampShader.setColorRampType(
                    Qgis.ShaderInterpolationMethod.Discrete
                )
            else:
                colorRampShader.setColorRampType(
                    QgsColorRampShader.Type.Discrete
                )

            renderer = QgsPalettedRasterRenderer(
                layer.dataProvider(),
                1,
                QgsPalettedRasterRenderer.colorTableToClassData(
                    colorRampItems
                ),
            )

        layer.setRenderer(renderer)
        # layer.setCacheImage(None)
        layer.triggerRepaint()
        self.iface.layerTreeView().refreshLayerSymbology(layer.id())
        QgsProject.instance().setDirty(True)

    def __updateAnalyticTabs(self, value):
        # Enables/Disables Analytic tabs
        for i in range(
            1, self.tabWidget.count() - 1
        ):  # Last tab is 'Messages'
            self.tabWidget.setTabEnabled(i, value)

    def fl(self, cr, v):
        for i in cr:
            if i.value == v:
                return i.label
        return ""

    def saveModel(self) -> None:
        assert self.inputs is not None

        if "model" not in self.inputs:
            QMessageBox.warning(
                self,
                self.tr("Model is missed"),
                self.tr("There is no trained model available"),
            )
            return

        file_name = utils.saveDialog(
            self,
            self.settings,
            self.tr("Save MOLUSCE model"),
            self.tr("MOLUSCE (*.molusce *.MOLUSCE)"),
            "molusce",
        )

        if not file_name:
            return

        try:
            model_params = ModelParams.from_data(
                self.inputs["model"],
                self.inputs["initial"],
                self.inputs["factors"],
            )
        except SerializerError as error:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Serialization error: %s" % str(error)),
            )
            return
        except Exception as error:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Unknown error: %s" % str(error)),
            )
            return

        try:
            ModelParamsSerializer.to_file(model_params, file_name)
        except Exception as error:
            QMessageBox.warning(
                self,
                self.tr("Failed to save model"),
                self.tr("Error: %s" % str(error)),
            )
            return

        QMessageBox.warning(
            self,
            self.tr("Success!"),
            self.tr("Model was succesfully saved"),
        )

    def loadModel(self) -> None:
        assert self.inputs is not None

        file_name = utils.openRasterDialog(
            self,
            self.settings,
            self.tr("Open file"),
            self.tr("MOLUSCE (*.molusce *.MOLUSCE)"),
        )
        if not file_name:
            return

        try:
            model_params = ModelParamsSerializer.from_file(file_name)
        except SerializerError as error:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Serialization error: %s" % str(error)),
            )
            return
        except Exception as error:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Unknown error: %s" % str(error)),
            )
            return

        consistency = self.checkConsistencyLoadedModel(model_params)

        index = self.cmbSimulationMethod.findText(
            self.tr("Model from file"), Qt.MatchFlag.MatchFixedString
        )
        self.cmbSimulationMethod.setCurrentIndex(index)

        self.modelWidget.loadedModelTextEdit.clear()
        self.modelWidget.loadedModelTextEdit.append(
            self.tr(
                '<font color="black">Model is loaded. Configuration:</font>\n'
            )
        )
        self.modelWidget.loadedModelTextEdit.append(
            self.tr("Model type: {}\n").format(model_params.model_type)
        )
        self.modelWidget.loadedModelTextEdit.append(
            self.tr("Creation date: {}\n").format(model_params.creation_ts)
        )
        self.modelWidget.loadedModelTextEdit.append(
            self.tr("MOLUSCE version: {}\n").format(
                model_params.molusce_version
            )
        )
        if (
            pluginMetadata("molusce", "version")
            != model_params.molusce_version
        ):
            self.modelWidget.loadedModelTextEdit.append(
                self.tr(
                    '<font color="red">[WARNING!] Model was created in different MOLUSCE version ({})</font>\n'.format(
                        model_params.molusce_version
                    )
                )
            )
        self.modelWidget.loadedModelTextEdit.append(
            self.tr("Spatial domain dimensions: {}x{}\n").format(
                model_params.base_xsize, model_params.base_ysize
            )
        )
        self.modelWidget.loadedModelTextEdit.append(
            self.tr("Classes: {}\n").format(model_params.base_classes)
        )
        self.modelWidget.loadedModelTextEdit.append(
            self.tr("List of factors used for training:\n")
        )
        for factor in model_params.factors_metadata:
            self.modelWidget.loadedModelTextEdit.append(
                self.tr("  * {} ({} bands)\n").format(
                    factor["name"], factor["bandcount"]
                )
            )
        if consistency:
            self.inputs["model"] = model_params.model
            self.modelWidget.loadedModelTextEdit.append(
                self.tr(
                    '<font color="green">[SUCCESS!] Model is compatible with current inputs setup. Successfully loaded</font>'
                )
            )
        else:
            self.modelWidget.loadedModelTextEdit.append(
                self.tr(
                    '<font color="red">[WARNING!] Model is incompatible with current inputs setup. Impossible to use</font>'
                )
            )
            self.modelWidget.loadedModelTextEdit.append(
                self.tr("\n=======================\n")
            )
            self.modelWidget.loadedModelTextEdit.append(
                self.tr("Current inputs setup:\n")
            )
            self.modelWidget.loadedModelTextEdit.append(
                self.tr("Spatial domain dimensions: {}x{}\n").format(
                    self.inputs["initial"].getXSize(),
                    self.inputs["initial"].getYSize(),
                )
            )
            self.modelWidget.loadedModelTextEdit.append(
                self.tr("Classes: {}\n").format(
                    self.inputs["initial"].getUniqueValues()
                )
            )
            self.modelWidget.loadedModelTextEdit.append(
                self.tr("List of factors used for training:\n")
            )
            for factor_name, factor_content in self.inputs["factors"].items():
                self.modelWidget.loadedModelTextEdit.append(
                    self.tr("  * {} ({} bands)\n").format(
                        factor_name, factor_content.bandcount
                    )
                )

    def __checking_file_saving(self, filepath: Path) -> Optional[bool]:
        parent_path = filepath.parent
        if not parent_path.exists() or str(parent_path) == ".":
            QMessageBox.warning(
                self,
                self.tr("Can't save file"),
                self.tr(
                    "Can't save file in the specified path '{}'. Please specify output path correctly and try again"
                ).format(filepath),
            )
            return None

        if filepath.exists():
            if utils.is_file_used_by_project(filepath):
                QMessageBox.warning(
                    self,
                    self.tr("Can't rewrite file"),
                    self.tr(
                        "File '{}' is used in the QGIS project. It is not possible to overwrite the file, specify a different file name and try again"
                    ).format(filepath),
                )
                return None
            return True
        return False

    @pyqtSlot(str, str)
    def show_error(self, title: str, message: str) -> None:
        """
        Display a warning message box with the error details.

        :param title: The title of the warning message box.
        :param message: The error message to display.
        """
        QMessageBox.warning(self, title, message)
