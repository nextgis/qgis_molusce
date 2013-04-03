# -*- coding: utf-8 -*-

#******************************************************************************
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
#******************************************************************************

import datetime
import locale
import operator

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from qgis.core import *

sklearnMissed = False

try:
  import sklearn
except ImportError:
  sklearnMissed = True

if not sklearnMissed:
  import logisticregressionwidget

import neuralnetworkwidget
import weightofevidencewidget
import multicriteriaevaluationwidget

from ui.ui_moluscedialogbase import Ui_Dialog

import molusceutils as utils

from algorithms.dataprovider import Raster
from algorithms.models.crosstabs.manager import CrossTableManager
from algorithms.models.area_analysis.manager import AreaAnalyst
from algorithms.models.simulator.sim import Simulator

class MolusceDialog(QDialog, Ui_Dialog):
  def __init__(self, iface):
    QDialog.__init__(self)
    self.setupUi(self)

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
    #  "model" : object
    # }
    # Layer ids are necessary to handle factors changes (e.g. adding new or removing
    # existing factor)
    self.inputs = dict()

    self.settings = QSettings("NextGIS", "MOLUSCE")

    self.grpSampling.setSettings(self.settings)

    # connect signals and slots
    self.btnSetInitialRaster.clicked.connect(self.setInitialRaster)
    self.btnSetFinalRaster.clicked.connect(self.setFinalRaster)
    self.btnAddFactor.clicked.connect(self.addFactor)
    self.btnRemoveFactor.clicked.connect(self.removeFactor)
    self.btnRemoveAllFactors.clicked.connect(self.removeAllFactors)

    self.btnUpdateStatistics.clicked.connect(self.updateStatisticsTable)
    self.btnCreateChangeMap.clicked.connect(self.createChangeMap)

    self.cmbSamplingMode.currentIndexChanged.connect(self.__modeChanged)
    self.cmbSimulationMethod.currentIndexChanged.connect(self.__modelChanged)

    self.chkRiskFunction.toggled.connect(self.__toggleLineEdit)
    self.chkRiskValidation.toggled.connect(self.__toggleLineEdit)
    self.chkMonteCarlo.toggled.connect(self.__toggleLineEdit)
    self.chkReuseMatrix.toggled.connect(self.__toggleLineEdit)

    self.btnSelectRiskFunction.clicked.connect(self.__selectSimulationOutput)
    self.btnSelectRiskValidation.clicked.connect(self.__selectSimulationOutput)
    self.btnSelectMonteCarlo.clicked.connect(self.__selectSimulationOutput)

    #self.btnSelectMatrix.clicked.connect()

    self.btnStartSimulation.clicked.connect(self.startSimulation)

    self.manageGui()
    self.__logMessage(self.tr("Start logging"))

  def manageGui(self):
    self.restoreGeometry(self.settings.value("/ui/geometry").toByteArray())

    self.tabWidget.setCurrentIndex(0)

    self.__populateLayers()
    self.__populateSamplingModes()
    self.__populateSimulationMethods()

    self.__readSettings()

  def closeEvent(self, e):
    self.settings.setValue("/ui/geometry", QVariant(self.saveGeometry()))

    self.__writeSettings()

    QDialog.closeEvent(self, e)

  def setInitialRaster(self):
    layerName = self.lstLayers.selectedItems()[0].text()
    self.initRasterId = self.lstLayers.selectedItems()[0].data(Qt.UserRole)
    self.leInitRasterName.setText(layerName)
    rx = QRegExp("(19|2\d)\d\d")
    pos = rx.indexIn(layerName)
    year = rx.cap()
    self.leInitYear.setText(year)

    self.inputs["initial"] = Raster(unicode(utils.getLayerById(self.initRasterId).source()))
    self.__logMessage(self.tr("Set intial layer to %1").arg(layerName))

  def setFinalRaster(self):
    layerName = self.lstLayers.selectedItems()[0].text()
    self.finalRasterId = self.lstLayers.selectedItems()[0].data(Qt.UserRole)
    self.leFinalRasterName.setText(layerName)
    rx = QRegExp("(19|2\d)\d\d")
    pos = rx.indexIn(layerName)
    year = rx.cap()
    self.leFinalYear.setText(year)

    self.inputs["final"] = Raster(unicode(utils.getLayerById(self.finalRasterId).source()))
    self.__logMessage(self.tr("Set final layer to %1").arg(layerName))

  def addFactor(self):
    layerName = self.lstLayers.selectedItems()[0].text()
    if len(self.lstFactors.findItems(layerName, Qt.MatchExactly)) > 0:
      return

    item = QListWidgetItem(self.lstLayers.selectedItems()[0])
    layerId = unicode(item.data(Qt.UserRole).toString())
    self.lstFactors.insertItem(self.lstFactors.count() + 1, item)

    if "factors" in self.inputs:
      self.inputs["factors"][layerId] = Raster(unicode(utils.getLayerById(layerId).source()))
    else:
      d = dict()
      d[layerId] = Raster(unicode(utils.getLayerById(layerId).source()))
      self.inputs["factors"] = d

    self.inputs["bandCount"] = self.__bandCount()

    self.__logMessage(self.tr("Added factor layer %1").arg(layerName))

  def removeFactor(self):
    layerId = unicode(self.lstFactors.currentItem().data(Qt.UserRole).toString())
    layerName = self.lstFactors.currentItem().text()
    self.lstFactors.takeItem(self.lstFactors.currentRow())

    del self.inputs["factors"][layerId]

    self.inputs["bandCount"] = self.__bandCount()

    self.__logMessage(self.tr("Removed factor layer %1").arg(layerName))

  def removeAllFactors(self):
    self.lstFactors.clear()

    del self.inputs["factors"]
    del self.inputs["bandCount"]

    self.__logMessage(self.tr("Factors list cleared"))

  def updateStatisticsTable(self):
    self.inputs["crosstab"] = CrossTableManager(self.inputs["initial"], self.inputs["final"])

    # class statistics
    stat = self.inputs["crosstab"].getTransitionStat()

    dimensions = len(stat["init"])
    self.tblStatistics.clear()
    self.tblStatistics.setRowCount(dimensions)
    self.tblStatistics.setColumnCount(6)

    labels = [self.leInitYear.text(),
              self.leFinalYear.text(),
              u"Δ",
              self.leInitYear.text() + " %",
              self.leFinalYear.text() + " %",
              u"Δ %"
             ]
    self.tblStatistics.setHorizontalHeaderLabels(labels)

    self.__addTableColumn(0, stat["init"])
    self.__addTableColumn(1, stat["final"])
    self.__addTableColumn(2, stat["deltas"])
    self.__addTableColumn(3, stat["initPerc"])
    self.__addTableColumn(4, stat["finalPerc"])
    self.__addTableColumn(5, stat["deltasPerc"])

    self.tblStatistics.resizeRowsToContents()
    self.tblStatistics.resizeColumnsToContents()

    # transitional matrix
    transition = self.inputs["crosstab"].getTransitionMatrix()
    dimensions = len(transition)

    self.tblTransMatrix.clear()
    self.tblTransMatrix.setRowCount(dimensions)
    self.tblTransMatrix.setColumnCount(dimensions)

    for row in xrange(0, dimensions):
      for col in xrange(0, dimensions):
        item = QTableWidgetItem(unicode(transition[row, col]))
        self.tblTransMatrix.setItem(row, col, item)

    self.tblTransMatrix.resizeRowsToContents()
    self.tblTransMatrix.resizeColumnsToContents()
    self.__logMessage(self.tr("Class statistics and transition matrix are updated"))

  def createChangeMap(self):
    fileName = utils.saveRasterDialog(self,
                                      self.settings,
                                      self.tr("Save change map"),
                                      self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
                                     )

    if fileName.isEmpty():
      return

    self.inputs["changeMapName"] = unicode(fileName)

    if ("initial" in self.inputs) and ("final" in self.inputs):
      self.analyst = AreaAnalyst(self.inputs["initial"], self.inputs["final"])
      self.analyst.moveToThread(self.workThread)
      self.workThread.started.connect(self.analyst.getChangeMap)
      self.analyst.rangeChanged.connect(self.__setProgressRange)
      self.analyst.updateProgress.connect(self.__showProgress)
      self.analyst.processFinished.connect(self.changeMapDone)
      self.analyst.processFinished.connect(self.workThread.quit)
      self.workThread.start()

  def changeMapDone(self, raster):
    self.inputs["changeMap"] = raster
    self.inputs["changeMap"].save(self.inputs["changeMapName"])
    self.__addRasterToCanvas(self.inputs["changeMapName"])
    del self.inputs["changeMapName"]
    self.workThread.started.disconnect(self.analyst.getChangeMap)
    self.analyst.rangeChanged.disconnect(self.__setProgressRange)
    self.analyst.updateProgress.disconnect(self.__showProgress)
    self.analyst.processFinished.disconnect(self.changeMapDone)
    self.analyst.processFinished.disconnect(self.workThread.quit)
    self.analyst = None
    self.__restoreProgressState()

  def startSimulation(self):
    self.simulator = Simulator(self.inputs["initial"],
                               self.inputs["factors"].values(),
                               self.inputs["model"],
                               self.inputs["crosstab"]
                              )

    self.simulator.moveToThread(self.workThread)

    self.workThread.started.connect(self.simulator.sim)
    self.simulator.rangeChanged.connect(self.__setProgressRange)
    self.simulator.updateProgress.connect(self.__showProgress)
    self.simulator.processFinished.connect(self.simulationDone)
    self.simulator.processFinished.connect(self.workThread.quit)
    self.workThread.start()

  def simulationDone(self):
    if self.chkRiskFunction.isChecked():
      if not self.leRiskFunctionPath.text().isEmpty():
        res = self.simulator.getConfidence()
        res.save(unicode(self.leRiskFunctionPath.text()))
      else:
        self.__logMessage(self.tr("Output path for risk function map is not set. Skipping this step"))

    if self.chkRiskValidation.isChecked():
      if not self.leRiskValidationPath.text().isEmpty():
        res = self.simulator.errorMap(self.inputs["final"])
        res.save(unicode(self.leRiskValidationPath.text()))
      else:
        self.__logMessage(self.tr("Output path for estimation errors for risk classes map is not set. Skipping this step"))

    if self.chkMonteCarlo.isChecked():
      if not self.leMonteCarloPath.text().isEmpty():
        res = self.simulator.getState()
        res.save(unicode(self.leMonteCarloPath.text()))
      else:
        self.__logMessage(self.tr("Output path for simulated risk map is not set. Skipping this step"))

    self.workThread.started.disconnect(self.simulator.sim)
    self.simulator.rangeChanged.disconnect(self.__setProgressRange)
    self.simulator.updateProgress.disconnect(self.__showProgress)
    self.simulator.processFinished.disconnect(self.simulationDone)
    self.simulator.processFinished.disconnect(self.workThread.quit)
    self.simulator = None
    self.__restoreProgressState()

# ******************************************************************************

  def __populateLayers(self):
    layers = utils.getRasterLayers()
    relations = self.iface.legendInterface().groupLayerRelationship()
    for layer in sorted(layers.iteritems(), cmp=locale.strcoll, key=operator.itemgetter(1)):
      groupName = utils.getLayerGroup(relations, layer[0])
      item = QListWidgetItem()
      if groupName == "":
        item.setText(layer[1])
        item.setData(Qt.UserRole, layer[0])
      else:
        item.setText(QString("%1 - %2").arg(layer[1]).arg(groupName))
        item.setData(Qt.UserRole, layer[0])

      self.lstLayers.addItem(item)

  def __populateSimulationMethods(self):
    self.cmbSimulationMethod.addItems([
                                       self.tr("Artificial Neural Network"),
                                       self.tr("Weights of Evidence"),
                                       self.tr("Multi Criteria Evaluation")
                                     ])
    if not sklearnMissed:
      self.cmbSimulationMethod.addItem(self.tr("Logistic Regression"))

  def __populateSamplingModes(self):
    self.cmbSamplingMode.addItem(self.tr("All"), 0)
    self.cmbSamplingMode.addItem(self.tr("Normal"), 1)
    self.cmbSamplingMode.addItem(self.tr("Balanced"), 2)

  def __modeChanged(self, index):
    mode = self.cmbSamplingMode.itemData(index).toInt()[0]
    if mode == 0:
      self.inputs["samplingMode"] = "All"
    elif mode == 1:
      self.inputs["samplingMode"] = "Normal"
    elif mode == 2:
      self.inputs["samplingMode"] = "Balanced"

  def __modelChanged(self):
    if self.modelWidget is not None:
      self.widgetStackMethods.removeWidget(self.modelWidget)

      self.modelWidget = None
      del self.modelWidget

    modelName = self.cmbSimulationMethod.currentText()

    if modelName == self.tr("Logistic Regression"):
      self.modelWidget = logisticregressionwidget.LogisticRegressionWidget(self)
    elif modelName == self.tr("Artificial Neural Network"):
      self.modelWidget = neuralnetworkwidget.NeuralNetworkWidget(self)
    elif modelName == self.tr("Weights of Evidence"):
      self.modelWidget = weightofevidencewidget.WeightOfEvidenceWidget(self)
    elif modelName == self.tr("Multi Criteria Evaluation"):
      self.modelWidget = multicriteriaevaluationwidget.MultiCriteriaEvaluationWidget(self)

    self.widgetStackMethods.addWidget(self.modelWidget)
    self.widgetStackMethods.setCurrentWidget(self.modelWidget)

  def __toggleLineEdit(self, checked):
    senderName = self.sender().objectName()
    if senderName == "chkRiskFunction":
      if checked:
        self.leRiskFunctionPath.setEnabled(True)
        self.btnSelectRiskFunction.setEnabled(True)
      else:
        self.leRiskFunctionPath.setEnabled(False)
        self.btnSelectRiskFunction.setEnabled(False)
    elif senderName == "chkRiskValidation":
      if checked:
        self.leRiskValidationPath.setEnabled(True)
        self.btnSelectRiskValidation.setEnabled(True)
      else:
        self.leRiskValidationPath.setEnabled(False)
        self.btnSelectRiskValidation.setEnabled(False)
    elif senderName == "chkMonteCarlo":
      if checked:
        self.leMonteCarloPath.setEnabled(True)
        self.btnSelectMonteCarlo.setEnabled(True)
        self.lblIterations.setEnabled(True)
        self.spnIterations.setEnabled(True)
      else:
        self.leMonteCarloPath.setEnabled(False)
        self.btnSelectMonteCarlo.setEnabled(False)
        self.lblIterations.setEnabled(False)
        self.spnIterations.setEnabled(False)
    elif senderName == "chkReuseMatrix":
      if checked:
        self.leMatrixPath.setEnabled(True)
        self.btnSelectMatrix.setEnabled(True)
      else:
        self.leMatrixPath.setEnabled(False)
        self.btnSelectMatrix.setEnabled(False)

  def __selectSimulationOutput(self):
    senderName = self.sender().objectName()

    fileName = utils.saveRasterDialog(self,
                                      self.settings,
                                      self.tr("Save file"),
                                      self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
                                     )
    if fileName.isEmpty():
      return

    if senderName == "btnSelectRiskFunction":
      self.leRiskFunctionPath.setText(fileName)
    elif senderName == "btnSelectRiskValidation":
      self.leRiskValidationPath.setText(fileName)
    elif senderName == "btnSelectMonteCarlo":
      self.leMonteCarloPath.setText(fileName)

  def __logMessage(self, message):
    self.txtMessages.append(QString("[%1] %2")
                            .arg(datetime.datetime.now().strftime("%a %b %d %Y %H:%M:%S"))
                            .arg(message)
                           )

  def __addTableColumn(self, col, values):
    dimensions = len(values)
    for r in xrange(0, dimensions):
      item = QTableWidgetItem(unicode(values[r]))
      self.tblStatistics.setItem(r, col, item)

  def __addRasterToCanvas(self, filePath):
    layer = QgsRasterLayer(filePath, QFileInfo(filePath).baseName())
    if layer.isValid():
      QgsMapLayerRegistry.instance().addMapLayers([layer])
    else:
      self.__logMessage(self.tr("Can't load raster %1").arg(filePath))

  def __bandCount(self):
    bands = 0
    for k, v in self.inputs["factors"].iteritems():
      bands += len(v.bands)
    return bands

  def __setProgressRange(self, message, maxValue):
    self.progressBar.setFormat(message)
    self.progressBar.setRange(0, maxValue)

  def __showProgress(self):
    self.progressBar.setValue(self.progressBar.value() + 1)

  def __restoreProgressState(self):
    self.progressBar.setFormat("%p%")
    self.progressBar.setRange(0, 1)
    self.progressBar.setValue(0)

  def __writeSettings(self):
    # samples and model tab
    self.settings.setValue("ui/samplingMode", self.cmbSamplingMode.itemData(self.cmbSamplingMode.currentIndex()).toInt()[0])
    self.settings.setValue("ui/samplesCount", self.spnSamplesCount.value())

    # simulation tab
    self.settings.setValue("ui/createRiskFunction", self.chkRiskFunction.isChecked())
    self.settings.setValue("ui/createRiskValidation", self.chkRiskValidation.isChecked())
    self.settings.setValue("ui/createMonteCarlo", self.chkMonteCarlo.isChecked())
    self.settings.setValue("ui/monteCarloIterations", self.spnIterations.value())

    self.settings.setValue("ui/reuseMatrix", self.chkReuseMatrix.isChecked())

  def __readSettings(self):
    # samples and model tab
    samplingMode = self.settings.value("ui/samplingMode", 0).toInt()[0]
    self.cmbSamplingMode.setCurrentIndex(self.cmbSamplingMode.findData(samplingMode))
    self.spnSamplesCount.setValue(self.settings.value("ui/samplesCount", 10000).toInt()[0])

    # simulation tab
    self.chkRiskFunction.setChecked(self.settings.value("ui/createRiskFunction", False).toBool())
    self.chkRiskValidation.setChecked(self.settings.value("ui/createRiskValidation", False).toBool())
    self.chkMonteCarlo.setChecked(self.settings.value("ui/createMonteCarlo", False).toBool())
    self.spnIterations.setValue(self.settings.value("ui/monteCarloIterations", 1).toInt()[0])

    self.chkReuseMatrix.setChecked(self.settings.value("ui/reuseMatrix", False).toBool())
