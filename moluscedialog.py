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

import numpy
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams

import neuralnetworkwidget
import weightofevidencewidget
import multicriteriaevaluationwidget
import logisticregressionwidget

from ui.ui_moluscedialogbase import Ui_Dialog

from algorithms.dataprovider import Raster, ProviderError
from algorithms.models.correlation.model import DependenceCoef, CoeffError
from algorithms.models.crosstabs.manager import CrossTableManager
from algorithms.models.area_analysis.manager import AreaAnalyst
from algorithms.models.simulator.sim import Simulator
from algorithms.models.errorbudget.ebmodel import EBudget

import molusceutils as utils

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
    #  "crosstab" : list,
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

    self.chkAllCorr.stateChanged.connect(self.__toggleCorrLayers)
    self.btnStartCorrChecking.clicked.connect(self.correlationChecking)

    self.btnUpdateStatistics.clicked.connect(self.startUpdateStatisticsTable)
    self.btnCreateChangeMap.clicked.connect(self.createChangeMap)

    self.cmbSamplingMode.currentIndexChanged.connect(self.__modeChanged)
    self.cmbSimulationMethod.currentIndexChanged.connect(self.__modelChanged)

    self.btnSelectSamples.clicked.connect(self.__selectSamplesOutput)

    self.chkRiskFunction.toggled.connect(self.__toggleLineEdit)
    self.chkRiskValidation.toggled.connect(self.__toggleLineEdit)
    self.chkMonteCarlo.toggled.connect(self.__toggleLineEdit)

    self.btnSelectRiskFunction.clicked.connect(self.__selectSimulationOutput)
    self.btnSelectRiskValidation.clicked.connect(self.createValidationMap)
    self.btnSelectMonteCarlo.clicked.connect(self.__selectSimulationOutput)

    self.btnStartSimulation.clicked.connect(self.startSimulation)

    self.btnSelectSimulatedMap.clicked.connect(self.__selectValidationMap)
    self.btnSelectReferenceMap.clicked.connect(self.__selectValidationMap)

    self.btnStartValidation.clicked.connect(self.startValidation)

    self.btnKappaCalc.clicked.connect(self.startKappaValidation)

    self.tabWidget.currentChanged.connect(self.tabChanged)

    self.manageGui()
    self.logMessage(self.tr("Start logging"))

  def manageGui(self):
    self.restoreGeometry(self.settings.value("/ui/geometry").toByteArray())

    self.tabWidget.setCurrentIndex(0)

    self.__populateLayers()
    self.__populateCorrCheckingMet()
    self.__populateSamplingModes()
    self.__populateSimulationMethods()
    self.__populateRasterNames()
    self.__populateValidationPlot()

    self.__readSettings()

  def closeEvent(self, e):
    self.settings.setValue("/ui/geometry", QVariant(self.saveGeometry()))

    self.__writeSettings()

    QDialog.closeEvent(self, e)

  def setInitialRaster(self):
    try:
      layerName = self.lstLayers.selectedItems()[0].text()
      self.initRasterId = self.lstLayers.selectedItems()[0].data(Qt.UserRole)
      self.leInitRasterName.setText(layerName)
    except IndexError:
      QMessageBox.warning(self,
                          self.tr("Missed selected row"),
                          self.tr("Initial raster is not selected. Please specify input data and try again")
                         )
      return
    rx = QRegExp("(19|2\d)\d\d")
    pos = rx.indexIn(layerName)
    year = rx.cap()
    self.leInitYear.setText(year)

    self.inputs["initial"] = Raster(unicode(utils.getLayerById(self.initRasterId).source()))
    self.logMessage(self.tr("Set intial layer to %1").arg(layerName))

  def setFinalRaster(self):
    try:
      layerName = self.lstLayers.selectedItems()[0].text()
      self.finalRasterId = self.lstLayers.selectedItems()[0].data(Qt.UserRole)
      self.leFinalRasterName.setText(layerName)
    except IndexError:
      QMessageBox.warning(self,
                          self.tr("Missed selected row"),
                          self.tr("Final raster is not selected. Please specify input data and try again")
                         )
      return
    rx = QRegExp("(19|2\d)\d\d")
    pos = rx.indexIn(layerName)
    year = rx.cap()
    self.leFinalYear.setText(year)

    self.inputs["final"] = Raster(unicode(utils.getLayerById(self.finalRasterId).source()))
    self.logMessage(self.tr("Set final layer to %1").arg(layerName))

  def addFactor(self):
    try:
      layerName = self.lstLayers.selectedItems()[0].text()
    except IndexError:
      QMessageBox.warning(self,
                          self.tr("Missed selected row"),
                          self.tr("Factor raster is not selected. Please specify input data and try again")
                         )
      return
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

    self.logMessage(self.tr("Added factor layer %1").arg(layerName))

  def removeFactor(self):
    try:
      layerId = unicode(self.lstFactors.currentItem().data(Qt.UserRole).toString())
    except AttributeError:
      QMessageBox.warning(self,
                          self.tr("Missed selected row"),
                          self.tr("Factor raster is not selected. Please specify input data and try again")
                         )
      return
    layerName = self.lstFactors.currentItem().text()
    self.lstFactors.takeItem(self.lstFactors.currentRow())

    del self.inputs["factors"][layerId]
    if self.inputs["factors"] == {}:
      del self.inputs["factors"]
      del self.inputs["bandCount"]
    else:
      self.inputs["bandCount"] = self.__bandCount()

    self.logMessage(self.tr("Removed factor layer %1").arg(layerName))

  def removeAllFactors(self):
    self.lstFactors.clear()
    try:
      del self.inputs["factors"]
      del self.inputs["bandCount"]
    except KeyError:
      pass

    self.logMessage(self.tr("Factors list cleared"))

  def correlationChecking(self):
    if self.chkAllCorr.isChecked():
      self.__checkAllCorr()
    else:
      self.__checkTwoCorr()

    self.tblCorrelation.resizeRowsToContents()
    self.tblCorrelation.resizeColumnsToContents()

  def startUpdateStatisticsTable(self):
    if not utils.checkInputRasters(self.inputs):
      QMessageBox.warning(self,
                          self.tr("Missed input data"),
                          self.tr("Initial or final raster is not set. Please specify input data and try again")
                         )
      return

    crossTabMan = CrossTableManager(self.inputs["initial"], self.inputs["final"])
    self.inputs["crosstab"] = crossTabMan

    # class statistics
    crossTabMan.moveToThread(self.workThread)

    self.workThread.started.connect(crossTabMan.computeCrosstable)
    crossTabMan.rangeChanged.connect(self.setProgressRange)
    crossTabMan.updateProgress.connect(self.showProgress)
    crossTabMan.crossTableFinished.connect(self.updateStatisticsTableDone)
    self.workThread.start()

  def updateStatisticsTableDone(self):
    crossTabMan = self.inputs["crosstab"]
    self.workThread.started.disconnect(crossTabMan.computeCrosstable)
    crossTabMan.rangeChanged.disconnect(self.setProgressRange)
    crossTabMan.updateProgress.disconnect(self.showProgress)
    crossTabMan.crossTableFinished.disconnect(self.updateStatisticsTableDone)
    self.workThread.quit()
    self.restoreProgressState()

    stat = self.inputs["crosstab"].getTransitionStat()
    dimensions = len(stat["init"])
    self.tblStatistics.clear()
    self.tblStatistics.setRowCount(dimensions)
    self.tblStatistics.setColumnCount(7)

    labels = []
    colors = []
    layer = utils.getLayerById(self.initRasterId)
    if layer.renderer().type().contains("singlebandpseudocolor"):
      legend = layer.legendSymbologyItems()
      for i in legend:
        labels.append(unicode(i[0]))
        colors.append(i[1])
    else:
      labels = [unicode(i) for i in xrange(1, 7)]

    self.tblStatistics.setVerticalHeaderLabels(labels)

    labels = [self.tr("Class color"),
              self.leInitYear.text(),
              self.leFinalYear.text(),
              u"Δ",
              self.leInitYear.text() + " %",
              self.leFinalYear.text() + " %",
              u"Δ %"
             ]
    self.tblStatistics.setHorizontalHeaderLabels(labels)

    # legend colors
    d = len(colors)
    if d > 0:
      for i in xrange(0, d):
        item = QTableWidgetItem("")
        item.setBackground(QBrush(colors[i]))
        self.tblStatistics.setItem(i, 0, item)

    self.__addTableColumn(1, stat["init"])
    self.__addTableColumn(2, stat["final"])
    self.__addTableColumn(3, stat["deltas"])
    self.__addTableColumn(4, stat["initPerc"])
    self.__addTableColumn(5, stat["finalPerc"])
    self.__addTableColumn(6, stat["deltasPerc"])

    self.tblStatistics.resizeRowsToContents()
    self.tblStatistics.resizeColumnsToContents()

    # transitional matrix
    transition = self.inputs["crosstab"].getTransitionMatrix()
    dimensions = len(transition)

    self.tblTransMatrix.clear()
    self.tblTransMatrix.setRowCount(dimensions)
    self.tblTransMatrix.setColumnCount(dimensions)

    labels = []
    layer = utils.getLayerById(self.initRasterId)
    if layer.renderer().type().contains("singlebandpseudocolor"):
      legend = layer.legendSymbologyItems()
      for i in legend:
        labels.append(unicode(i[0]))
    else:
      labels = [unicode(i) for i in xrange(1, dimensions + 1)]

    self.tblTransMatrix.setVerticalHeaderLabels(labels)
    self.tblTransMatrix.setHorizontalHeaderLabels(labels)

    for row in xrange(0, dimensions):
      for col in xrange(0, dimensions):
        item = QTableWidgetItem(unicode(transition[row, col]))
        self.tblTransMatrix.setItem(row, col, item)

    self.tblTransMatrix.resizeRowsToContents()
    self.tblTransMatrix.resizeColumnsToContents()
    self.logMessage(self.tr("Class statistics and transition matrix are updated"))

  def createChangeMap(self):
    if not utils.checkInputRasters(self.inputs):
      QMessageBox.warning(self,
                          self.tr("Missed input data"),
                          self.tr("Initial or final raster is not set. Please specify input data and try again")
                         )
      return

    fileName = utils.saveRasterDialog(self,
                                      self.settings,
                                      self.tr("Save change map"),
                                      self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
                                     )

    if fileName.isEmpty():
      self.logMessage(self.tr("No file selected"))
      return

    self.inputs["changeMapName"] = unicode(fileName)

    self.analyst = AreaAnalyst(self.inputs["initial"], self.inputs["final"])
    self.analyst.moveToThread(self.workThread)
    self.workThread.started.connect(self.analyst.getChangeMap)
    self.analyst.rangeChanged.connect(self.setProgressRange)
    self.analyst.updateProgress.connect(self.showProgress)
    self.analyst.processFinished.connect(self.changeMapDone)
    self.analyst.processFinished.connect(self.workThread.quit)
    self.workThread.start()

  def changeMapDone(self, raster):
    self.inputs["changeMap"] = raster
    self.inputs["changeMap"].save(self.inputs["changeMapName"])
    self.__addRasterToCanvas(self.inputs["changeMapName"])
    self.applyRasterStyleLabels(utils.getLayerByName(QFileInfo(self.inputs["changeMapName"]).baseName()), self.analyst, False)
    del self.inputs["changeMapName"]
    self.workThread.started.disconnect(self.analyst.getChangeMap)
    self.analyst.rangeChanged.disconnect(self.setProgressRange)
    self.analyst.updateProgress.disconnect(self.showProgress)
    self.analyst.processFinished.disconnect(self.changeMapDone)
    self.analyst.processFinished.disconnect(self.workThread.quit)
    self.restoreProgressState()

  def startSimulation(self):
    if not utils.checkInputRasters(self.inputs):
      QMessageBox.warning(self,
                          self.tr("Missed input data"),
                          self.tr("Initial raster is not set. Please specify it and try again")
                         )
      return

    if not utils.checkFactors(self.inputs):
      QMessageBox.warning(self,
                          self.tr("Missed input data"),
                          self.tr("Factors rasters is not set. Please specify them and try again")
                         )
      return

    if not "model" in self.inputs:
      QMessageBox.warning(self,
                          self.tr("Missed model"),
                          self.tr("Model not selected please select and train model.")
                         )
      return

    if not "crosstab" in self.inputs:
      QMessageBox.warning(self,
                          self.tr("Missed transition matrix"),
                          self.tr("Please calculate transition matrix and try again")
                         )
      return

    self.simulator = Simulator(self.inputs["initial"],
                               self.inputs["factors"].values(),
                               self.inputs["model"],
                               self.inputs["crosstab"]
                              )

    self.simulator.moveToThread(self.workThread)

    self.btnStartSimulation.setEnabled(False)

    self.workThread.started.connect(self.propagateSimulation)
    self.simulator.rangeChanged.connect(self.setProgressRange)
    self.simulator.updateProgress.connect(self.showProgress)
    self.simulator.simFinished.connect(self.simulationDone)
    self.workThread.start()

  def simulationDone(self):
    self.btnStartSimulation.setEnabled(True)
    if self.chkRiskFunction.isChecked():
      if not self.leRiskFunctionPath.text().isEmpty():
        res = self.simulator.getConfidence()
        grad = res.getBandGradation(1)
        saved = False
        # Try to use some Values as No-data Value
        maxVal = res.getGDALMaxVal()
        for noData in [0, maxVal]:
            if not noData in grad:
                res.save(unicode(self.leRiskFunctionPath.text()), nodata=noData)
                saved = True
                break
        if not saved:
            res.save(unicode(self.leRiskFunctionPath.text()), nodata=maxVal-1)
      else:
        self.logMessage(self.tr("Output path for risk function map is not set. Skipping this step"))

    if self.chkMonteCarlo.isChecked():
      if not self.leMonteCarloPath.text().isEmpty():
        res = self.simulator.getState()
        grad = res.getBandGradation(1)
        saved = False
        # Try to use some Values as No-data Value
        maxVal = res.getGDALMaxVal()
        for noData in [0, maxVal]:
            if not noData in grad:
                res.save(unicode(self.leMonteCarloPath.text()), nodata=noData)
                saved = True
                break
        if not saved:
            res.save(unicode(self.leRiskFunctionPath.text()), nodata=maxVal-1)
        self.__addRasterToCanvas(self.leMonteCarloPath.text())
        if utils.copySymbology(utils.getLayerByName(self.leInitRasterName.text()), utils.getLayerByName(QFileInfo(self.leMonteCarloPath.text()).baseName())):
          self.iface.legendInterface().refreshLayerSymbology(utils.getLayerByName(QFileInfo(self.leMonteCarloPath.text()).baseName()))
          self.iface.mapCanvas().refresh()
          QgsProject.instance().dirty(True)
      else:
        self.logMessage(self.tr("Output path for simulated risk map is not set. Skipping this step"))

    self.workThread.started.disconnect(self.propagateSimulation)
    self.simulator.rangeChanged.disconnect(self.setProgressRange)
    self.simulator.updateProgress.disconnect(self.showProgress)
    self.simulator.simFinished.disconnect(self.simulationDone)
    self.workThread.quit()
    self.simulator = None
    self.restoreProgressState()

  def startValidation(self):
    try:
      reference = Raster(unicode(self.leReferenceMapPath.text()))
    except ProviderError:
      QMessageBox.warning(self,
                          self.tr("Can't read file"),
                          self.tr("Can't read file: '%s'" % unicode(self.leReferenceMapPath.text()))
                         )
      return
    try:
      simulated = Raster(unicode(self.leSimulatedMapPath.text()))
    except ProviderError:
      QMessageBox.warning(self,
                          self.tr("Can't read file"),
                          self.tr("Can't read file: '%s'" % unicode(self.leSimulatedMapPath.text()))
                         )
      return

    self.eb = EBudget(reference, simulated)

    self.eb.moveToThread(self.workThread)

    self.workThread.started.connect(self.validate)
    self.eb.rangeChanged.connect(self.setProgressRange)
    self.eb.updateProgress.connect(self.showProgress)
    self.eb.validationFinished.connect(self.validationDone)
    self.workThread.start()

  def validationDone(self, stat):
    self.workThread.started.disconnect(self.validate)
    self.eb.rangeChanged.disconnect(self.setProgressRange)
    self.eb.updateProgress.disconnect(self.showProgress)
    self.eb.validationFinished.disconnect(self.validationDone)
    self.workThread.quit()
    self.eb = None
    self.restoreProgressState()

    self.scaleData = stat.keys()
    self.noNoData, self.noMedData, self.medMedData, self.medPerData, self.perPerData = [], [], [], [], []
    for k in stat.keys():
      self.noNoData.append(stat[k]['NoNo'])
      self.noMedData.append(stat[k]['NoMed'])
      self.medMedData.append(stat[k]['MedMed'])
      self.medPerData.append(stat[k]['MedPer'])
      self.perPerData.append(stat[k]['PerPer'])

    self.valAxes.set_xbound(lower=0, upper=len(self.scaleData)-1)
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

  def startKappaValidation(self):
    try:
      reference = Raster(unicode(self.leReferenceMapPath.text()))
    except ProviderError:
      QMessageBox.warning(self,
                          self.tr("Can't read file"),
                          self.tr("Can't read file: '%s'" % unicode(self.leReferenceMapPath.text()))
                         )
      return
    try:
      simulated = Raster(unicode(self.leSimulatedMapPath.text()))
    except ProviderError:
      QMessageBox.warning(self,
                          self.tr("Can't read file"),
                          self.tr("Can't read file: '%s'" % unicode(self.leSimulatedMapPath.text()))
                         )
      return
    for raster in [reference, simulated]:
      if raster.isCountinues(bandNo=1):
        QMessageBox.warning(self,
                            self.tr("Kappa is not applicable"),
                            self.tr("Kappa is not applicable to the file: '%s' because it's contains continues value" % unicode(raster.getFileName()))
                           )
        return

    # Kappa
    self.depCoef = DependenceCoef(reference.getBand(1), simulated.getBand(1), expand=True)

    self.depCoef.moveToThread(self.workThread)

    self.workThread.started.connect(self.depCoef.calculateCrosstable)
    self.depCoef.rangeChanged.connect(self.setProgressRange)
    self.depCoef.updateProgress.connect(self.showProgress)
    self.depCoef.processFinished.connect(self.kappaValDone)
    self.workThread.start()

  def kappaValDone(self):
    self.workThread.started.disconnect(self.depCoef.calculateCrosstable)
    self.depCoef.rangeChanged.disconnect(self.setProgressRange)
    self.depCoef.updateProgress.disconnect(self.showProgress)
    self.depCoef.processFinished.disconnect(self.kappaValDone)
    self.workThread.quit()
    self.restoreProgressState()

    kappas = self.depCoef.kappa(mode='all')
    self.leKappaOveral.setText(QString.number(kappas["overal"]))
    self.leKappaHisto.setText(QString.number(kappas["histo"]))
    self.leKappaLoc.setText(QString.number(kappas["loc"]))
    # % of Correctness
    percent = self.depCoef.correctness()
    self.leKappaCorrectness.setText(QString.number(percent))
    self.depCoef = None

  def createValidationMap(self):
    try:
      reference = Raster(unicode(self.leReferenceMapPath.text()))
    except ProviderError:
      QMessageBox.warning(self,
                          self.tr("Can't read file"),
                          self.tr("Can't read file: '%s'" % unicode(self.leReferenceMapPath.text()))
                         )
      return
    try:
      simulated = Raster(unicode(self.leSimulatedMapPath.text()))
    except ProviderError:
      QMessageBox.warning(self,
                          self.tr("Can't read file"),
                          self.tr("Can't read file: '%s'" % unicode(self.leSimulatedMapPath.text()))
                         )
      return

    fileName = utils.saveRasterDialog(self,
                                      self.settings,
                                      self.tr("Save validation map"),
                                      self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
                                     )

    if fileName.isEmpty():
      self.logMessage(self.tr("No file selected"))
      return

    self.inputs["valMapName"] = unicode(fileName)

    self.analystVM = AreaAnalyst(reference, simulated)
    self.analystVM.moveToThread(self.workThread)
    self.workThread.started.connect(self.analystVM.getChangeMap)
    self.analystVM.rangeChanged.connect(self.setProgressRange)
    self.analystVM.updateProgress.connect(self.showProgress)
    self.analystVM.processFinished.connect(self.validationMapDone)
    self.analystVM.processFinished.connect(self.workThread.quit)
    self.workThread.start()

  def validationMapDone(self, raster):
    validationMap = raster
    validationMap.save(self.inputs["valMapName"])
    self.__addRasterToCanvas(self.inputs["valMapName"])
    self.applyRasterStyleLabels(utils.getLayerByName(QFileInfo(self.inputs["valMapName"]).baseName()), self.analystVM, True)
    del self.inputs["valMapName"]
    self.workThread.started.disconnect(self.analystVM.getChangeMap)
    self.analystVM.rangeChanged.disconnect(self.setProgressRange)
    self.analystVM.updateProgress.disconnect(self.showProgress)
    self.analystVM.processFinished.disconnect(self.validationMapDone)
    self.analystVM.processFinished.disconnect(self.workThread.quit)
    del self.analystVM
    self.restoreProgressState()

  def tabChanged(self, index):
    if  index == 1:     # tabCorrelationChecking
      self.__populateRasterNames()

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

  def __populateRasterNames(self):
    self.cmbFirstRaster.clear()
    self.cmbSecondRaster.clear()
    for index in xrange(self.lstFactors.count()):
      item = self.lstFactors.item(index)
      self.cmbFirstRaster.addItem(item.text(), item.data(Qt.UserRole))
      self.cmbSecondRaster.addItem(item.text(), item.data(Qt.UserRole))

  def __populateCorrCheckingMet(self):
    self.cmbCorrCheckMethod.addItems([
                                       self.tr("Correlation"),
                                       self.tr("Kappa (classic)"),
                                       self.tr("Kappa (loc)"),
                                       self.tr("Kappa (histo)"),
                                       self.tr("Cramer's Coefficient"),
                                       self.tr("Joint Information Uncertainty")
                                     ])

  def __populateSimulationMethods(self):
    self.cmbSimulationMethod.addItems([
                                       self.tr("Artificial Neural Network"),
                                       self.tr("Weights of Evidence"),
                                       self.tr("Multi Criteria Evaluation"),
                                       self.tr("Logistic Regression")
                                     ])

  def __populateSamplingModes(self):
    self.cmbSamplingMode.addItem(self.tr("All"), 0)
    self.cmbSamplingMode.addItem(self.tr("Normal"), 1)
    self.cmbSamplingMode.addItem(self.tr("Balanced"), 2)

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
    self.noNo = self.valAxes.plot(self.noNoData,
                            linewidth=1,
                            color="green", linestyle='dashed', marker='o',
                            )[0]
    self.noMedData = []
    self.noMed = self.valAxes.plot(self.noMedData,
                            linewidth=1,
                            color="red", marker='o',
                            )[0]
    self.medMedData = []
    self.medMed = self.valAxes.plot(self.medMedData,
                            linewidth=1,
                            color="purple", linestyle='dashed', marker='v',
                            )[0]
    self.medPerData = []
    self.medPer = self.valAxes.plot(self.medPerData,
                            linewidth=1,
                            color="black", linestyle='dashed', marker='+',
                            )[0]
    self.perPerData = []
    self.perPer = self.valAxes.plot(self.perPerData,
                            linewidth=1,
                            color="yellow", marker='*',
                            )[0]
    box = self.valAxes.get_position()
    self.valAxes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    leg = self.valAxes.legend(('No location, no quantity inform.', 'No location, medium quantity inform.', 'Medium location, medium quantity inform.', 'Perfect location, medium quantity inform.', 'Perfect location, perfect quantity inform.'), loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3, shadow=False)
    for t in leg.get_texts():
        t.set_fontsize('small')

  def __checkAllCorr(self):
    dim = self.__bandCount()
    self.tblCorrelation.setRowCount(dim)
    self.tblCorrelation.setColumnCount(dim)

    labels = []
    mapping, labNo = {}, 0    # Maping between raster ID and label number
    for k, v in self.inputs["factors"].iteritems():
      mapping[k] = {}
      for b in xrange(v.getBandsCount()):
        if v.getBandsCount()>1:
          name = QString(u"%s (band %s)" % (utils.getLayerById(k).name(), unicode(b+1)))
        else:
          name = QString(u"%s" % (utils.getLayerById(k).name(), ))
        mapping[k][b] = labNo
        labNo = labNo + 1
        labels.append(name)
    self.tblCorrelation.setVerticalHeaderLabels(labels)
    self.tblCorrelation.setHorizontalHeaderLabels(labels)

    method = self.cmbCorrCheckMethod.currentText()
    discreteMethods = [ # The methods need categorial values
            self.tr("Kappa (classic)"), self.tr("Kappa (loc)"), self.tr("Kappa (histo)"),
            self.tr("Cramer's Coefficient"), self.tr("Joint Information Uncertainty")
    ]
    # Loop over all rasters and all bands
    self.setProgressRange(self.tr('Correlation checking'), dim*(dim-1)/2)
    for i, fact1 in self.inputs["factors"].iteritems():
        for b1 in range(fact1.getBandsCount()):
          labNo1 = mapping[i][b1]
          for j, fact2 in self.inputs["factors"].iteritems():
            for b2 in range(fact2.getBandsCount()):
              labNo2 = mapping[j][b2]
              if labNo2 < labNo1:
                continue
              if labNo2==labNo1:
                item = QTableWidgetItem(unicode("--"))
              # Check if method is applicable to the bands
              elif (fact1.isCountinues(b1+1) or fact2.isCountinues(b2+1)) and  method in discreteMethods:
                item = QTableWidgetItem(unicode(self.tr("Not applicable")))
              else:
                depCoef = DependenceCoef(fact1.getBand(b1+1), fact2.getBand(b2+1))
                if method == self.tr("Correlation"):
                  coef = depCoef.correlation()
                elif method ==self.tr("Kappa (classic)"):
                  coef = depCoef.kappa(mode=None)
                elif method ==self.tr("Kappa (loc)"):
                  coef = depCoef.kappa(mode='loc')
                elif method ==self.tr("Kappa (histo)"):
                  coef = depCoef.kappa(mode='histo')
                elif method ==self.tr("Joint Information Uncertainty"):
                  coef = depCoef.jiu()
                elif method ==self.tr("Cramer's Coefficient"):
                  coef = depCoef.cramer()
                item = QTableWidgetItem(unicode(coef))
              self.tblCorrelation.setItem(labNo1, labNo2, item)
              self.showProgress()
    self.restoreProgressState()

  def __checkTwoCorr(self):
    index = self.cmbFirstRaster.currentIndex()
    layerId = unicode(self.cmbFirstRaster.itemData(index, Qt.UserRole).toString())
    first = {'Raster': self.inputs["factors"][layerId], 'Name': self.cmbFirstRaster.currentText()}
    index = self.cmbSecondRaster.currentIndex()
    layerId = unicode(self.cmbSecondRaster.itemData(index, Qt.UserRole).toString())
    second = {'Raster': self.inputs["factors"][layerId], 'Name': self.cmbSecondRaster.currentText()}

    dimensions = first['Raster'].getBandsCount(), second['Raster'].getBandsCount()
    self.tblCorrelation.setRowCount(dimensions[0])
    self.tblCorrelation.setColumnCount(dimensions[1])
    labels = []
    for i in range(dimensions[0]):
      raster = first["Raster"]
      if raster.getBandsCount()>1:
        name = QString(u"%s (band %s)" % (first['Name'], unicode(i+1)))
      else:
        name = QString(u"%s" % (first['Name'], ))
      labels.append(name)
    self.tblCorrelation.setVerticalHeaderLabels(labels)
    labels = []
    for i in range(dimensions[1]):
      raster = second["Raster"]
      if raster.getBandsCount()>1:
        name = QString(u"%s (band %s)" % (second['Name'], unicode(i+1)))
      else:
        name = QString(u"%s" % (second['Name'], ))
      labels.append(name)
    self.tblCorrelation.setHorizontalHeaderLabels(labels)

    method = self.cmbCorrCheckMethod.currentText()
    if method == self.tr("Correlation"):
      for col in xrange(dimensions[1]):
        for row in xrange(dimensions[0]):
          depCoef = DependenceCoef(first["Raster"].getBand(row+1), second["Raster"].getBand(col + 1))
          corr = depCoef.correlation()
          item = QTableWidgetItem(unicode(corr))
          self.tblCorrelation.setItem(row, col, item)
    elif method == self.tr("Kappa (classic)"):
      try:
        for col in xrange(dimensions[1]):
          for row in xrange(dimensions[0]):
            depCoef = DependenceCoef(first["Raster"].getBand(row+1), second["Raster"].getBand(col + 1))
            if first["Raster"].isCountinues(row+1) or second["Raster"].isCountinues(col + 1):
              item = QTableWidgetItem(unicode(self.tr("Not applicable")))
            else:
              corr = depCoef.kappa(mode=None)
              item = QTableWidgetItem(unicode(corr))
            self.tblCorrelation.setItem(row, col, item)
      except CoeffError as ex:
        QMessageBox.warning(self,
                          self.tr("Checking"),
                          ex.msg
                         )
    elif method == self.tr("Kappa (loc)"):
      try:
        for col in xrange(dimensions[1]):
          for row in xrange(dimensions[0]):
            depCoef = DependenceCoef(first["Raster"].getBand(row+1), second["Raster"].getBand(col + 1))
            if first["Raster"].isCountinues(row+1) or second["Raster"].isCountinues(col + 1):
              item = QTableWidgetItem(unicode(self.tr("Not applicable")))
            else:
              corr = depCoef.kappa(mode=None)
              item = QTableWidgetItem(unicode(corr))
            self.tblCorrelation.setItem(row, col, item)
      except CoeffError as ex:
        QMessageBox.warning(self,
                          self.tr("Checking"),
                          ex.msg
                         )
    elif method == self.tr("Kappa (histo)"):
      try:
        for col in xrange(dimensions[1]):
          for row in xrange(dimensions[0]):
            depCoef = DependenceCoef(first["Raster"].getBand(row+1), second["Raster"].getBand(col + 1))
            if first["Raster"].isCountinues(row+1) or second["Raster"].isCountinues(col + 1):
              item = QTableWidgetItem(unicode(self.tr("Not applicable")))
            else:
              corr = depCoef.kappa(mode=None)
              item = QTableWidgetItem(unicode(corr))
            self.tblCorrelation.setItem(row, col, item)
      except CoeffError as ex:
        QMessageBox.warning(self,
                          self.tr("Checking"),
                          ex.msg
                         )
    elif method == self.tr("Cramer's Coefficient"):
      for col in xrange(dimensions[1]):
        for row in xrange(dimensions[0]):
          depCoef = DependenceCoef(first["Raster"].getBand(row+1), second["Raster"].getBand(col + 1))
          if first["Raster"].isCountinues(row+1) or second["Raster"].isCountinues(col + 1):
              item = QTableWidgetItem(unicode(self.tr("Not applicable")))
          else:
              corr = depCoef.kappa(mode=None)
              item = QTableWidgetItem(unicode(corr))
          self.tblCorrelation.setItem(row, col, item)
    elif method == self.tr("Joint Information Uncertainty"):
      for col in xrange(dimensions[1]):
        for row in xrange(dimensions[0]):
          depCoef = DependenceCoef(first["Raster"].getBand(row+1), second["Raster"].getBand(col + 1))
          if first["Raster"].isCountinues(row+1) or second["Raster"].isCountinues(col + 1):
              item = QTableWidgetItem(unicode(self.tr("Not applicable")))
          else:
              corr = depCoef.kappa(mode=None)
              item = QTableWidgetItem(unicode(corr))
          self.tblCorrelation.setItem(row, col, item)

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
      self.grpSampling.show()
    elif modelName == self.tr("Artificial Neural Network"):
      self.modelWidget = neuralnetworkwidget.NeuralNetworkWidget(self)
      self.grpSampling.show()
    elif modelName == self.tr("Weights of Evidence"):
      self.modelWidget = weightofevidencewidget.WeightOfEvidenceWidget(self)
      self.grpSampling.hide()
    elif modelName == self.tr("Multi Criteria Evaluation"):
      self.modelWidget = multicriteriaevaluationwidget.MultiCriteriaEvaluationWidget(self)
      self.grpSampling.hide()

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
        self.btnSelectRiskValidation.setEnabled(True)
      else:
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

  def __selectSamplesOutput(self):
    if not "model" in self.inputs:
      QMessageBox.warning(self,
                          self.tr("Missed model"),
                          self.tr("Nothing to save, samples were not yet generated as the model was not trained. Train the model first.")
                         )
      return
    model = self.inputs["model"]
    if not hasattr(model, 'saveSamples'):
      QMessageBox.warning(self,
                          self.tr("Missed samples"),
                          self.tr("Selected model does't use samples")
                         )
      return

    fileName = utils.saveVectorDialog(self,
                                      self.settings,
                                      self.tr("Save file"),
                                      self.tr("Shape files (*.shp *.SHP *.Shp)")
                                     )
    if fileName.isEmpty():
      return
    model.saveSamples(unicode(fileName))

    if self.chkLoadSamples.isChecked():
      newLayer = QgsVectorLayer(fileName, QFileInfo(fileName).baseName(), "ogr")

      if newLayer.isValid():
        QgsMapLayerRegistry.instance().addMapLayer(newLayer)
      else:
        QMessageBox.warning(self,
                            self.tr("Can't open file"),
                            self.tr("Error loading output shapefile:\n%1")
                            .arg(unicode(fileName))
                           )

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
    elif senderName == "btnSelectMonteCarlo":
      self.leMonteCarloPath.setText(fileName)

  def propagateSimulation(self):
    iterCount = self.spnIterations.value()
    self.simulator.simN(iterCount)

  def __toggleCorrLayers(self, state):
    if state == Qt.Checked:
      self.cmbFirstRaster.setEnabled(False)
      self.cmbSecondRaster.setEnabled(False)
    else:
      self.cmbFirstRaster.setEnabled(True)
      self.cmbSecondRaster.setEnabled(True)

  def __selectValidationMap(self):
    senderName = self.sender().objectName()

    fileName = utils.openRasterDialog(self,
                                      self.settings,
                                      self.tr("Open file"),
                                      self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
                                     )
    if fileName.isEmpty():
      return

    if senderName == "btnSelectReferenceMap":
      self.leReferenceMapPath.setText(fileName)
    elif senderName == "btnSelectSimulatedMap":
      self.leSimulatedMapPath.setText(fileName)

  def validate(self):
    nIter=self.spnValIterCount.value()
    self.eb.getStat(nIter)

  def logMessage(self, message):
    self.txtMessages.append(QString("[%1] %2")
                            .arg(datetime.datetime.now().strftime(u"%a %b %d %Y %H:%M:%S".encode("utf-8")).decode("utf-8"))
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
      self.logMessage(self.tr("Can't load raster %1").arg(filePath))

  def __bandCount(self):
    bands = 0
    for k, v in self.inputs["factors"].iteritems():
      bands +=  v.getBandsCount()
    return bands

  def setProgressRange(self, message, maxValue):
    self.progressBar.setFormat(message)
    self.progressBar.setRange(0, maxValue)

  def showProgress(self):
    self.progressBar.setValue(self.progressBar.value() + 1)

  def restoreProgressState(self):
    self.progressBar.setFormat("%p%")
    self.progressBar.setRange(0, 1)
    self.progressBar.setValue(0)

  def __writeSettings(self):
    # samples and model tab
    self.settings.setValue("ui/samplingMode", self.cmbSamplingMode.itemData(self.cmbSamplingMode.currentIndex()).toInt()[0])
    self.settings.setValue("ui/samplesCount", self.spnSamplesCount.value())
    self.settings.setValue("ui/loadSamples", self.chkLoadSamples.isChecked())

    # simulation tab
    self.settings.setValue("ui/createRiskFunction", self.chkRiskFunction.isChecked())
    self.settings.setValue("ui/createRiskValidation", self.chkRiskValidation.isChecked())
    self.settings.setValue("ui/createMonteCarlo", self.chkMonteCarlo.isChecked())
    self.settings.setValue("ui/monteCarloIterations", self.spnIterations.value())

    # correlation tab
    self.settings.setValue("ui/checkAllRasters", self.chkAllCorr.isChecked())

  def __readSettings(self):
    # samples and model tab
    samplingMode = self.settings.value("ui/samplingMode", 1).toInt()[0]
    self.cmbSamplingMode.setCurrentIndex(self.cmbSamplingMode.findData(samplingMode))
    self.spnSamplesCount.setValue(self.settings.value("ui/samplesCount", 1000).toInt()[0])
    self.chkLoadSamples.setChecked(self.settings.value("ui/loadSamples", False).toBool())

    # simulation tab
    self.chkRiskFunction.setChecked(self.settings.value("ui/createRiskFunction", False).toBool())
    self.chkRiskValidation.setChecked(self.settings.value("ui/createRiskValidation", False).toBool())
    self.chkMonteCarlo.setChecked(self.settings.value("ui/createMonteCarlo", False).toBool())
    self.spnIterations.setValue(self.settings.value("ui/monteCarloIterations", 1).toInt()[0])

    # correlation tab
    self.chkAllCorr.setChecked(self.settings.value("ui/checkAllRasters", False).toBool())

  def applyRasterStyleLabels(self, layer, analyst, tr):
    l = utils.getLayerByName(self.leInitRasterName.text())
    if not l.renderer().type().contains("singlebandpseudocolor"):
      self.logMessage("Init raster should be in PseudoColor mode. Style not applied.")
      return

    r = Raster(unicode(layer.source()))
    stat = r.getBandStat(1)
    minVal = float(stat["min"])
    maxVal = float(stat["max"])
    numberOfEntries = int(maxVal - minVal + 1)

    entryValues = []
    entryColors = []

    colorRamp = QgsStyleV2().defaultStyle().colorRamp("Spectral")
    currentValue = float(minVal)
    intervalDiff = float(maxVal - minVal) / float(numberOfEntries - 1)

    for i in xrange(numberOfEntries):
      entryValues.append(currentValue)
      currentValue += intervalDiff

      entryColors.append(colorRamp.color(float(i) / float(numberOfEntries)))

    rasterShader = QgsRasterShader()
    colorRampShader = QgsColorRampShader()

    cr = l.renderer().shader().rasterShaderFunction().colorRampItemList()

    colorRampItems = []
    for i in xrange(len(entryValues)):
      item = QgsColorRampShader.ColorRampItem()

      item.value = entryValues[i]
      ic, fc = self.analyst.decode(int(entryValues[i]))
      item.label = unicode(self.fl(cr, ic) + u" → " + self.fl(cr, fc))
      item.color = entryColors[i]
      if ic == fc and tr:
        item.color = QColor(255, 255, 255, 0)
      colorRampItems.append(item)

    colorRampShader.setColorRampItemList(colorRampItems)
    colorRampShader.setColorRampType(QgsColorRampShader.INTERPOLATED)
    rasterShader.setRasterShaderFunction(colorRampShader)

    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, rasterShader)
    renderer.setClassificationMin(minVal)
    renderer.setClassificationMax(maxVal)
    renderer.setClassificationMinMaxOrigin(QgsRasterRenderer.minMaxOriginFromName("FullExtent"))

    layer.setRenderer(renderer)
    layer.setCacheImage(None)
    layer.triggerRepaint()
    self.iface.legendInterface().refreshLayerSymbology(layer)
    QgsProject.instance().dirty(True)


  def fl(self, cr, v):
    for i in cr:
      if i.value == v:
        return i.label
    return ""
