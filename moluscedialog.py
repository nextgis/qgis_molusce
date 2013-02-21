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

import logisticregressionwidget
import neuralnetworkwidget
import weightofevidencewidget
import multicriteriaevaluationwidget

from ui.ui_moluscedialogbase import Ui_Dialog

import molusceutils as utils

from algorithms.dataprovider import Raster
from algorithms.models.area_analysis.manager import AreaAnalyst

class MolusceDialog(QDialog, Ui_Dialog):
  def __init__(self, iface):
    QDialog.__init__(self)
    self.setupUi(self)

    self.iface = iface
    self.modelWidget = None

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
    #              }
    # }
    # Layer ids are necessary to handle factors changes (e.g. adding new or removing
    # existing factor)
    self.inputs = dict()

    self.settings = QSettings("NextGIS", "MOLUSCE")

    # connect signals and slots
    self.btnSetInitialRaster.clicked.connect(self.setInitialRaster)
    self.btnSetFinalRaster.clicked.connect(self.setFinalRaster)
    self.btnAddFactor.clicked.connect(self.addFactor)
    self.btnRemoveFactor.clicked.connect(self.removeFactor)
    self.btnRemoveAllFactors.clicked.connect(self.removeAllFactors)

    self.btnUpdateStatistics.clicked.connect(self.updateStatisticsTable)
    self.btnCreateChangeMap.clicked.connect(self.createChangeMap)

    self.cmbMethod.currentIndexChanged.connect(self.__modelChanged)

    self.manageGui()
    self.__logMessage(self.tr("Start logging"))

  def manageGui(self):
    self.restoreGeometry(self.settings.value("/ui/geometry").toByteArray())

    self.tabWidget.setCurrentIndex(0)

    self.__populateLayers()
    self.__populateMethods()

    # TODO: restore settings

  def closeEvent(self, e):
    self.settings.setValue("/ui/geometry", QVariant(self.saveGeometry()))

    # TODO: save settings

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

    self.__logMessage(self.tr("Added factor layer %1").arg(layerName))

  def removeFactor(self):
    layerId = unicode(self.lstFactors.currentItem().data(Qt.UserRole).toString())
    layerName = self.lstFactors.currentItem().text()
    self.lstFactors.takeItem(self.lstFactors.currentRow())

    del self.inputs["factors"][layerId]

    self.__logMessage(self.tr("Removed factor layer %1").arg(layerName))

  def removeAllFactors(self):
    self.lstFactors.clear()

    del self.inputs["factors"]

    self.__logMessage(self.tr("Factors list cleared"))

  def updateStatisticsTable(self):
    pass

  def createChangeMap(self):
    lastDir = self.settings.value("ui/lastRasterDir", ".").toString()
    fileName = QFileDialog.getSaveFileName(self,
                                           self.tr("Save change map"),
                                           lastDir,
                                           self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
                                          )

    if fileName.isEmpty():
      return

    if not fileName.toLower().contains(QRegExp("\.tif{1,2}")):
      fileName += ".tif"

    #~ rasterPath = unicode(utils.getLayerById(self.initRasterId).source())
    #~ initRaster = Raster(rasterPath)
    #~ initRaster.setMask([0, 255])      # Let 0 and 255 values are No-data values

    #~ rasterPath = unicode(utils.getLayerById(self.finalRasterId).source())
    #~ finalRaster = Raster(rasterPath)

    if ("initial" in self.inputs) and ("final" in self.inputs):
      analyst = AreaAnalyst(self.inputs["initial"], self.inputs["final"])
      changeMapRaster = analyst.makeChangeMap()
      changeMapRaster.save(unicode(fileName))
      self.__logMessage(self.tr("Change map image saved to: %1").arg(fileName))

      self.settings.setValue("ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath())
    else:
      self.__logMessage(self.tr("Can't create change map. Initial or final land use map is not set"))

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

  def __populateMethods(self):
    self.cmbMethod.addItems([
                             self.tr("Logistic Regression"),
                             self.tr("Artificial Neural Network"),
                             self.tr("Weights of Evidence"),
                             self.tr("Multi Criteria Evaluation")
                           ])

  def __modelChanged(self):
    if self.modelWidget is not None:
      self.widgetStackMethods.removeWidget(self.modelWidget)

      self.modelWidget = None
      del self.modelWidget

    modelName = self.cmbMethod.currentText()

    if modelName == self.tr("Logistic Regression"):
      self.modelWidget = logisticregressionwidget.LogisticRegressionWidget()
    elif modelName == self.tr("Artificial Neural Network"):
      self.modelWidget = neuralnetworkwidget.NeuralNetworkWidget()
    elif modelName == self.tr("Weights of Evidence"):
      self.modelWidget = weightofevidencewidget.WeightOfEvidenceWidget()
    elif modelName == self.tr("Multi Criteria Evaluation"):
      self.modelWidget = multicriteriaevaluationwidget.MultiCriteriaEvaluationWidget()

    self.widgetStackMethods.addWidget(self.modelWidget)
    self.widgetStackMethods.setCurrentWidget(self.modelWidget)

  def __logMessage(self, message):
    self.txtMessages.append(QString("[%1] %2")
                            .arg(datetime.datetime.now().strftime("%a %b %d %Y %H:%M:%S"))
                            .arg(message)
                           )
