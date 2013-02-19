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

from ui_moluscedialogbase import Ui_Dialog

import molusceutils as utils

class MolusceDialog(QDialog, Ui_Dialog):
  def __init__(self, iface):
    QDialog.__init__(self)
    self.setupUi(self)

    self.iface = iface
    self.modelWidget = None

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
    self.__logMessage("Started logging")

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
    self.leInitRasterName.setText(layerName)
    rx = QRegExp("(19|21)\d\d")
    pos = rx.indexIn(layerName)
    year = rx.cap()
    self.leInitYear.setText(year)

  def setFinalRaster(self):
    layerName = self.lstLayers.selectedItems()[0].text()
    self.leFinalRasterName.setText(layerName)
    rx = QRegExp("(19|21)\d\d")
    pos = rx.indexIn(layerName)
    year = rx.cap()
    self.leFinalYear.setText(year)

  def addFactor(self):
    layerName = self.lstLayers.selectedItems()[0].text()
    if len(self.lstFactors.findItems(layerName, Qt.MatchExactly)) > 0:
      return

    self.lstFactors.insertItem(self.lstFactors.count() + 1, layerName)

  def removeFactor(self):
    self.lstFactors.takeItem(self.lstFactors.currentRow())

  def removeAllFactors(self):
    self.lstFactors.clear()

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

    self.settings.setValue("ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath())

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
      self.modelWidget = QWidget()
    elif modelName == self.tr("Weights of Evidence"):
      self.modelWidget = QWidget()
    elif modelName == self.tr("Multi Criteria Evaluation"):
      self.modelWidget = QWidget()

    self.widgetStackMethods.addWidget(self.modelWidget)
    self.widgetStackMethods.setCurrentWidget(self.modelWidget)

  def __logMessage(self, message):
    self.txtMessages.append(QString("[%1] %2\n")
                            .arg(datetime.datetime.now().strftime("%a %b %d %Y %H:%M:%S"))
                            .arg(message)
                           )
