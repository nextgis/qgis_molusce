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

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from qgis.core import *

from algorithms.models.lr.lr import LR

from ui.ui_logisticregressionwidgetbase import Ui_Widget

import molusceutils as utils

class LogisticRegressionWidget(QWidget, Ui_Widget):
  def __init__(self, plugin, parent=None):
    QWidget.__init__(self, parent)
    self.setupUi(self)

    self.plugin = plugin
    self.inputs = plugin.inputs

    self.model = None

    self.settings = QSettings("NextGIS", "MOLUSCE")

    self.btnFitModel.clicked.connect(self.fitModel)

    self.manageGui()

  def manageGui(self):
    self.spnNeighbourhood.setValue(self.settings.value("ui/LR/neighborhood", 1).toInt()[0])

  def fitModel(self):
    if not utils.checkInputRasters(self.inputs):
      QMessageBox.warning(self.plugin,
                          self.tr("Missed input data"),
                          self.tr("Initial or final raster is not set. Please specify input data and try again")
                         )
      return

    if not utils.checkFactors(self.inputs):
      QMessageBox.warning(self.plugin,
                          self.tr("Missed input data"),
                          self.tr("Factors rasters is not set. Please specify them and try again")
                         )
      return

    self.settings.setValue("ui/LR/neighborhood", self.spnNeighbourhood.value())

    self.model = LR(ns=self.spnNeighbourhood.value())

    self.model.setTrainingData(self.inputs["initial"],
                               self.inputs["factors"].values(),
                               self.inputs["final"],
                               mode=self.inputs["samplingMode"],
                               samples=self.plugin.spnSamplesCount.value()
                              )

    self.model.train()

    # populate table
    self.showCoefficients()

    self.inputs["model"] = self.model

  def showCoefficients(self):
    if self.model is None:
      QMessageBox.warning(self.plugin,
                          self.tr("Model is not initialised"),
                          self.tr("To get coefficients you need to train model first")
                         )
      return

    fm = self.model.getIntercept()
    coef = self.model.getCoef()

    colCount = len(fm)
    rowCount = len(coef[0]) + 1
    self.tblCoefficients.clear()
    self.tblCoefficients.setColumnCount(colCount)
    self.tblCoefficients.setRowCount(rowCount)

    labels = []
    for i in range(rowCount):
      labels.append(u"b%s" % (i,))
    self.tblCoefficients.setVerticalHeaderLabels(labels)
    labels = []
    for i in range(colCount):
      labels.append(u"Class %s" % (i+1,))
    self.tblCoefficients.setHorizontalHeaderLabels(labels)

    for i in xrange(len(fm)):
      item = QTableWidgetItem(unicode(fm[i]))
      self.tblCoefficients.setItem(0, i, item)
      for j in xrange(len(coef[i])):
        item = QTableWidgetItem(unicode(coef[i][j]))
        self.tblCoefficients.setItem(j + 1, i, item)

    self.tblCoefficients.resizeRowsToContents()
    self.tblCoefficients.resizeColumnsToContents()
