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

from qgis.core import *
from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import *

from . import molusceutils as utils
from .algorithms.models.lr.lr import LR
from .ui.ui_logisticregressionwidgetbase import Ui_Widget


class LogisticRegressionWidget(QWidget, Ui_Widget):
  def __init__(self, plugin, parent=None):
    QWidget.__init__(self, parent)
    self.setupUi(self)

    self.plugin = plugin
    self.inputs = plugin.inputs

    self.settings = QSettings("NextGIS", "MOLUSCE")

    self.btnFitModel.clicked.connect(self.startFitModel)

    self.manageGui()

  def manageGui(self):
    self.tabLRResults.setCurrentIndex(0)
    self.spnNeighbourhood.setValue(int(self.settings.value("ui/LR/neighborhood", 1)))

  def startFitModel(self):
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

    if not utils.checkChangeMap(self.inputs):
      QMessageBox.warning(self.plugin,
                          self.tr("Missed input data"),
                          self.tr("Change map raster is not set. Please create it try again")
                         )
      return

    self.settings.setValue("ui/LR/neighborhood", self.spnNeighbourhood.value())

    self.plugin.logMessage(self.tr("Init LR model"))

    model = LR(ns=self.spnNeighbourhood.value())
    self.inputs["model"] = model
    model.setMaxIter(self.spnMaxIterations.value())

    model.setState(self.inputs["initial"])
    model.setFactors(list(self.inputs["factors"].values()))
    model.setOutput(self.inputs["changeMap"])
    model.setMode(self.inputs["samplingMode"])
    model.setSamples(self.plugin.spnSamplesCount.value())

    self.plugin.logMessage(self.tr("Set training data"))
    model.moveToThread(self.plugin.workThread)
    self.plugin.workThread.started.connect(model.startTrain)
    self.plugin.setProgressRange("Train LR model", 0)
    model.finished.connect(self.__trainFinished)
    model.errorReport.connect(self.plugin.logErrorReport)
    model.finished.connect(self.plugin.workThread.quit)
    self.plugin.workThread.start()

  def __trainFinished(self):
    model = self.inputs["model"]
    self.plugin.workThread.started.disconnect(model.startTrain)
    self.plugin.restoreProgressState()

    # Transition labels for the coef. tables
    analyst = self.plugin.analyst
    self.labels = list(model.labelCodes)
    self.labels = ["%s → %s" % analyst.decode(int(c)) for c in self.labels]

    # populate table
    self.showCoefficients()
    self.showStdDeviations()
    self.showPValues()

    self.plugin.logMessage(self.tr("LR model trained"))

  def showCoefficients(self):
    model = self.inputs["model"]
    if model is None:
      QMessageBox.warning(self.plugin,
                          self.tr("Model is not initialised"),
                          self.tr("To get coefficients you need to train model first")
                         )
      return

    fm = model.getIntercept()
    coef = model.getCoef()
    accuracy = model.getPseudoR()

    colCount = len(fm)
    rowCount = len(coef[0]) + 1
    self.tblCoefficients.clear()
    self.tblCoefficients.setColumnCount(colCount)
    self.tblCoefficients.setRowCount(rowCount)

    labels = [f"β{i}" for i in range(rowCount)]
    self.tblCoefficients.setVerticalHeaderLabels(labels)
    self.tblCoefficients.setHorizontalHeaderLabels(self.labels)

    for i in range(len(fm)):
      item = QTableWidgetItem(str(fm[i]))
      self.tblCoefficients.setItem(0, i, item)
      for j in range(len(coef[i])):
        item = QTableWidgetItem(str(coef[i][j]))
        self.tblCoefficients.setItem(j + 1, i, item)

    self.tblCoefficients.resizeRowsToContents()
    self.tblCoefficients.resizeColumnsToContents()

    self.lePseudoR.setText("%6.5f" % (accuracy))

  def showStdDeviations(self):
    model = self.inputs["model"]
    if model is None:
      QMessageBox.warning(self.plugin,
                          self.tr("Model is not initialised"),
                          self.tr("To get standard deviations you need to train model first")
                         )
      return

    stdErrW = model.getStdErrWeights()
    stdErrI = model.getStdErrIntercept()
    colCount = len(stdErrI)
    rowCount = len(stdErrW[0]) + 1

    self.tblStdDev.clear()
    self.tblStdDev.setColumnCount(colCount)
    self.tblStdDev.setRowCount(rowCount)

    labels = [f"β{i}" for i in range(rowCount)]
    self.tblStdDev.setVerticalHeaderLabels(labels)
    self.tblStdDev.setHorizontalHeaderLabels(self.labels)

    for i in range(len(stdErrI)):
      item = QTableWidgetItem("%6.5f" % (stdErrI[i]))
      self.tblStdDev.setItem(0, i, item)
      for j in range(len(stdErrW[i])):
        item = QTableWidgetItem("%6.5f" % (stdErrW[i][j]))
        self.tblStdDev.setItem(j + 1, i, item)

    self.tblStdDev.resizeRowsToContents()
    self.tblStdDev.resizeColumnsToContents()

  def showPValues(self):
    model = self.inputs["model"]
    def significance(p):
      if p <= 0.01:
        return "**"
      elif p<= 0.05:
        return "*"
      else:
        return "-"

    if model is None:
      QMessageBox.warning(self.plugin,
                          self.tr("Model is not initialised"),
                          self.tr("To get p-values you need to train model first")
                         )
      return

    fm = model.get_PvalIntercept()
    coef = model.get_PvalWeights()

    colCount = len(fm)
    rowCount = len(coef[0]) + 1
    self.tblPValues.clear()
    self.tblPValues.setColumnCount(colCount)
    self.tblPValues.setRowCount(rowCount)

    labels = [f"β{i}" for i in range(rowCount)]
    self.tblPValues.setVerticalHeaderLabels(labels)
    self.tblPValues.setHorizontalHeaderLabels(self.labels)

    for i in range(len(fm)):
      s = "%f %s" % (fm[i], significance(fm[i]))
      item = QTableWidgetItem(str(s))
      self.tblPValues.setItem(0, i, item)
      for j in range(len(coef[i])):
        s = "%f %s" % (coef[i][j], significance(coef[i][j]))
        item = QTableWidgetItem(str(s))
        self.tblPValues.setItem(j + 1, i, item)

    self.tblPValues.resizeRowsToContents()
    self.tblPValues.resizeColumnsToContents()
