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

import numpy

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams

from algorithms.models.mlp.manager import MlpManager

from ui.ui_neuralnetworkwidgetbase import Ui_Widget

import molusceutils as utils

class NeuralNetworkWidget(QWidget, Ui_Widget):
  def __init__(self, plugin, parent=None):
    QWidget.__init__(self, parent)
    self.setupUi(self)

    self.plugin = plugin
    self.inputs = plugin.inputs
    self.settings = QSettings("NextGIS", "MOLUSCE")

    # init plot for learning curve
    self.figure = Figure()
    self.axes = self.figure.add_subplot(111)
    self.axes.grid(True)
    self.figure.suptitle(self.tr("Neural Network learning curve"))
    self.canvas = FigureCanvas(self.figure)
    self.mpltoolbar = NavigationToolbar(self.canvas, None)
    lstActions = self.mpltoolbar.actions()
    self.mpltoolbar.removeAction(lstActions[7])
    self.layoutPlot.addWidget(self.canvas)
    self.layoutPlot.addWidget(self.mpltoolbar)

    # and configure matplotlib params
    rcParams['font.serif'] = "Verdana, Arial, Liberation Serif"
    rcParams['font.sans-serif'] = "Tahoma, Arial, Liberation Sans"
    rcParams['font.cursive'] = "Courier New, Arial, Liberation Sans"
    rcParams['font.fantasy'] = "Comic Sans MS, Arial, Liberation Sans"
    rcParams['font.monospace'] = "Courier New, Liberation Mono"

    self.btnTrainNetwork.clicked.connect(self.trainNetwork)

    self.manageGui()

  def manageGui(self):
    self.spnNeigbourhood.setValue(self.settings.value("ui/ANN/neighborhood", 1).toInt()[0])
    self.spnLearnRate.setValue(self.settings.value("ui/ANN/learningRate", 0.1).toFloat()[0])
    self.spnMaxIterations.setValue(self.settings.value("ui/ANN/maxIterations", 1000).toInt()[0])
    self.leTopology.setText(self.settings.value("ui/ANN/topology", "10").toString())
    self.spnMomentum.setValue(self.settings.value("ui/ANN/momentum", 0.05).toFloat()[0])

  def trainNetwork(self):
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

    if self.leTopology.text().isEmpty():
      QMessageBox.warning(self.plugin,
                          self.tr("Wriong network topology"),
                          self.tr("Network topology is undefined. Please define it and try again")
                         )
      return

    self.settings.setValue("ui/ANN/neighborhood", self.spnNeigbourhood.value())
    self.settings.setValue("ui/ANN/learningRate", self.spnLearnRate.value())
    self.settings.setValue("ui/ANN/maxIterations", self.spnMaxIterations.value())
    self.settings.setValue("ui/ANN/topology", self.leTopology.text())
    self.settings.setValue("ui/ANN/momentum", self.spnMomentum.value())

    self.plugin.logMessage(self.tr("Init ANN model"))

    self.model = MlpManager(ns=self.spnNeigbourhood.value())
    self.model.createMlp(self.inputs["initial"],
                         self.inputs["factors"].values(),
                         self.inputs["changeMap"],
                         [int(n) for n in self.leTopology.text().split(" ")]
                        )

    self.plugin.logMessage(self.tr("Set training data"))
    self.model.setTrainingData(self.inputs["initial"],
                               self.inputs["factors"].values(),
                               self.inputs["changeMap"],
                               mode=self.inputs["samplingMode"],
                               samples=self.plugin.spnSamplesCount.value()
                              )

    self.model.setEpochs(self.spnMaxIterations.value())
    self.model.setValPercent(20)
    self.model.setLRate(self.spnLearnRate.value())
    self.model.setMomentum(self.spnMomentum.value())
    self.model.setContinueTrain()

    self.dataTrain = []
    self.dataVal = []
    self.plotTrain = self.axes.plot(self.dataTrain,
                                    linewidth=1,
                                    color="green",  marker='o'
                                   )[0]
    self.plotVal = self.axes.plot(self.dataVal,
                                  linewidth=1,
                                  color="red",
                                 )[0]
    leg = self.axes.legend(('Train', 'Validation'), 'upper right', shadow=False)
    for t in leg.get_texts():
        t.set_fontsize('small')
    self.model.moveToThread(self.plugin.workThread)

    self.plugin.workThread.started.connect(self.model.startTrain)
    self.model.updateGraph.connect(self.__updateGraph)
    self.model.updateDeltaRMS.connect(self.__updateRMS)
    self.model.updateMinValErr.connect(self.__updateValidationError)
    self.model.updateKappa.connect(self.__updateKappa)
    self.model.processFinished.connect(self.__trainFinished)
    self.model.processFinished.connect(self.plugin.workThread.quit)

    self.plugin.logMessage(self.tr("Start trainig ANN model"))
    self.plugin.workThread.start()

    self.inputs["model"] = self.model

  def __trainFinished(self):
    self.plugin.workThread.started.disconnect(self.model.startTrain)
    self.plugin.logMessage(self.tr("ANN model trained"))

  def __updateRMS(self, dRMS):
    self.leDeltaRMS.setText(QString.number(dRMS))

  def __updateValidationError(self, error):
    self.leValidationError.setText(QString.number(error))

  def __updateKappa(self, kappa):
    self.leKappa.setText(QString.number(kappa))

  def __updateGraph(self, errTrain, errVal):
    self.dataTrain.append(errTrain)
    self.dataVal.append(errVal)

    ymin = min([min(self.dataTrain), min(self.dataVal)])
    ymax = max([max(self.dataTrain), max(self.dataVal)])

    self.axes.set_xbound(lower=0, upper=len(self.dataVal))
    self.axes.set_ybound(lower=ymin, upper=ymax)

    self.plotTrain.set_xdata(numpy.arange(len(self.dataTrain)))
    self.plotTrain.set_ydata(numpy.array(self.dataTrain))

    self.plotVal.set_xdata(numpy.arange(len(self.dataVal)))
    self.plotVal.set_ydata(numpy.array(self.dataVal))

    self.canvas.draw()
