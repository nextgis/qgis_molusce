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

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams

from algorithms.models.mlp.manager import MlpManager

from ui.ui_neuralnetworkwidgetbase import Ui_Widget

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

    self.chkCreateReport.toggled.connect(self.__toggleLineEdit)
    self.chkSaveSamples.toggled.connect(self.__toggleLineEdit)

    self.btnSelectReport.clicked.connect(self.__selectFile)
    self.btnSelectSamples.clicked.connect(self.__selectFile)

    self.btnTrainNetwork.clicked.connect(self.trainNetwork)

    self.manageGui()

  def manageGui(self):
    self.spnNeigbourhood.setValue(self.settings.value("ui/ANN/neighborhood", 1).toInt()[0])
    self.spnLearnRate.setValue(self.settings.value("ui/ANN/learningRate", 0.1).toFloat()[0])
    self.spnMaxIterations.setValue(self.settings.value("ui/ANN/maxIterations", 1000).toInt()[0])
    self.spnRMS.setValue(self.settings.value("ui/ANN/rms", 0.001).toFloat()[0])
    self.leTopology.setText(self.settings.value("ui/ANN/topology", "10").toString())

    self.chkCreateReport.setChecked(self.settings.value("ui/ANN/createReport", False).toBool())
    self.chkSaveSamples.setChecked(self.settings.value("ui/ANN/saveSamples", False).toBool())

  def trainNetwork(self):
    model = MlpManager(ns=self.spnNeigbourhood.value())
    model.createMlp(self.inputs["initial"],
                    self.inputs["factors"].values(),
                    self.inputs["changeMap"],
                    self.leTopology.text().split(" ")
                   )

    self.plugin.__logMessage(self.tr("ANN training started"))
    model.train(self.spnMaxIterations.value(),
                valPercent=20
               )
    self.plugin.__logMessage(self.tr("ANN training completed"))

    #self.inputs["model"] = model

  def __selectFile(self):
    senderName = self.sender().objectName()

    # TODO: implement dialog for necessary data type
    fileName = utils.saveRasterDialog(self,
                                      self.settings,
                                      self.tr("Save file"),
                                      self.tr("GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
                                     )
    if fileName.isEmpty():
      return

    if senderName == "btnSelectReport":
      self.leReportPath.setText(fileName)
    elif senderName == "btnSelectSamples":
      self.leSamplesPath.setText(fileName)

  def __toggleLineEdit(self, checked):
    senderName = self.sender().objectName()
    if senderName == "chkCreateReport":
      if checked:
        self.leReportPath.setEnabled(True)
        self.btnSelectReport.setEnabled(True)
      else:
        self.leReportPath.setEnabled(False)
        self.btnSelectReport.setEnabled(False)
    elif senderName == "chkSaveSamples":
      if checked:
        self.leSamplesPath.setEnabled(True)
        self.btnSelectSamples.setEnabled(True)
      else:
        self.leSamplesPath.setEnabled(False)
        self.btnSelectSamples.setEnabled(False)
