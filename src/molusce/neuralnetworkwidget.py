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

from matplotlib import rcParams
from matplotlib.figure import Figure

from molusce.algorithms.models.sampler.sampler import SamplerError

from . import molusceutils as utils
from .algorithms.models.mlp.manager import MlpManager, MlpManagerError
from .ui.ui_neuralnetworkwidgetbase import Ui_NeuralNetworkWidgetBase


class NeuralNetworkWidget(QWidget, Ui_NeuralNetworkWidgetBase):
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
        rcParams["font.serif"] = "Verdana, Arial, Liberation Serif"
        rcParams["font.sans-serif"] = "Tahoma, Arial, Liberation Sans"
        rcParams["font.cursive"] = "Courier New, Arial, Liberation Sans"
        rcParams["font.fantasy"] = "Comic Sans MS, Arial, Liberation Sans"
        rcParams["font.monospace"] = "Courier New, Liberation Mono"

        self.btnTrainNetwork.clicked.connect(self.trainNetwork)

        self.manageGui()

    def manageGui(self):
        self.spnNeigbourhood.setValue(
            int(self.settings.value("ui/ANN/neighborhood", 1))
        )
        self.spnLearnRate.setValue(
            float(self.settings.value("ui/ANN/learningRate", 0.1))
        )
        self.spnMaxIterations.setValue(
            int(self.settings.value("ui/ANN/maxIterations", 1000))
        )
        self.leTopology.setText(self.settings.value("ui/ANN/topology", "10"))
        self.spnMomentum.setValue(
            float(self.settings.value("ui/ANN/momentum", 0.05))
        )
        self.btnStop.setEnabled(False)

    @pyqtSlot()
    def trainNetwork(self) -> None:
        """
        Validates the input data and initializes training of the Artificial Neural Network (ANN) model.

        This function performs the following steps:
        - Validates the presence of input rasters, factor rasters, and a change map.
        - Validates the ANN topology string.
        - Saves current UI ANN configuration settings to persistent storage.
        - Initializes and configures the MLP model (neural network).
        - Sets training data and hyperparameters such as learning rate, momentum, number of iterations.
        - Sets up the training/validation graph UI.
        - Moves the model to a worker thread and connects appropriate signals for training lifecycle.
        - Starts training in a separate thread.

        Emits log messages and error warnings when required.
        Disables the "Train" button and enables "Stop" during training.

        Raises:
            MlpManagerError: If MLP model setup fails (e.g., due to invalid output layer configuration).
        """
        if not utils.checkInputRasters(self.inputs):
            QMessageBox.warning(
                self.plugin,
                self.tr("Missed input data"),
                self.tr(
                    "Initial or final raster is not set. Please specify input data and try again"
                ),
            )
            return

        if not utils.checkFactors(self.inputs):
            QMessageBox.warning(
                self.plugin,
                self.tr("Missed input data"),
                self.tr(
                    "Factors rasters is not set. Please specify them and try again"
                ),
            )
            return

        if not utils.checkChangeMap(self.inputs):
            QMessageBox.warning(
                self.plugin,
                self.tr("Missed input data"),
                self.tr(
                    "Change map raster is not set. Please create it try again"
                ),
            )
            return

        if self.leTopology.text() == "":
            QMessageBox.warning(
                self.plugin,
                self.tr("Wrong network topology"),
                self.tr(
                    "Network topology is undefined. Please define it and try again"
                ),
            )
            return

        self.settings.setValue(
            "ui/ANN/neighborhood", self.spnNeigbourhood.value()
        )
        self.settings.setValue(
            "ui/ANN/learningRate", self.spnLearnRate.value()
        )
        self.settings.setValue(
            "ui/ANN/maxIterations", self.spnMaxIterations.value()
        )
        self.settings.setValue("ui/ANN/topology", self.leTopology.text())
        self.settings.setValue("ui/ANN/momentum", self.spnMomentum.value())

        self.btnStop.setEnabled(True)
        self.btnTrainNetwork.setEnabled(False)

        self.plugin.logMessage(self.tr("Init ANN model"))

        model = MlpManager(ns=self.spnNeigbourhood.value())
        self.inputs["model"] = model
        try:
            model.createMlp(
                self.inputs["initial"],
                list(self.inputs["factors"].values()),
                self.inputs["changeMap"],
                [int(n) for n in self.leTopology.text().split(" ")],
            )
            self.plugin.logMessage(self.tr("Set training data"))
            model.setTrainingData(
                self.inputs["initial"],
                list(self.inputs["factors"].values()),
                self.inputs["changeMap"],
                mode=self.inputs["samplingMode"],
                samples=self.plugin.spnSamplesCount.value(),
            )
        except SamplerError as error:
            QMessageBox.warning(
                self,
                self.tr("Sampling error"),
                str(error),
            )
            return
        except MlpManagerError as error:
            QMessageBox.warning(
                self,
                self.tr("Output layer error"),
                str(error),
            )
            return

        model.setEpochs(self.spnMaxIterations.value())
        model.setValPercent(20)
        model.setLRate(self.spnLearnRate.value())
        model.setMomentum(self.spnMomentum.value())
        model.setContinueTrain()

        self.axes.cla()
        self.axes.grid(True)
        self.dataTrain = []
        self.dataVal = []
        self.plotTrain = self.axes.plot(
            self.dataTrain, linewidth=1, color="green", marker="o"
        )[0]
        self.plotVal = self.axes.plot(
            self.dataVal,
            linewidth=1,
            color="red",
        )[0]
        leg = self.axes.legend((self.tr("Train"), self.tr("Validation")))
        for t in leg.get_texts():
            t.set_fontsize("small")
        model.moveToThread(self.plugin.workThread)

        self.plugin.workThread.started.connect(model.startTrain)
        self.btnStop.clicked.connect(model.stopTrain)
        model.updateGraphValues.connect(self.__updateGraphValues)
        model.updateGraph.connect(self.__update_graph)
        model.updateDeltaRMS.connect(self.__updateRMS)
        model.updateMinValErr.connect(self.__updateValidationError)
        model.updateKappa.connect(self.__updateKappa)
        model.processInterrupted.connect(self.__trainInterrupted)
        model.rangeChanged.connect(self.plugin.setProgressRange)
        model.updateProgress.connect(self.plugin.showProgress)
        model.error_occurred.connect(self.show_model_error)
        model.errorReport.connect(self.plugin.logErrorReport)
        model.processFinished.connect(self.__trainFinished)
        model.processFinished.connect(self.plugin.workThread.quit)

        self.plugin.logMessage(self.tr("Start training ANN model"))
        self.plugin.workThread.start()

    @pyqtSlot()
    def __trainFinished(self) -> None:
        """
        Slot that handles the completion of the ANN model training process.

        This method performs the following actions:
        - Disconnects all signals connected to the model training process.
        - Restores the progress bar state.
        - Updates the UI by disabling the "Stop" button and enabling the "Train Network" button.
        - Logs a message indicating that training has finished.
        - Triggers the graph update to reflect the new model state.
        """
        model = self.inputs["model"]
        self.plugin.workThread.started.disconnect(model.startTrain)
        model.rangeChanged.disconnect(self.plugin.setProgressRange)
        model.updateProgress.disconnect(self.plugin.showProgress)
        model.error_occurred.disconnect(self.show_model_error)
        model.errorReport.disconnect(self.plugin.logErrorReport)
        self.plugin.restoreProgressState()
        self.btnStop.setEnabled(False)
        self.btnTrainNetwork.setEnabled(True)
        self.plugin.logMessage(self.tr("ANN model is trained"))
        self.__update_graph()

    def __trainInterrupted(self):
        self.plugin.workThread.quit()
        self.btnStop.setEnabled(False)
        self.btnTrainNetwork.setEnabled(True)
        self.plugin.logMessage(self.tr("ANN model training interrupted"))

    @pyqtSlot(float)
    def __updateRMS(self, dRMS):
        self.leDeltaRMS.setText("%6.5f" % (dRMS))

    @pyqtSlot(float)
    def __updateValidationError(self, error):
        self.leValidationError.setText("%6.5f" % (error))

    @pyqtSlot(float)
    def __updateKappa(self, kappa):
        self.leKappa.setText("%6.5f" % (kappa))

    @pyqtSlot(float, float)
    def __updateGraphValues(self, errTrain: float, errVal: float):
        self.dataTrain.append(errTrain)
        self.dataVal.append(errVal)

    @pyqtSlot()
    def __update_graph(self):
        ymin = min([min(self.dataTrain), min(self.dataVal)])
        ymax = max([max(self.dataTrain), max(self.dataVal)])

        self.axes.set_xbound(lower=0, upper=len(self.dataVal))
        self.axes.set_ybound(lower=ymin, upper=ymax)

        self.plotTrain.set_xdata(numpy.arange(len(self.dataTrain)))
        self.plotTrain.set_ydata(numpy.array(self.dataTrain))

        self.plotVal.set_xdata(numpy.arange(len(self.dataVal)))
        self.plotVal.set_ydata(numpy.array(self.dataVal))

        self.canvas.draw()

    @pyqtSlot(str, str)
    def show_model_error(self, title: str, message: str) -> None:
        """
        Display a warning message box with the model error details.

        :param title: The title of the warning message box.
        :param message: The error message to display.
        """
        QMessageBox.warning(self, title, message)
