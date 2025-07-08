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

from typing import TYPE_CHECKING, Dict, List, Optional

from qgis.PyQt.QtCore import QSettings, Qt, pyqtSlot
from qgis.PyQt.QtWidgets import (
    QMessageBox,
    QTableWidgetItem,
    QWidget,
)

from molusce import molusceutils as utils
from molusce.algorithms import dataprovider
from molusce.algorithms.models.woe.manager import WoeManager, WoeManagerError
from molusce.spinboxdelegate import SpinBoxDelegate
from molusce.ui.ui_weightofevidencewidgetbase import (
    Ui_WeightOfEvidenceWidgetBase,
)

if TYPE_CHECKING:
    from molusce.moluscedialog import MolusceDialog


class WeightOfEvidenceWidget(QWidget, Ui_WeightOfEvidenceWidgetBase):
    """
    Widget for configuring and training the Weights of Evidence (WoE) model.
    """

    def __init__(
        self, plugin: "MolusceDialog", parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize the Multi Criteria Evaluation Widget.

        :param plugin: The instance of the MolusceDialog class
                       that provides access to inputs and settings.
        :param parent: The parent widget, defaults to None.
        """
        super().__init__(parent)
        self.setupUi(self)

        self.plugin = plugin
        self.inputs = plugin.inputs

        self.settings = QSettings("NextGIS", "MOLUSCE")

        self.btnTrainModel.clicked.connect(self.trainModel)

        self.btnResetBins.clicked.connect(self.__resetBins)

        self.manageGui()

    def manageGui(self) -> None:
        """
        Configure and initialize the GUI elements for the WoE widget.

        This method checks the presence of required input data, prepares the
        reclassification table for all factor rasters, and sets up delegates
        for editing bin counts.
        """
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

        self.tblReclass.clearContents()
        self.delegate = SpinBoxDelegate(
            self.tblReclass.model(),
            min_range=2,
            max_range=dataprovider.MAX_CATEGORIES,
        )

        row = 0
        for layer_id, factor_raster in self.inputs["factors"].items():
            factor_raster.denormalize()

            for band in range(1, factor_raster.getBandsCount() + 1):
                if not factor_raster.isCountinues(band):
                    continue

                self.tblReclass.insertRow(row)

                band_name = (
                    f"{utils.getLayerById(layer_id).name()} (band {band})"
                    if factor_raster.getBandsCount() > 1
                    else utils.getLayerById(layer_id).name()
                )

                stat = factor_raster.getBandStat(band)
                values = [
                    band_name,
                    str(stat["min"]),
                    str(stat["max"]),
                    "2",
                    "",
                ]
                for column, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    if column < 3:
                        item.setFlags(
                            item.flags() ^ Qt.ItemFlag.ItemIsEditable
                        )
                    self.tblReclass.setItem(row, column, item)

                row += 1

        self.tblReclass.setItemDelegateForColumn(3, self.delegate)
        self.tblReclass.resizeRowsToContents()
        self.tblReclass.resizeColumnsToContents()

    @pyqtSlot()
    def trainModel(self) -> None:
        """
        Starts the training of the WoE (Weight of Evidence) model. The method checks if
        the necessary input rasters and factor rasters are provided, initializes the model,
        and starts the training process in a separate thread.

        If any of the input conditions are not met, an appropriate warning message is displayed.
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

        myBins = self.__getBins()

        self.plugin.logMessage(self.tr("Init WoE model"))
        try:
            model = WoeManager(
                list(self.inputs["factors"].values()),
                self.plugin.analyst,
                bins=myBins,
            )
        except AttributeError:
            QMessageBox.warning(
                self.plugin,
                self.tr("Error"),
                self.tr("Model training failed"),
            )
            return
        except WoeManagerError as err:
            QMessageBox.warning(
                self.plugin, self.tr("Initialization error"), err.msg
            )
            return

        self.inputs["model"] = model

        if not model.checkBins():
            QMessageBox.warning(
                self.plugin,
                self.tr("Wrong ranges"),
                self.tr(
                    "Ranges are not correctly specified. Please specify them and try again (use space as separator)"
                ),
            )
            return

        model.moveToThread(self.plugin.workThread)
        self.plugin.workThread.started.connect(model.train)
        model.updateProgress.connect(self.plugin.showProgress)
        model.rangeChanged.connect(self.plugin.setProgressRange)
        model.error_occurred.connect(self.show_model_error)
        model.errorReport.connect(self.plugin.logErrorReport)
        model.processFinished.connect(self.__trainFinished)
        model.processFinished.connect(self.plugin.workThread.quit)

        self.plugin.workThread.start()

    @pyqtSlot()
    def __trainFinished(self) -> None:
        """
        Slot that is triggered when the WoE model training process is finished.
        It performs the following actions:
        - Disconnects signals related to model training.
        - Restores the progress state.
        - Logs a message indicating that the model has been trained.
        - Displays the model's weights in the text area.
        """
        model = self.inputs["model"]
        self.plugin.workThread.started.disconnect(model.train)
        model.updateProgress.disconnect(self.plugin.showProgress)
        model.rangeChanged.connect(self.plugin.setProgressRange)
        model.processFinished.disconnect(self.__trainFinished)
        model.processFinished.disconnect(self.plugin.workThread.quit)
        model.error_occurred.disconnect(self.show_model_error)
        model.errorReport.disconnect(self.plugin.logErrorReport)
        self.plugin.restoreProgressState()
        self.plugin.logMessage(self.tr("WoE model is trained"))
        self.pteWeightsInform.appendPlainText(str(model.weightsToText()))

    def __getBins(self) -> Optional[Dict[int, List[List[int]]]]:
        """
        Extract bin boundaries for all continuous factor bands from the reclassification table.

        :return: Mapping from factor index to a list of bin thresholds per band.
                Returns None if any range is malformed or missing.
        :rtype: Dict[int, List[Optional[List[int]]]]
        """
        bins: Dict[int, List[List[int]]] = {}

        for factor_index, (layer_id, raster) in enumerate(
            self.inputs["factors"].items()
        ):
            band_bins: List[List[int]] = []

            for band in range(raster.getBandsCount()):
                if not raster.isCountinues(band + 1):
                    band_bins.append([])
                    continue

                name = (
                    f"{utils.getLayerById(layer_id).name()} (band {band + 1})"
                    if raster.getBandsCount() > 1
                    else utils.getLayerById(layer_id).name()
                )

                items = self.tblReclass.findItems(
                    name, Qt.MatchFlag.MatchExactly
                )
                row = self.tblReclass.indexFromItem(items[0]).row()
                item_text = self.tblReclass.item(row, 4).text()

                try:
                    bin_values = [
                        int(value) for value in item_text.strip().split()
                    ]
                    band_bins.append(bin_values)
                except ValueError:
                    QMessageBox.warning(
                        self.plugin,
                        self.tr("Wrong ranges"),
                        self.tr(
                            "Ranges are not correctly specified. Please specify them and try again (use space as separator)"
                        ),
                    )
                    return None

            bins[factor_index] = band_bins

        return bins

    def __resetBins(self):
        for row in range(self.tblReclass.rowCount()):
            try:
                rangeMin = float(self.tblReclass.item(row, 1).text())
                rangeMax = float(self.tblReclass.item(row, 2).text())
                intervals = int(float(self.tblReclass.item(row, 3).text()))
            except ValueError:
                continue
            delta = (rangeMax - rangeMin) / intervals
            item = [
                str(int(rangeMin + delta * (i))) for i in range(1, intervals)
            ]
            item = " ".join(item)
            item = QTableWidgetItem(item)
            self.tblReclass.setItem(row, 4, item)

    @pyqtSlot(str, str)
    def show_model_error(self, title: str, message: str) -> None:
        """
        Display a warning message box with the model error details.

        :param title: The title of the warning message box.
        :param message: The error message to display.
        """
        QMessageBox.warning(self, title, message)
