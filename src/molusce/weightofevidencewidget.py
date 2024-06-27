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

from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *

from qgis.core import *

from .algorithms.models.area_analysis.manager import AreaAnalyst
from .algorithms.models.woe.manager import WoeManager, WoeManagerError
from .algorithms import dataprovider

from . import spinboxdelegate

from .ui.ui_weightofevidencewidgetbase import Ui_Widget

from . import molusceutils as utils

class WeightOfEvidenceWidget(QWidget, Ui_Widget):
  def __init__(self, plugin, parent=None):
    QWidget.__init__(self, parent)
    self.setupUi(self)

    self.plugin = plugin
    self.inputs = plugin.inputs

    self.settings = QSettings("NextGIS", "MOLUSCE")

    self.btnTrainModel.clicked.connect(self.trainModel)

    self.btnResetBins.clicked.connect(self.__resetBins)

    self.manageGui()

  def manageGui(self):
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

    self.tblReclass.clearContents()
    self.delegate = spinboxdelegate.SpinBoxDelegate(self.tblReclass.model(), minRange=2, maxRange=dataprovider.MAX_CATEGORIES)

    row = 0
    for k, v in self.inputs["factors"].items():
      v.denormalize()   # Denormalize the factor's bands if they are normalized
      for b in range(1, v.getBandsCount()+1):
        if v.isCountinues(b):
          self.tblReclass.insertRow(row)
          if v.getBandsCount()>1:
            name = "%s (band %s)" % (utils.getLayerById(k).name(), str(b))
          else:
            name = "%s" % (utils.getLayerById(k).name(), )
          stat = v.getBandStat(b)
          for n, item_data in enumerate([name, ("%f" % (stat["min"], )), ("%f" % (stat["max"])), "" , "" ]):
            item = QTableWidgetItem(item_data)
            if n < 3:
              item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.tblReclass.setItem(row, n, item)
          row += 1
    rowCount = row

    self.tblReclass.setItemDelegateForColumn(3, self.delegate)
    for row in range(rowCount):
      # Set 2 bins as default value
      self.tblReclass.setItem(row, 3, QTableWidgetItem('2'))

    self.tblReclass.resizeRowsToContents()
    self.tblReclass.resizeColumnsToContents()

  def trainModel(self):
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

    myBins = self.__getBins()

    self.plugin.logMessage(self.tr("Init WoE model"))
    try:
      model = WoeManager(list(self.inputs["factors"].values()), self.plugin.analyst, bins=myBins)
    except WoeManagerError as err:
      QMessageBox.warning(self.plugin,
                          self.tr("Initialization error"),
                          err.msg
                         )
      return

    self.inputs["model"] = model

    if not model.checkBins():
      QMessageBox.warning(self.plugin,
                          self.tr("Wrong ranges"),
                          self.tr("Ranges are not correctly specified. Please specify them and try again (use space as separator)")
                         )
      return

    model.moveToThread(self.plugin.workThread)
    self.plugin.workThread.started.connect(model.train)
    model.updateProgress.connect(self.plugin.showProgress)
    model.rangeChanged.connect(self.plugin.setProgressRange)
    model.errorReport.connect(self.plugin.logErrorReport)
    model.processFinished.connect(self.__trainFinished)
    model.processFinished.connect(self.plugin.workThread.quit)

    self.plugin.workThread.start()

  def __trainFinished(self):
    model = self.inputs["model"]
    self.plugin.workThread.started.disconnect(model.train)
    model.updateProgress.disconnect(self.plugin.showProgress)
    model.rangeChanged.connect(self.plugin.setProgressRange)
    model.processFinished.disconnect(self.__trainFinished)
    model.processFinished.disconnect(self.plugin.workThread.quit)
    model.errorReport.disconnect(self.plugin.logErrorReport)
    self.plugin.restoreProgressState()
    self.plugin.logMessage(self.tr("WoE model trained"))
    self.pteWeightsInform.appendPlainText( str(model.weightsToText()) )

  def __getBins(self):
    bins = dict()
    n = 0
    for k, v in self.inputs["factors"].items():
      lst = []
      for b in range(v.getBandsCount()):
        lst.append(None)
        if v.isCountinues(b+1):
          if v.getBandsCount()>1:
            name = "%s (band %s)" % (utils.getLayerById(k).name(), str(b+1))
          else:
            name = "%s" % (utils.getLayerById(k).name(), )
          items = self.tblReclass.findItems(name, Qt.MatchExactly)
          idx = self.tblReclass.indexFromItem(items[0])
          reclassList = self.tblReclass.item(idx.row(), 4).text()
          try:
            lst[b] = [int(j) for j in reclassList.split(" ")]
          except ValueError:
            QMessageBox.warning(self.plugin,
                          self.tr("Wrong ranges"),
                          self.tr("Ranges are not correctly specified. Please specify them and try again (use space as separator)")
                         )
            return {}
      bins[n] = lst
      n += 1

    return bins

  def __resetBins(self):
    for row in range(self.tblReclass.rowCount()):
      try:
        rangeMin = float(self.tblReclass.item(row, 1).text())
        rangeMax = float(self.tblReclass.item(row, 2).text())
        intervals = int(float(self.tblReclass.item(row, 3).text()))
      except ValueError:
        continue
      delta = (rangeMax - rangeMin)/intervals
      item = [str( int(rangeMin + delta*(i)) )  for i in range(1,intervals)]
      item = " ".join(item)
      item = QTableWidgetItem(item)
      self.tblReclass.setItem(row, 4, item)
