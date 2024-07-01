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
from qgis.PyQt.QtWidgets import QTableWidget


class MolusceTableWidget(QTableWidget):
  def __init__(self, parent=None):
    QTableWidget.__init__(self, parent)

  def keyPressEvent(self, e):
    if (e.modifiers() == Qt.ControlModifier or e.modifiers() == Qt.MetaModifier) and e.key() == Qt.Key_C:
      data = ""

      # table header
      data += "\t"
      for i in range(0, self.columnCount()):
        data += self.horizontalHeaderItem(i).text() + "\t"
      data += "\n"

      # table contents
      for r in range(0, self.rowCount()):
        data += self.verticalHeaderItem(r).text() + "\t"
        for c in range(0, self.columnCount()):
          data += self.item(r, c).text() + "\t"
        data += "\n"

      if data != "":
        clipBoard = QApplication.clipboard()
        clipBoard.setText(data)
    else:
      QTableWidget.keyPressEvent(self, e)
