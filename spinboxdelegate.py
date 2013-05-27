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

class SpinBoxDelegate(QItemDelegate):
  def __init__(self, parent=None, minRange=1, maxRange=9):
    QItemDelegate.__init__(self, parent)
    self.minRange = minRange
    self.maxRange = maxRange

  def createEditor(self, parent, options, index):
    editor = QSpinBox(parent)
    editor.setRange(self.minRange, self.maxRange)
    return editor

  def setEditorData(self, editor, index):
    value = index.model().data(index, Qt.EditRole).toInt()[0]
    editor.setValue(value)

  def setModelData(self, editor, model, index):
    model.setData(index, editor.value(), Qt.EditRole)
