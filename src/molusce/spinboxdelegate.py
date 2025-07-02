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

from typing import Optional

from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import *


class SpinBoxDelegate(QItemDelegate):
    """
    A delegate that provides a QSpinBox editor for editing
    integer values in a QTableView or QTreeView.

    This delegate sets a spin box with a defined
    minimum and maximum range for cell editing.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        minRange: int = 1,
        maxRange: int = 9,
    ) -> None:
        """
        Initialize the SpinBoxDelegate.

        This constructor sets up a delegate that uses a QSpinBox for editing
        integer values in item views like QTableView or QTreeView. The spin box
        will enforce the specified minimum and maximum values.

        :param parent: Optional parent widget for the delegate.
        :type parent: Optional[QWidget]
        :param minRange: Minimum value allowed in the spin box (inclusive). Defaults to 1.
        :type minRange: int
        :param maxRange: Maximum value allowed in the spin box (inclusive). Defaults to 9.
        :type maxRange: int
        """
        super().__init__(parent)
        self.minRange = minRange
        self.maxRange = maxRange

    def createEditor(
        self,
        parent: QWidget,
        _options: QStyleOptionViewItem,
        _index: QModelIndex,
    ) -> QSpinBox:
        """
        Creates the spin box editor widget.

        :param parent: The parent widget for the editor.
        :type parent: QWidget
        :param options: Style options for the item.
        :type options: QStyleOptionViewItem
        :param index: The index of the item to be edited.
        :type index: QModelIndex

        :return: Configured QSpinBox editor.
        :rtype: QSpinBox
        """
        editor = QSpinBox(parent)
        editor.setRange(self.minRange, self.maxRange)
        return editor

    def setEditorData(self, editor: QSpinBox, index: QModelIndex) -> None:
        """
        Sets the value from the model into the spin box editor.

        :param editor: The spin box editor.
        :type editor: QSpinBox
        :param index: The index of the item in the model.
        :type index: QModelIndex
        """
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        try:
            editor.setValue(value)
        except TypeError:  # Check None-value, ""-value, etc.
            value = self.minRange

    def setModelData(
        self, editor: QSpinBox, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        """
        Updates the model with the value from the spin box editor.

        :param editor: The spin box editor.
        :type editor: QSpinBox
        :param model: The data model to update.
        :type model: QAbstractItemModel
        :param index: The index of the item in the model.
        :type index: QModelIndex
        """
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)
