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

from qgis.PyQt.QtCore import QPoint, Qt
from qgis.PyQt.QtGui import QGuiApplication, QKeyEvent
from qgis.PyQt.QtWidgets import (
    QAction,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class MolusceTableWidget(QTableWidget):
    """
    Custom QTableWidget for MOLUSCE with extended copy and context menu features.

    This widget allows copying selected cells or the entire table to the clipboard
    and provides a context menu for these actions.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the MolusceTableWidget.

        :param parent: The parent widget, defaults to None.
        :type parent: Optional[QWidget]
        """
        super().__init__(parent)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handle key press events for the table widget.

        Copies selected cells to the clipboard if Ctrl+C or Cmd+C is pressed.

        :param event: The key event.
        :type event: QKeyEvent
        """
        if (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            or event.modifiers() == Qt.KeyboardModifier.MetaModifier
        ) and event.key() == Qt.Key.Key_C:
            self.copy_selected_cells()
        else:
            QTableWidget.keyPressEvent(self, event)

    def show_context_menu(self, position: QPoint) -> None:
        """
        Display context menu for copy actions.

        :param position: Position where the menu should appear.
        :type position: QPoint
        """
        context_menu = QMenu(self)

        copy_selected_cells_action = QAction(
            self.tr("Copy selected cells"), self
        )
        copy_entire_table_action = QAction(self.tr("Copy entire table"), self)

        context_menu.addAction(copy_selected_cells_action)
        context_menu.addAction(copy_entire_table_action)

        copy_selected_cells_action.triggered.connect(self.copy_selected_cells)
        copy_entire_table_action.triggered.connect(self.copy_full_table)

        context_menu.exec(self.viewport().mapToGlobal(position))

    def copy_selected_cells(self) -> None:
        """
        Copy selected cells from the table to the clipboard.
        """
        if len(self.selectedIndexes()) == 0:
            return

        data = ""
        selected_cells = sorted(self.selectedIndexes())
        max_column = max(cell.column() for cell in selected_cells)

        for cell in selected_cells:
            row = cell.row()
            column = cell.column()
            data += self.__item_to_text(self.item(row, column), max_column)

        if data != "":
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(data)

    def copy_full_table(self) -> None:
        """
        Copy the entire table content to the clipboard, including headers.
        """
        data = ""
        max_column = self.columnCount() - 1

        # table header
        data += "\t"
        for column_index in range(self.columnCount()):
            data += self.horizontalHeaderItem(column_index).text() + "\t"
        data += "\n"

        # table contents
        for row in range(self.rowCount()):
            header = self.verticalHeaderItem(row)
            header_text = header.text() if header is not None else str(row)
            data += f"{header_text}\t"
            for column in range(self.columnCount()):
                data += self.__item_to_text(self.item(row, column), max_column)

        if data != "":
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(data)

    def __item_to_text(
        self, item: Optional[QTableWidgetItem], max_column: int
    ) -> str:
        """
        Convert a table item to its text representation.

        :param item: Table item to convert.
        :type item: Optional[QTableWidgetItem]
        :param max_column: Last column index of the row.
        :type max_column: int

        :return: String for clipboard output.
        :rtype: str
        """
        if item is None:
            return "\t"

        if item.column() == 0 and item.text() == "":
            return item.background().color().name() + "\t"

        if item.column() == max_column:
            return item.text() + "\n"

        return item.text() + "\t"
