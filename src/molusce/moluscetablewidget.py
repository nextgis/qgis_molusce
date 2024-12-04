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

from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import QAction, QMenu, QTableWidget


class MolusceTableWidget(QTableWidget):
    def __init__(self, parent=None):
        QTableWidget.__init__(self, parent)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if (
            e.modifiers() == Qt.ControlModifier
            or e.modifiers() == Qt.MetaModifier
        ) and e.key() == Qt.Key_C:
            self.copy_selected_cells()
        else:
            QTableWidget.keyPressEvent(self, e)

    def show_context_menu(self, pos: QPoint) -> None:
        context_menu = QMenu(self)

        copy_selected_cells_action = QAction(
            self.tr("Copy selected cells"), self
        )
        copy_entire_table_action = QAction(self.tr("Copy entire table"), self)

        context_menu.addAction(copy_selected_cells_action)
        context_menu.addAction(copy_entire_table_action)

        copy_selected_cells_action.triggered.connect(self.copy_selected_cells)
        copy_entire_table_action.triggered.connect(self.copy_full_table)

        context_menu.exec_(self.viewport().mapToGlobal(pos))

    def copy_selected_cells(self) -> None:
        data = ""
        selected_cells = sorted(self.selectedIndexes())
        max_column = max(cell.column() for cell in selected_cells)

        # table header
        data += "\t"
        for cell in selected_cells:
            if self.horizontalHeaderItem(cell.column()).text() not in data:
                data += self.horizontalHeaderItem(cell.column()).text() + "\t"
        data += "\n"

        # table contents
        for cell in selected_cells:
            if self.verticalHeaderItem(cell.row()).text() not in data:
                data += self.verticalHeaderItem(cell.row()).text() + "\t"
            if cell.column() == 0:
                data += (
                    self.item(cell.row(), 0).background().color().name() + "\t"
                )
            else:
                data += self.item(cell.row(), cell.column()).text()
                if cell.column() == max_column:
                    data += "\n"
                else:
                    data += "\t"

        if data != "":
            clipBoard = QGuiApplication.clipboard()
            clipBoard.setText(data)

    def copy_full_table(self) -> None:
        data = ""

        # table header
        data += "\t"
        for i in range(self.columnCount()):
            data += self.horizontalHeaderItem(i).text() + "\t"
        data += "\n"

        # table contents
        for r in range(self.rowCount()):
            data += self.verticalHeaderItem(r).text() + "\t"
            for c in range(self.columnCount()):
                if c == 0:
                    data += self.item(r, 0).background().color().name() + "\t"
                else:
                    data += self.item(r, c).text() + "\t"
            data += "\n"

        if data != "":
            clipBoard = QGuiApplication.clipboard()
            clipBoard.setText(data)
