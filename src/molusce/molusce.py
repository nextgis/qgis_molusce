# QGIS MOLUSCE Plugin
# Copyright (C) 2026  NextGIS
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <https://www.gnu.org/licenses/>.


import os
from pathlib import Path

from qgis.core import Qgis, QgsApplication
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QFileInfo,
    QLocale,
    QSettings,
    QTranslator,
    QUrl,
)
from qgis.PyQt.QtGui import QDesktopServices, QIcon
from qgis.PyQt.QtWidgets import (
    QAction,
    QMenu,
    QMessageBox,
)

from molusce import resources_rc  # noqa: F401
from molusce.aboutdialog import AboutDialog
from molusce.moluscedialog import MolusceDialog
from molusce.molusceutils import getLocaleShortName
from molusce.processing.provider import MolusceProcessingProvider


class MoluscePlugin:
    def __init__(self, iface):
        self.iface = iface

        try:
            self.QgisVersion = str(Qgis.QGIS_VERSION_INT)
        except Exception:
            self.QgisVersion = str(Qgis.qgisVersion)[0]

        # Processing provider instance (will be registered in initGui)
        self.processing_provider = None

        # For i18n support
        userPluginPath = (
            QFileInfo(QgsApplication.qgisUserDatabaseFilePath()).path()
            + "/python/plugins/molusce"
        )
        systemPluginPath = (
            QgsApplication.prefixPath() + "/python/plugins/molusce"
        )

        overrideLocale = QSettings().value("locale/overrideFlag", False)
        if not overrideLocale:
            localeFullName = QLocale.system().name()
        else:
            localeFullName = QSettings().value("locale/userLocale", "")

        if QFileInfo(userPluginPath).exists():
            translationPath = (
                userPluginPath + "/i18n/molusce_" + localeFullName + ".qm"
            )
        else:
            translationPath = (
                systemPluginPath + "/i18n/molusce_" + localeFullName + ".qm"
            )

        self.localePath = translationPath
        translator = QTranslator()
        if translator.load(self.localePath):
            self.translator = translator
            QCoreApplication.installTranslator(self.translator)

    def initProcessing(self) -> None:
        self.processing_provider = MolusceProcessingProvider()
        QgsApplication.processingRegistry().addProvider(
            self.processing_provider
        )

    def initGui(self):
        if int(self.QgisVersion) < 10900:
            qgisVersion = (
                str(self.QgisVersion[0])
                + "."
                + str(self.QgisVersion[2])
                + "."
                + str(self.QgisVersion[3])
            )
            QMessageBox.warning(
                self.iface.mainWindow(),
                QCoreApplication.translate("MOLUSCE", "Error"),
                QCoreApplication.translate("MOLUSCE", "QGIS %s detected.\n")
                % (qgisVersion)
                + QCoreApplication.translate(
                    "MOLUSCE",
                    "This version of MOLUSCE requires at least QGIS version 3.22.0\nPlugin will not be enabled.",
                ),
            )
            return

        self.initProcessing()

        self.actionRun = QAction(
            QCoreApplication.translate("MOLUSCE", "MOLUSCE"),
            self.iface.mainWindow(),
        )
        self.iface.registerMainWindowAction(self.actionRun, "Shift+M")
        self.actionRun.setIcon(
            QIcon(":/plugins/molusce/icons/molusce_logo.svg")
        )
        self.actionRun.setWhatsThis("Start MOLUSCE plugin")

        self.actionQuickHelp = QAction(
            QCoreApplication.translate("MOLUSCE", "Quick Help..."),
            self.iface.mainWindow(),
        )
        self.actionQuickHelp.setIcon(
            QIcon(":/plugins/molusce/icons/quickhelp.png")
        )
        self.actionQuickHelp.setWhatsThis("Show Quick Help")

        self.actionAbout = QAction(
            QCoreApplication.translate("MOLUSCE", "About MOLUSCE..."),
            self.iface.mainWindow(),
        )
        self.actionAbout.setIcon(
            QgsApplication.getThemeIcon("mActionPropertiesWidget.svg")
        )
        self.actionAbout.setWhatsThis("About MOLUSCE")

        self.__molusce_menu = QMenu(
            QCoreApplication.translate("MOLUSCE", "MOLUSCE")
        )
        self.__molusce_menu.setIcon(
            QIcon(":/plugins/molusce/icons/molusce_logo.svg")
        )

        self.__molusce_menu.addAction(self.actionRun)
        self.__molusce_menu.addAction(self.actionQuickHelp)
        self.__molusce_menu.addAction(self.actionAbout)

        self.iface.rasterMenu().addMenu(self.__molusce_menu)
        self.iface.addRasterToolBarIcon(self.actionRun)

        self.actionRun.triggered.connect(self.run)
        self.actionQuickHelp.triggered.connect(self.showQuickHelp)
        self.actionAbout.triggered.connect(self.about)

        self.actionHelp = QAction(
            QIcon(":/plugins/molusce/icons/molusce_logo.svg"), "MOLUSCE"
        )
        self.actionHelp.triggered.connect(self.about)

        plugin_help_menu = self.iface.pluginHelpMenu()
        assert plugin_help_menu is not None
        plugin_help_menu.addAction(self.actionHelp)

    def unload(self):
        self.iface.unregisterMainWindowAction(self.actionRun)

        self.iface.removeRasterToolBarIcon(self.actionRun)
        self.iface.removePluginRasterMenu(
            QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionRun
        )
        self.iface.removePluginRasterMenu(
            QCoreApplication.translate("MOLUSCE", "MOLUSCE"),
            self.actionQuickHelp,
        )
        self.iface.removePluginRasterMenu(
            QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionAbout
        )
        self.__molusce_menu.deleteLater()

        # --- Unregister MOLUSCE Processing provider ---
        if self.processing_provider is not None:
            QgsApplication.processingRegistry().removeProvider(
                self.processing_provider
            )
            self.processing_provider = None

    def run(self):
        d = MolusceDialog(self.iface)
        d.show()
        d.exec()

    def about(self):
        package_name = str(Path(__file__).parent.name)
        d = AboutDialog(package_name)
        d.exec()

    def showQuickHelp(self):
        dir_name = os.path.dirname(__file__)
        localeShortName = getLocaleShortName()
        guidePath = dir_name + "/doc/" + localeShortName + "/QuickHelp.pdf"
        if os.path.isfile(guidePath):
            QDesktopServices.openUrl(QUrl.fromLocalFile(guidePath))
        else:  # Try to see english documentation
            guidePath = dir_name + "/doc/" + "en" + "/QuickHelp.pdf"
            QDesktopServices.openUrl(QUrl.fromLocalFile(guidePath))
