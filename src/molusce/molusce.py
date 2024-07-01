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

import os

# import resources_rc
from qgis.core import *
from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *

from . import aboutdialog, moluscedialog
from .molusceutils import getLocaleShortName


class MoluscePlugin:
  def __init__(self, iface):
    self.iface = iface

    try:
      self.QgisVersion = str(QGis.QGIS_VERSION_INT)
    except:
      self.QgisVersion = str(QGis.qgisVersion)[ 0 ]

    # For i18n support
    userPluginPath = QFileInfo(QgsApplication.qgisUserDbFilePath()).path() + "/python/plugins/molusce"
    systemPluginPath = QgsApplication.prefixPath() + "/python/plugins/molusce"

    overrideLocale = QSettings().value("locale/overrideFlag", False)
    if not overrideLocale:
      localeFullName = QLocale.system().name()
    else:
      localeFullName = QSettings().value("locale/userLocale", "")

    if QFileInfo(userPluginPath).exists():
      translationPath = userPluginPath + "/i18n/molusce_" + localeFullName + ".qm"
    else:
      translationPath = systemPluginPath + "/i18n/molusce_" + localeFullName + ".qm"

    self.localePath = translationPath
    if QFileInfo(self.localePath).exists():
      self.translator = QTranslator()
      self.translator.load(self.localePath)
      QCoreApplication.installTranslator(self.translator)

  def initGui(self):
    if int(self.QgisVersion) < 10900:
      qgisVersion = str(self.QgisVersion[ 0 ]) + "." + str(self.QgisVersion[ 2 ]) + "." + str(self.QgisVersion[ 3 ])
      QMessageBox.warning(self.iface.mainWindow(),
                           QCoreApplication.translate("MOLUSCE", "Error"),
                           QCoreApplication.translate("MOLUSCE", "Quantum GIS %s detected.\n") %s (qgisVersion) +
                           QCoreApplication.translate("MOLUSCE", "This version of MOLUSCE requires at least QGIS version 1.9.0\nPlugin will not be enabled."))
      return None

    self.actionRun = QAction(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.iface.mainWindow())
    self.iface.registerMainWindowAction(self.actionRun, "Shift+M")
    self.actionRun.setIcon(QIcon(":/icons/molusce.png"))
    self.actionRun.setWhatsThis("Start MOLUSCE plugin")
    self.actionQuickHelp = QAction(QCoreApplication.translate("MOLUSCE", "Quick Help..."), self.iface.mainWindow())
    self.actionQuickHelp.setIcon(QIcon(":/icons/quickhelp.png"))
    self.actionQuickHelp.setWhatsThis("Show Quick Help")
    self.actionAbout = QAction(QCoreApplication.translate("MOLUSCE", "About MOLUSCE..."), self.iface.mainWindow())
    self.actionAbout.setIcon(QIcon(":/icons/about.png"))
    self.actionAbout.setWhatsThis("About MOLUSCE")

    self.iface.addPluginToRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionRun)
    self.iface.addPluginToRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionQuickHelp)
    self.iface.addPluginToRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionAbout)
    self.iface.addRasterToolBarIcon(self.actionRun)

    self.actionRun.triggered.connect(self.run)
    self.actionQuickHelp.triggered.connect(self.showQuickHelp)
    self.actionAbout.triggered.connect(self.about)

  def unload(self):
    self.iface.unregisterMainWindowAction(self.actionRun)

    self.iface.removeRasterToolBarIcon(self.actionRun)
    self.iface.removePluginRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionRun)
    self.iface.removePluginRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionQuickHelp)
    self.iface.removePluginRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionAbout)

  def run(self):
    d = moluscedialog.MolusceDialog(self.iface)
    d.show()
    d.exec_()

  def about(self):
    d = aboutdialog.AboutDialog()
    d.exec_()

  def showQuickHelp(self):
    dir_name =  os.path.dirname(__file__)
    localeShortName = getLocaleShortName()
    guidePath = dir_name+ "/doc/" + localeShortName + "/QuickHelp.pdf"
    if os.path.isfile(guidePath):
        QDesktopServices.openUrl(QUrl.fromLocalFile(guidePath))
    else: # Try to see english documentation
      guidePath = dir_name+ "/doc/" + "en" + "/QuickHelp.pdf"
      QDesktopServices.openUrl(QUrl.fromLocalFile(guidePath))