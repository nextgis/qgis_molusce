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

from qgis.core import *

import moluscedialog
import aboutdialog

import resources_rc

class MoluscePlugin:
  def __init__(self, iface):
    self.iface = iface

    try:
      self.QgisVersion = unicode(QGis.QGIS_VERSION_INT)
    except:
      self.QgisVersion = unicode(QGis.qgisVersion)[ 0 ]

    # For i18n support
    userPluginPath = QFileInfo(QgsApplication.qgisUserDbFilePath()).path() + "/python/plugins/molusce"
    systemPluginPath = QgsApplication.prefixPath() + "/python/plugins/molusce"

    overrideLocale = QSettings().value("locale/overrideFlag", QVariant(False)).toBool()
    if not overrideLocale:
      localeFullName = QLocale.system().name()
    else:
      localeFullName = QSettings().value("locale/userLocale", QVariant("")).toString()

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
                           QCoreApplication.translate("MOLUSCE", "Quantum GIS %1 detected.\n").arg(qgisVersion) +
                           QCoreApplication.translate("MOLUSCE", "This version of MOLUSCE requires at least QGIS version 1.9.0\nPlugin will not be enabled."))
      return None

    self.actionRun = QAction(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.iface.mainWindow())
    self.iface.registerMainWindowAction(self.actionRun, "Shift+M")
    self.actionRun.setIcon(QIcon(":/icons/molusce.png"))
    self.actionRun.setWhatsThis("Start MOLUSCE plugin")
    self.actionAbout = QAction(QCoreApplication.translate("MOLUSCE", "About MOLUSCE..."), self.iface.mainWindow())
    self.actionAbout.setIcon(QIcon(":/icons/about.png"))
    self.actionAbout.setWhatsThis("About MOLUSCE")

    self.iface.addPluginToRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionRun)
    self.iface.addPluginToRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionAbout)
    self.iface.addRasterToolBarIcon(self.actionRun)

    self.actionRun.triggered.connect(self.run)
    self.actionAbout.triggered.connect(self.about)

  def unload(self):
    self.iface.unregisterMainWindowAction(self.actionRun)

    self.iface.removeRasterToolBarIcon(self.actionRun)
    self.iface.removePluginRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionRun)
    self.iface.removePluginRasterMenu(QCoreApplication.translate("MOLUSCE", "MOLUSCE"), self.actionAbout)

  def run(self):
    d = moluscedialog.MolusceDialog(self.iface)
    d.show()
    d.exec_()

  def about(self):
    d = aboutdialog.AboutDialog()
    d.exec_()
