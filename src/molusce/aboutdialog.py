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

import configparser  # noqa: I001
import os

from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox

from .molusceutils import getLocaleShortName
from .ui.ui_aboutdialogbase import Ui_AboutDialogBase

from . import resources_rc  # noqa: F401


class AboutDialog(QDialog, Ui_AboutDialogBase):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

        self.btnHelp = self.buttonBox.button(QDialogButtonBox.Help)

        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), "metadata.txt"))
        version = cfg.get("general", "version")

        self.lblLogo.setPixmap(QPixmap(":/icons/molusce.png"))
        self.lblVersion.setText(self.tr("Version: %s") % (version))
        doc = QTextDocument()
        doc.setHtml(self.getAboutText())
        self.textBrowser.setDocument(doc)

        self.buttonBox.helpRequested.connect(self.openHelp)

    def reject(self):
        QDialog.reject(self)

    def openHelp(self):
        localeShortName = getLocaleShortName()
        if localeShortName in ["ru", "uk"]:
            QDesktopServices.openUrl(
                QUrl("http://hub.qgis.org/projects/molusce/wiki")
            )
        else:
            QDesktopServices.openUrl(
                QUrl("http://hub.qgis.org/projects/molusce/wiki")
            )

    def getAboutText(self):
        return self.tr("""<p>Modules for Land Use Change Simulations.</p>
<p>Plugin provides a set of algorithms for land use change simulations such as
ANN, LR, WoE, MCE. There is also validation using kappa statistics.</p>
<p>Developed by <a href="http://www.asiaairsurvey.com/">Asia Air Survey</a> and <a href="http://nextgis.org">NextGIS</a>.</p>
<p><strong>Homepage</strong>: <a href="http://hub.qgis.org/projects/molusce">http://hub.qgis.org/projects/molusce</a></p>
<p>Please report bugs at <a href="http://hub.qgis.org/projects/molusce/issues">bugtracker</a></p>
""")
