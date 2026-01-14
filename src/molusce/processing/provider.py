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

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from molusce.processing.prepare_raster import (
    MoluscePrepareRasterAlgorithm,
)
from molusce.processing.prepare_vector import (
    MoluscePrepareVectorAlgorithm,
)


class MolusceProcessingProvider(QgsProcessingProvider):
    """
    Processing provider exposing MOLUSCE related algorithms.
    """

    def __init__(self):
        super().__init__()

    def id(self):
        """Returns the unique provider id, used for identifying the provider."""
        return "molusce"

    def name(self):
        """Returns the provider name, which is used to describe the provider within the GUI."""
        return self.tr("MOLUSCE")

    def longName(self):
        """Returns the longer version of the provider name, which can include extra details."""
        return self.tr("MOLUSCE – Land Use Change Tools")

    def icon(self):
        """Returns an icon for the provider."""
        return QIcon(":/plugins/molusce/icons/molusce_logo.svg")

    def loadAlgorithms(self):
        """
        Called by QGIS to let provider add its algorithms.
        """
        self.addAlgorithm(MoluscePrepareRasterAlgorithm())
        self.addAlgorithm(MoluscePrepareVectorAlgorithm())
