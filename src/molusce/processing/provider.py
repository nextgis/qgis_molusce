from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtCore import QCoreApplication
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

    # --- Basic metadata ---

    def id(self):
        # Unique provider id (used internally by QGIS)
        return "molusce"

    def name(self):
        # Short provider name shown in GUI
        return self.tr("MOLUSCE")

    def longName(self):
        # Optional, more descriptive name
        return self.tr("MOLUSCE – Land Use Change Tools")

    def icon(self):
        return QIcon(":/plugins/molusce/icons/molusce_logo.svg")

    # --- Algorithms registration ---

    def loadAlgorithms(self, *args, **kwargs):
        """
        Called by QGIS to let provider add its algorithms.
        """
        self.addAlgorithm(MoluscePrepareRasterAlgorithm())
        self.addAlgorithm(MoluscePrepareVectorAlgorithm())

    # --- Utils ---

    def tr(self, string):
        return QCoreApplication.translate("MolusceProcessingProvider", string)

    def unload(self):
        """
        Called when provider is removed from registry.
        Nothing special required here for basic templates.
        """
        pass