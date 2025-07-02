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

from pathlib import Path
from typing import Any, Dict, List, Optional

from qgis.core import (
    Qgis,
    QgsMapLayer,
    QgsProject,
    QgsRasterLayer,
    QgsReadWriteContext,
    QgsSettings,
)
from qgis.PyQt.QtCore import QFileInfo, QLocale, QObject, QSettings
from qgis.PyQt.QtWidgets import QFileDialog, QWidget
from qgis.PyQt.QtXml import QDomDocument, QDomImplementation


def getLocaleShortName() -> str:
    overrideLocale = QSettings().value("locale/overrideFlag", False)
    if not overrideLocale:
        localeFullName = QLocale.system().name()
    else:
        localeFullName = QSettings().value("locale/userLocale", "")

    localeShortName = localeFullName[0:2]
    return localeShortName


def getRasterLayers() -> Dict[str, str]:
    layerMap = QgsProject.instance().mapLayers()
    layers: Dict[str, str] = dict()
    for layer in layerMap.values():
        if (
            isinstance(layer, QgsRasterLayer)
            and layer.providerType() == "gdal"
            and layer.id() not in layers
        ):
            layers[layer.id()] = str(layer.name())
    return layers


def getLayerMask(
    layer: Optional[QgsMapLayer],
) -> Optional[Dict[int, List[float]]]:
    if not isinstance(layer, QgsRasterLayer):
        return None

    provider = layer.dataProvider()
    maskVals = dict()
    bCount = layer.bandCount()
    for i in range(bCount):
        mask = [
            rasterRange.min()
            for rasterRange in provider.userNoDataValues(i + 1)
        ]

        # Provider nodata value ALWAYS used during raster reading
        # (see algorithms.dataprovider._read)
        # if provider.useSrcNoDataValue(i+1):
        #  mask.append(provider.srcNoDataValue(i+1))
        maskVals[i + 1] = mask
    return maskVals


def getLayerMaskById(layerId: str) -> Optional[Dict[int, List[float]]]:
    layer = getLayerById(layerId)
    maskVals = getLayerMask(layer)
    return maskVals


def getLayerMaskByName(layerName: str) -> Optional[Dict[int, List[float]]]:
    layer = getLayerByName(layerName)
    maskVals = getLayerMask(layer)
    return maskVals


def getLayerMaskBySource(layerSource: str) -> Optional[Dict[int, List[float]]]:
    layer = getLayerBySource(layerSource)
    maskVals = getLayerMask(layer)
    return maskVals


def getLayerById(layerId: str) -> Optional[QgsMapLayer]:
    layerMap = QgsProject.instance().mapLayers()
    for _name, layer in layerMap.items():
        if layer.id() == layerId:
            if layer.isValid():
                return layer
            return None
    return None


def getLayerByName(layerName: str) -> Optional[QgsMapLayer]:
    layerMap = QgsProject.instance().mapLayers()
    for _name, layer in layerMap.items():
        if layer.name() == layerName:
            if layer.isValid():
                return layer
            return None
    return None


def getLayerBySource(layerSource: str) -> Optional[QgsMapLayer]:
    layerMap = QgsProject.instance().mapLayers()
    for _name, layer in layerMap.items():
        if layer.source() == layerSource:
            if layer.isValid():
                return layer
            return None
    return None


def getLayerGroup(relations, layerId):
    group = None

    for item in relations:
        group = str(item[0])
        for lid in item[1]:
            if str(lid) == str(layerId):
                return group

    return group


def saveDialog(parent, settings, title, fileFilter, fileExt):
    lastDir = settings.value("ui/lastRasterDir", ".")
    fileName = QFileDialog.getSaveFileName(parent, title, lastDir, fileFilter)[
        0
    ]

    if fileName == "":
        return ""

    if not fileName.lower().endswith(fileExt):
        fileName += fileExt

    settings.setValue(
        "ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath()
    )

    return fileName


def saveRasterDialog(parent, settings, title, fileFilter):
    fileName = saveDialog(parent, settings, title, fileFilter, ".tif")
    return fileName


def saveVectorDialog(parent, settings, title, fileFilter):
    fileName = saveDialog(parent, settings, title, fileFilter, ".shp")
    return fileName


def openRasterDialog(parent, settings, title, fileFilter):
    lastDir = settings.value("ui/lastRasterDir", ".")
    fileName = QFileDialog.getOpenFileName(parent, title, lastDir, fileFilter)[
        0
    ]

    if fileName == "":
        return ""

    settings.setValue(
        "ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath()
    )

    return fileName


def openDirectoryDialog(
    parent: QWidget, settings: QgsSettings, title: str
) -> str:
    """
    Open a dialog for selecting a directory.

    The dialog remembers the last used directory and updates the settings
    with the new path if a directory is selected.

    :param parent: The parent widget for the dialog.
    :type parent: QWidget
    :param settings: QgsSettings instance for storing the last directory.
    :type settings: QgsSettings
    :param title: The title of the dialog window.
    :type title: str

    :returns: The selected directory path as a string, or an empty string if canceled.
    :rtype: str
    """
    lastDir = settings.value("ui/lastRasterDir", ".")
    destDir = QFileDialog.getExistingDirectory(
        parent, title, lastDir, QFileDialog.Option.ShowDirsOnly
    )
    if destDir == "":
        return ""
    return destDir


def checkInputRasters(userData):
    return bool("initial" in userData and "final" in userData)


def checkFactors(userData, sim=False):
    if not sim:
        return "factors" in userData
    return "factors_sim" in userData


def checkChangeMap(userData):
    return "changeMap" in userData


def copySymbology(src, dst):
    di = QDomImplementation()
    dt = di.createDocumentType("qgis", "http://mrcc.com/qgis.dtd", "SYSTEM")
    doc = QDomDocument(dt)
    root = doc.createElement("qgis")
    root.setAttribute("version", str(Qgis.QGIS_VERSION))
    doc.appendChild(root)
    errMsg = ""
    if not src.writeSymbology(
        root,
        doc,
        errMsg,
        QgsReadWriteContext(),
        QgsMapLayer.StyleCategory.AllStyleCategories,
    ):
        return False

    return dst.readSymbology(
        root,
        errMsg,
        QgsReadWriteContext(),
        QgsMapLayer.StyleCategory.AllStyleCategories,
    )


def is_file_used_by_project(destination_path: Path) -> bool:
    layers = [
        layer
        for layer in QgsProject().instance().mapLayers().values()
        if layer.providerType() in ("ogr", "gdal")
    ]
    layers_paths = [Path(layer.source().split("|")[0]) for layer in layers]

    return any(path == destination_path for path in layers_paths)


class PickleQObjectMixin:
    """
    A mixin class to enable pickling and unpickling of QObject-based classes.
    """

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        QObject.__init__(self)  # type: ignore reportArgumentType
