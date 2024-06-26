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
from PyQt4.QtXml import *

from qgis.core import *

def getLocaleShortName():
    overrideLocale = QSettings().value("locale/overrideFlag", False)
    if not overrideLocale:
      localeFullName = QLocale.system().name()
    else:
      localeFullName = QSettings().value("locale/userLocale", "")

    localeShortName = localeFullName[ 0:2 ]
    return localeShortName

def getRasterLayers():
  layerMap = QgsMapLayerRegistry.instance().mapLayers()
  layers = dict()
  for name, layer in layerMap.items():
    if layer.type() == QgsMapLayer.RasterLayer and layer.providerType() == "gdal":
      if layer.id() not in list(layers.keys()):
        layers[layer.id()] = str(layer.name())
  return layers

def getLayerMask(layer):
  if layer == None:
    return None
  else:
      provider = layer.dataProvider()
      maskVals = dict()
      bCount = layer.bandCount()
      for i in range(bCount):
        mask = []
        for rasterRange in provider.userNoDataValues(i+1):
          mask.append(rasterRange.min())

        # Provider nodata value ALWAYS used during raster reading
        # (see algorithms.dataprovider._read)
        #if provider.useSrcNoDataValue(i+1):
        #  mask.append(provider.srcNoDataValue(i+1))
        maskVals[i+1] = mask
      return maskVals

def getLayerMaskById(layerId):
  layer = getLayerById(layerId)
  maskVals = getLayerMask(layer)
  return maskVals

def getLayerMaskByName(layerName):
  layer = getLayerByName(layerName)
  maskVals = getLayerMask(layer)
  return maskVals

def getLayerMaskBySource(layerSource):
  layer = getLayerBySource(layerSource)
  maskVals = getLayerMask(layer)
  return maskVals

def getLayerById(layerId):
  layerMap = QgsMapLayerRegistry.instance().mapLayers()
  for name, layer in layerMap.items():
    if layer.id() == layerId:
      if layer.isValid():
        return layer
      else:
        return None

def getLayerByName(layerName):
  layerMap = QgsMapLayerRegistry.instance().mapLayers()
  for name, layer in layerMap.items():
    if layer.name() == layerName:
      if layer.isValid():
        return layer
      else:
        return None

def getLayerBySource(layerSource):
  layerMap = QgsMapLayerRegistry.instance().mapLayers()
  for name, layer in layerMap.items():
    if layer.source() == layerSource:
      if layer.isValid():
        return layer
      else:
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
  fileName = QFileDialog.getSaveFileName(parent,
                                         title,
                                         lastDir,
                                         fileFilter
                                        )

  if fileName == "":
    return ""

  if not fileName.lower().endswith(fileExt):
    fileName += fileExt

  settings.setValue("ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath())

  return fileName

def saveRasterDialog(parent, settings, title, fileFilter):
  fileName = saveDialog(parent, settings, title, fileFilter, ".tif")
  return fileName

def saveVectorDialog(parent, settings, title, fileFilter):
  fileName = saveDialog(parent, settings, title, fileFilter, ".shp")
  return fileName

def openRasterDialog(parent, settings, title, fileFilter):
  lastDir = settings.value("ui/lastRasterDir", ".")
  fileName = QFileDialog.getOpenFileName(parent,
                                         title,
                                         lastDir,
                                         fileFilter
                                        )

  if fileName == "":
    return ""

  settings.setValue("ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath())

  return fileName

def openDirectoryDialog(parent, settings, title):
  lastDir = settings.value("ui/lastRasterDir", ".")
  destDir = QFileDialog.getExistingDirectory(parent,
                                            title,
                                            lastDir,
                                            QFileDialog.ShowDirsOnly
  )
  if destDir == "":
    return ""
  return  destDir

def checkInputRasters(userData):
  if ("initial" in userData) and ("final" in userData):
    return True
  else:
    return False

def checkFactors(userData):
  if "factors" in userData:
    return True
  else:
    return False

def checkChangeMap(userData):
  if "changeMap" in userData:
    return True
  else:
    return False

def copySymbology(src, dst):
  di = QDomImplementation()
  dt = di.createDocumentType("qgis", "http://mrcc.com/qgis.dtd", "SYSTEM")
  doc = QDomDocument(dt)
  root = doc.createElement("qgis")
  root.setAttribute("version", "%s" % str(QGis.QGIS_VERSION))
  doc.appendChild(root)
  errMsg = ""
  if not src.writeSymbology(root, doc, errMsg):
    return False

  if not dst.readSymbology(root, errMsg):
    return False

  return True
