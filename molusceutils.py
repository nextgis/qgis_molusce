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

def getRasterLayers():
  layerMap = QgsMapLayerRegistry.instance().mapLayers()
  layers = dict()
  for name, layer in layerMap.iteritems():
    if layer.type() == QgsMapLayer.RasterLayer and layer.providerType() == "gdal":
      if layer.id() not in layers.keys():
        layers[layer.id()] = unicode(layer.name())
  return layers

def getLayerById(layerId):
  layerMap = QgsMapLayerRegistry.instance().mapLayers()
  for name, layer in layerMap.iteritems():
    if layer.id() == layerId:
      if layer.isValid():
        return layer
      else:
        return None

def getLayerGroup(relations, layerId):
  group = None

  for item in relations:
    group = unicode(item[0])
    for lid in item[1]:
      if unicode(lid) == unicode(layerId):
        return group

  return group

def saveRasterDialog(parent, settings, title, fileFilter):
  lastDir = settings.value("ui/lastRasterDir", ".").toString()
  fileName = QFileDialog.getSaveFileName(parent,
                                         title,
                                         lastDir,
                                         fileFilter
                                        )

  if fileName.isEmpty():
    return QString()

  if not fileName.toLower().contains(QRegExp("\.tif{1,2}")):
    fileName += ".tif"

  settings.setValue("ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath())

  return fileName

def openRasterDialog(parent, settings, title, fileFilter):
  lastDir = settings.value("ui/lastRasterDir", ".").toString()
  fileName = QFileDialog.getOpenFileName(parent,
                                         title,
                                         lastDir,
                                         fileFilter
                                        )

  if fileName.isEmpty():
    return QString()

  settings.setValue("ui/lastRasterDir", QFileInfo(fileName).absoluteDir().absolutePath())

  return fileName

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

