from typing import List, Optional, Tuple

import numpy as np
from numpy import ma as ma
from qgis.PyQt.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.utils import masks_identity
from molusce.molusceutils import PickleQObjectMixin


class AreaAnalizerError(Exception):
    """
    Base class for exceptions in the AreaAnalyst module.

    :param msg: Error message describing the issue.
    """

    def __init__(self, msg: str):
        self.msg = msg


class AreaAnalizerCategoryError(AreaAnalizerError):
    """
    Exception raised for category-related errors in AreaAnalyst.
    """


class AreaAnalyst(PickleQObjectMixin, QObject):
    """
    Generates an output raster, with geometry
    copied from the initial land use map.  The output is a 1-band raster
    with categories corresponding the (r,c) elements of the m-matrix of
    categories transitions, so that if for a given pixel the initial category is r,
    the final category c, and there are m categories, the output pixel will have
    value k = r*m + c
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal(object)
    errorReport = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    logMessage = pyqtSignal(str)

    def __init__(self, first: Raster, second: Optional[Raster] = None) -> None:
        """
        Initialize the AreaAnalyst.

        :param first: Raster of the first stage (state before transition).
        :param second: Raster of the second stage (state after transition).
        :raises AreaAnalizerError: If geometries mismatch or raster bands are invalid.
        """
        QObject.__init__(self)

        if second is not None and (not first.geoDataMatch(second)):
            raise AreaAnalizerError(
                self.tr("Geometries of the rasters are different!")
            )

        if first.getBandsCount() != 1:
            raise AreaAnalizerError(self.tr("First raster must have 1 band!"))

        if second is not None and second.getBandsCount() != 1:
            raise AreaAnalizerError(self.tr("Second raster must have 1 band!"))

        self.geodata = first.getGeodata()
        self.categories = first.getBandGradation(1)

        if second is not None:
            self.categoriesSecond = second.getBandGradation(1)
            first, second = masks_identity(
                first.getBand(1), second.getBand(1), dtype=np.uint8
            )

        self.first = first
        self.second = second

        if second is not None:
            for cat in self.categoriesSecond:
                if cat not in self.categories:
                    raise AreaAnalizerError(
                        self.tr(
                            "List of categories of the first raster doesn't contains a category of the second raster!"
                        )
                    )

        self.changeMap = None
        self.initRaster = None
        self.persistentCategoryCode = -1

    def codes(self, initialClass: int) -> List[int]:
        """
        Get list of possible encodes for initialClass (see 'encode').

        :param initialClass: Initial category class.

        :return: List of encoded values.
        """
        return [self.encode(initialClass, f) for f in self.categories]

    def decode(self, code: int) -> Tuple[int, int]:
        """
        Decode transition (initialClass -> finalClass).
        The procedure is the back operation of "encode" (see encode):
            code = initialClass*m + finalClass,
            the result is tuple of (initialClass, finalClass).

        :param code: Encoded transition value.

        :return: Tuple of (initialClass, finalClass).

        :raises AreaAnalizerCategoryError: If the code is invalid.
        """
        m = len(self.categories)
        initialClassIndex = code // m
        finalClassIndex = code - initialClassIndex * m
        try:
            initClass, finalClass = (
                self.categories[initialClassIndex],
                self.categories[finalClassIndex],
            )
        except ValueError as exc:
            raise AreaAnalizerCategoryError(
                self.tr("The code is not in list!")
            ) from exc
        return (initClass, finalClass)

    def encode(self, initialClass: int, finalClass: int) -> int:
        """
        Encode transition (initialClass -> finalClass):
        if for a given pixel the initial category is initialClass,
        the final category finalClass, and there are m categories, the output pixel will have
        value k = initialClass*m + finalClass

        :param initialClass: Initial category class.
        :param finalClass: Final category class.
        :return: Encoded transition value.

        :raises AreaAnalizerCategoryError: If the category is invalid.
        """
        m = len(self.categories)
        try:
            code = self.categories.index(
                initialClass
            ) * m + self.categories.index(finalClass)
        except ValueError as exc:
            raise AreaAnalizerCategoryError(
                self.tr("The category not in list of categories!")
            ) from exc

        return code

    def finalCodes(self, initialClass: int) -> List[int]:
        """
        For given initial category return codes of possible final categories. (see 'encode')

        :param initialClass: Initial category class.

        :return: List of encoded final categories.
        """
        return [self.encode(initialClass, c) for c in self.categories]

    def getChangeMap(self) -> Optional[Raster]:
        """
        Get the change map raster.

        :return: Change map raster.
        """
        if self.changeMap is None:
            try:
                self.makeChangeMap()
            except AreaAnalizerCategoryError as error:
                self.error_occurred.emit(
                    self.tr("Change map error"), str(error)
                )
                return None
        return self.changeMap

    def makeChangeMap(self) -> None:
        """
        Create a change map raster based on the input rasters.

        :raises MemoryError: If the system runs out of memory.
        :raises AreaAnalizerError: If an unknown error occurs.
        """
        rows, cols = self.geodata["ySize"], self.geodata["xSize"]
        band = np.zeros([rows, cols], dtype=np.int16)

        f, s = self.first, self.second
        if self.initRaster is None:
            checkPersistent = False
        else:
            checkPersistent = True
            t = self.initRaster.getBand(1)
        raster = None
        try:
            self.rangeChanged.emit(self.tr("Creating change map %p%"), rows)
            for i in range(rows):
                for j in range(cols):
                    if (f.mask.shape == ()) or (not f.mask[i, j]):
                        r = f[i, j]
                        c = s[i, j]
                        # Percistent category is the category that is constant for all three rasters
                        if checkPersistent and (r == c) and (r == t[i, j]):
                            band[i, j] = self.persistentCategoryCode
                        else:
                            band[i, j] = self.encode(r, c)
                self.updateProgress.emit()
            bands = [np.ma.array(data=band, mask=f.mask, dtype=np.int16)]
            raster = Raster()
            raster.create(bands, self.geodata)
            self.changeMap = raster
        except MemoryError:
            self.errorReport.emit(
                self.tr(
                    "The system is out of memory during change map creating"
                )
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during change map creating")
            )
            raise
        finally:
            self.processFinished.emit(raster)

    def removeInitialRaster(self) -> None:
        """
        Remove the initial raster from the analysis.
        """
        self.initRaster = None

    def setInitialRaster(self, initR: Raster) -> None:
        """
        Set the initial raster for the analysis.

        :param initR: Initial raster to set.

        :raises AreaAnalizerError: If geometries mismatch or raster bands are invalid.
        """
        if not initR.geoDataMatch(raster=None, geodata=self.geodata):
            raise AreaAnalizerError(
                self.tr("Geometries of the rasters are different!")
            )
        if initR.getBandsCount() != 1:
            raise AreaAnalizerError(self.tr("The raster must have 1 band!"))

        self.initRaster = initR
