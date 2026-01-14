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

import math
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

from osgeo import gdal
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,  # pyright: ignore[reportAttributeAccessIssue]
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsRasterLayer,
)
from qgis.PyQt.QtCore import QCoreApplication

from molusce.compat import ProcessingNumberParameterType


class MoluscePrepareRasterAlgorithm(QgsProcessingAlgorithm):
    """
    Prepare raster dataset for MOLUSCE.

    - Input raster is resampled/reprojected to the spatial domain of reference raster
    - Output has the same extent, rows/cols and CRS as the reference raster
    """

    class ResamplingAlgorithm(IntEnum):
        # Use own numeric values because gdal has gaps in its enum values
        NEAREST_NEIGHBOUR = auto()
        BILINEAR = auto()
        CUBIC = auto()
        CUBIC_BSPLINE = auto()
        LANCZOS = auto()
        AVERAGE = auto()
        MODE = auto()
        MAXIMUM = auto()
        MINIMUM = auto()
        MEDIAN = auto()
        FIRST_QUARTILE = auto()
        THIRD_QUARTILE = auto()

    INPUT_RASTER = "INPUT_RASTER"
    REFERENCE_RASTER = "REFERENCE_RASTER"
    NODATA = "NODATA"
    RESAMPLING = "RESAMPLING"
    OUTPUT = "OUTPUT"

    def tr(self, string: str) -> str:
        return QCoreApplication.translate(
            "MoluscePrepareRasterAlgorithm", string
        )

    def warp_tr(self, string: str) -> str:
        """QGIS translation context for gdal:warpreproject."""
        return QCoreApplication.translate("warp", string)

    def name(self) -> str:
        return "molusce_prepare_raster"

    def displayName(self) -> str:
        return self.tr("Prepare raster dataset for MOLUSCE")

    def shortHelpString(self) -> str:
        return self.tr(
            "This tool prepares raster datasets for use in the MOLUSCE workflow. It resamples and reprojects an input raster so that it matches the spatial domain of a given reference raster. The output raster inherits the coordinate reference system (CRS), spatial extent, and the number of rows and columns from the reference raster, ensuring pixel-wise alignment between both datasets.\n"
            "Use the input landcover raster as the reference to which all auxiliary spatial variables will be aligned.\n"
            "This tool streamlines the preparation of predictor rasters required for landcover change modelling in MOLUSCE.\n"
            "\n"
            "Choose the resampling method based on the characteristics of your dataset. For continuous data such as digital elevation models and their derivatives, cubic (or another interpolation-based) resampling is recommended. For categorical data such as land cover classes, nearest neighbour should be used to preserve class integrity.\n"
        )

    def createInstance(self):
        return MoluscePrepareRasterAlgorithm()

    def initAlgorithm(self, configuration: Dict[str, Any] = None) -> None:  # pyright: ignore[reportArgumentType]
        RESAMPLING_ALGORITHM_NAMES = {
            self.ResamplingAlgorithm.NEAREST_NEIGHBOUR: self.warp_tr(
                "Nearest Neighbour"
            ),
            self.ResamplingAlgorithm.BILINEAR: self.warp_tr(
                "Bilinear (2x2 Kernel)"
            ),
            self.ResamplingAlgorithm.CUBIC: self.warp_tr("Cubic (4x4 Kernel)"),
            self.ResamplingAlgorithm.CUBIC_BSPLINE: self.warp_tr(
                "Cubic B-Spline (4x4 Kernel)"
            ),
            self.ResamplingAlgorithm.LANCZOS: self.warp_tr(
                "Lanczos (6x6 Kernel)"
            ),
            self.ResamplingAlgorithm.AVERAGE: self.warp_tr("Average"),
            self.ResamplingAlgorithm.MODE: self.warp_tr("Mode"),
            self.ResamplingAlgorithm.MAXIMUM: self.warp_tr("Maximum"),
            self.ResamplingAlgorithm.MINIMUM: self.warp_tr("Minimum"),
            self.ResamplingAlgorithm.MEDIAN: self.warp_tr("Median"),
            self.ResamplingAlgorithm.FIRST_QUARTILE: self.warp_tr(
                "First Quartile (Q1)"
            ),
            self.ResamplingAlgorithm.THIRD_QUARTILE: self.warp_tr(
                "Third Quartile (Q3)"
            ),
        }

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr("Raster layer to be prepared"),
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.REFERENCE_RASTER,
                self.tr("Reference raster layer"),
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.NODATA,
                self.tr(
                    "NoData value for output (leave empty to copy from input raster)"
                ),
                type=ProcessingNumberParameterType.Double,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESAMPLING,
                self.tr("Resampling algorithm"),
                options=RESAMPLING_ALGORITHM_NAMES.values(),
                defaultValue=self.ResamplingAlgorithm.NEAREST_NEIGHBOUR.value,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr("Prepared raster"),
            )
        )

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: Optional[QgsProcessingFeedback],
    ) -> Dict[str, Any]:
        assert feedback is not None

        gdal.UseExceptions()

        input_layer, reference_layer = self._get_layers(parameters, context)

        input_source = input_layer.source()
        input_crs = input_layer.crs()
        if not input_crs.isValid():
            raise QgsProcessingException(
                self.tr("Input raster has invalid CRS.")
            )

        cols, rows, bounds, target_crs = self._collect_reference_info(
            reference_layer
        )

        output_path = self._get_output_path(parameters, context)

        nodata_value = self._parse_nodata(parameters)
        resampling_algorithm = self.ResamplingAlgorithm(
            self.parameterAsInt(parameters, self.RESAMPLING, context)
        )
        gdal_resampling_algorithm = self._gdal_resampling_algorithm(
            resampling_algorithm
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            self.tr("Running gdal.Warp to align raster to reference grid...")
        )

        self._run_warp(
            input_source=input_source,
            output_path=output_path,
            input_crs=input_crs,
            target_crs=target_crs,
            bounds=bounds,
            size=(cols, rows),
            resampling_algorithm=gdal_resampling_algorithm,
            nodata_value=nodata_value,
        )
        return {self.OUTPUT: output_path}

    def _get_layers(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
    ) -> Tuple[QgsRasterLayer, QgsRasterLayer]:
        input_layer = self.parameterAsRasterLayer(
            parameters, self.INPUT_RASTER, context
        )
        if input_layer is None:
            raise QgsProcessingException(
                self.tr("Input raster layer is not valid.")
            )

        reference_layer = self.parameterAsRasterLayer(
            parameters, self.REFERENCE_RASTER, context
        )
        if reference_layer is None:
            raise QgsProcessingException(
                self.tr("Reference raster layer is not valid.")
            )

        return input_layer, reference_layer

    def _get_output_path(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
    ) -> str:
        output_path = self.parameterAsOutputLayer(
            parameters, self.OUTPUT, context
        )
        if not output_path:
            raise QgsProcessingException(
                self.tr("Could not determine output raster path.")
            )
        return output_path

    def _collect_reference_info(
        self, raster_reference_layer: QgsRasterLayer
    ) -> Tuple[
        int,
        int,
        Tuple[float, float, float, float],
        QgsCoordinateReferenceSystem,
    ]:
        provider = raster_reference_layer.dataProvider()

        cols = provider.xSize()
        rows = provider.ySize()
        if cols <= 0 or rows <= 0:
            raise QgsProcessingException(
                self.tr(
                    "Reference raster has invalid size (width or height is zero)."
                )
            )

        extent = raster_reference_layer.extent()
        bounds = (
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        )

        target_crs = raster_reference_layer.crs()
        if not target_crs.isValid():
            raise QgsProcessingException(
                self.tr("Reference raster has invalid CRS."),
            )

        return (cols, rows, bounds, target_crs)

    def _parse_nodata(self, parameters: Dict[str, Any]) -> Optional[float]:
        if self.NODATA not in parameters:
            return None

        raw_value = parameters[self.NODATA]
        if raw_value in (None, ""):
            return None

        try:
            return float(raw_value)
        except Exception:
            return None

    def _gdal_resampling_algorithm(
        self, resampling_algorithm: ResamplingAlgorithm
    ) -> int:
        # Map our ResamplingAlgorithm to GDAL constants
        resampling_map = {
            self.ResamplingAlgorithm.NEAREST_NEIGHBOUR: gdal.GRA_NearestNeighbour,
            self.ResamplingAlgorithm.BILINEAR: gdal.GRA_Bilinear,
            self.ResamplingAlgorithm.CUBIC: gdal.GRA_Cubic,
            self.ResamplingAlgorithm.CUBIC_BSPLINE: gdal.GRA_CubicSpline,
            self.ResamplingAlgorithm.LANCZOS: gdal.GRA_Lanczos,
            self.ResamplingAlgorithm.AVERAGE: gdal.GRA_Average,
            self.ResamplingAlgorithm.MODE: gdal.GRA_Mode,
            self.ResamplingAlgorithm.MAXIMUM: gdal.GRA_Max,
            self.ResamplingAlgorithm.MINIMUM: gdal.GRA_Min,
            self.ResamplingAlgorithm.MEDIAN: gdal.GRA_Med,
            self.ResamplingAlgorithm.FIRST_QUARTILE: gdal.GRA_Q1,
            self.ResamplingAlgorithm.THIRD_QUARTILE: gdal.GRA_Q3,
        }
        return resampling_map.get(
            resampling_algorithm, gdal.GRA_NearestNeighbour
        )

    def _run_warp(
        self,
        input_source: str,
        output_path: str,
        input_crs: QgsCoordinateReferenceSystem,
        target_crs: QgsCoordinateReferenceSystem,
        bounds: Tuple[float, float, float, float],
        size: Tuple[int, int],
        resampling_algorithm: int,
        nodata_value: Optional[float],
    ) -> None:
        cols, rows = size
        try:
            warp_kwargs = {
                "format": "GTiff",
                "dstSRS": target_crs.toWkt(),
                "srcSRS": input_crs.toWkt(),
                "outputBounds": bounds,
                "width": cols,
                "height": rows,
                "resampleAlg": resampling_algorithm,
                "multithread": True,
            }

            if nodata_value is not None and not math.isnan(nodata_value):
                warp_kwargs["dstNodata"] = nodata_value

            dataset = gdal.Warp(
                destNameOrDestDS=output_path,
                srcDSOrSrcDSTab=input_source,
                **warp_kwargs,
            )

            if dataset is None:
                raise QgsProcessingException(
                    self.tr(
                        "gdal.Warp returned None – failed to create output raster."
                    )
                )

            # Flush to disk
            dataset.FlushCache()
            dataset = None

        except Exception as exc:
            raise QgsProcessingException(
                self.tr("Error while running gdal.Warp: {}").format(str(exc))
            ) from exc
