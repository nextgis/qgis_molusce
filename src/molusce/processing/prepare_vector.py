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

"""
Prepare vector dataset for MOLUSCE.

Creates a raster in the spatial domain of a reference raster layer based on
vector data, using either presence (rasterization) or proximity mode.
"""

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

from osgeo import gdal
from qgis import processing
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,  # pyright: ignore[reportAttributeAccessIssue]
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingUtils,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QCoreApplication

from molusce.compat import (
    ProcessingFieldParameterDataType,
    ProcessingNumberParameterType,
    ProcessingSourceType,
)


class MoluscePrepareVectorAlgorithm(QgsProcessingAlgorithm):
    """
    Prepare vector dataset for MOLUSCE.

    Output raster:
    - has the same CRS, extent and number of rows/columns as the reference raster
    - is pixel-wise aligned with the reference raster
    """

    INPUT_VECTOR = "INPUT_VECTOR"
    REFERENCE_RASTER = "REFERENCE_RASTER"
    MODE = "MODE"
    FIELD = "FIELD"
    BUFFER = "BUFFER"
    PROXIMITY_MODE = "PROXIMITY_MODE"
    OUTPUT = "OUTPUT"

    class Mode(IntEnum):
        PRESENCE = 0
        PROXIMITY = 1

    class ProximityMode(IntEnum):
        GEOREFERENCED_UNITS = 0
        PIXELS = 1

    def tr(self, string: str) -> str:
        return QCoreApplication.translate(
            "MoluscePrepareVectorAlgorithm", string
        )

    def name(self) -> str:
        return "molusce_prepare_vector"

    def displayName(self) -> str:
        return self.tr("Prepare vector dataset for MOLUSCE")

    def shortHelpString(self) -> str:
        return self.tr(
            "This tool prepares vector datasets for use within the MOLUSCE workflow. It supports two processing modes:\n"
            "• Presence mode — Vector features are rasterized as they are, with an optional buffer applied to expand feature influence around their geometry.\n"
            "• Proximity mode — A proximity surface is generated, producing a raster in which each pixel contains the distance to the nearest vector feature.\n\n"
            "The result is a new raster representation of the input vector data, fully aligned to the spatial domain of a reference raster. The output inherits the coordinate reference system (CRS), spatial extent, and pixel dimensions (rows/columns) from the reference raster to ensure pixel-wise alignment.\n"
            "Use the input landcover raster as the reference to ensure consistency across spatial variables.\n"
            "This tool enables efficient preparation of predictor rasters derived from vector data for use in landcover change modelling with MOLUSCE."
        )

    def createInstance(self):
        return MoluscePrepareVectorAlgorithm()

    def initAlgorithm(self, configuration: Dict[str, Any] = None) -> None:  # pyright: ignore[reportArgumentType]
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR,
                self.tr("Vector layer to be processed"),
                [
                    ProcessingSourceType.VectorPoint,
                    ProcessingSourceType.VectorLine,
                    ProcessingSourceType.VectorPolygon,
                ],
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.REFERENCE_RASTER,
                self.tr("Reference raster layer"),
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                self.tr("Mode"),
                options=[
                    self.tr("Presence (rasterize features)"),
                    self.tr("Proximity (distance to nearest feature)"),
                ],
                defaultValue=self.Mode.PRESENCE.value,
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD,
                self.tr(
                    "Numeric field for raster values (Presence mode only, leave empty to use 1 for each feature)"
                ),
                parentLayerParameterName=self.INPUT_VECTOR,
                type=ProcessingFieldParameterDataType.Numeric,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER,
                self.tr(
                    "Buffer zone size (map units, Presence mode only; leave empty for no buffer)"
                ),
                type=ProcessingNumberParameterType.Double,
                defaultValue=None,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.PROXIMITY_MODE,
                self.tr("Proximity units (Proximity mode only)"),
                options=[
                    self.tr("Georeferenced units"),
                    self.tr("Pixels"),
                ],
                defaultValue=self.ProximityMode.GEOREFERENCED_UNITS.value,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr("Output raster"),
            )
        )

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: Optional[QgsProcessingFeedback],
    ) -> Dict[str, Any]:
        assert feedback is not None

        vector_layer, raster_reference_layer = self._get_layers(
            parameters, context
        )
        output_path = self._get_output_path(parameters, context)

        mode = self.Mode(self.parameterAsEnum(parameters, self.MODE, context))
        field_name = self._get_field_name(parameters, context)
        buffer_size = self._get_buffer_size(parameters)
        proximity_mode_index = self.ProximityMode(
            self.parameterAsEnum(parameters, self.PROXIMITY_MODE, context)
        )

        if feedback.isCanceled():
            return {}

        cols, rows, extent_string, target_crs = self._collect_reference_info(
            raster_reference_layer
        )

        vector_layer = self._ensure_vector_layer_crs(
            vector_layer, target_crs, context, feedback
        )
        if feedback.isCanceled():
            return {}

        if mode == self.Mode.PRESENCE:
            return self._run_presence_mode(
                vector_layer,
                field_name,
                cols,
                rows,
                extent_string,
                buffer_size,
                output_path,
                context,
                feedback,
            )

        if mode == self.Mode.PROXIMITY:
            return self._run_proximity_mode(
                vector_layer,
                cols,
                rows,
                extent_string,
                proximity_mode_index,
                output_path,
                context,
                feedback,
            )

        raise QgsProcessingException(self.tr("Invalid mode selected."))

    def _get_layers(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
    ) -> Tuple[QgsVectorLayer, QgsRasterLayer]:
        vector_layer = self.parameterAsVectorLayer(
            parameters, self.INPUT_VECTOR, context
        )
        raster_reference_layer = self.parameterAsRasterLayer(
            parameters, self.REFERENCE_RASTER, context
        )

        if vector_layer is None:
            raise QgsProcessingException(self.tr("Vector layer is not valid."))
        if raster_reference_layer is None:
            raise QgsProcessingException(
                self.tr("Reference raster layer is not valid.")
            )

        return vector_layer, raster_reference_layer

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

    def _get_field_name(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
    ) -> str:
        """Field name for presence mode"""
        field_name = self.parameterAsString(parameters, self.FIELD, context)
        return field_name if field_name is not None else ""

    def _get_buffer_size(self, parameters: Dict[str, Any]) -> Optional[float]:
        """Buffer size for presence mode"""
        raw_buffer = parameters.get(self.BUFFER, None)
        if raw_buffer in (None, ""):
            return None

        try:
            return float(raw_buffer)
        except Exception:
            return None

    def _collect_reference_info(
        self, raster_reference_layer: QgsRasterLayer
    ) -> Tuple[int, int, str, QgsCoordinateReferenceSystem]:
        ref_provider = raster_reference_layer.dataProvider()
        cols = ref_provider.xSize()
        rows = ref_provider.ySize()

        if cols <= 0 or rows <= 0:
            raise QgsProcessingException(
                self.tr(
                    "Reference raster has invalid size (width or height is zero)."
                )
            )

        reference_layer_extent = raster_reference_layer.extent()
        extent_string = "{},{},{},{}".format(
            reference_layer_extent.xMinimum(),
            reference_layer_extent.xMaximum(),
            reference_layer_extent.yMinimum(),
            reference_layer_extent.yMaximum(),
        )

        target_crs = raster_reference_layer.crs()
        if not target_crs.isValid():
            raise QgsProcessingException(
                self.tr("Reference raster has invalid CRS.")
            )

        return cols, rows, extent_string, target_crs

    def _ensure_vector_layer_crs(
        self,
        vector_layer: QgsVectorLayer,
        target_crs: QgsCoordinateReferenceSystem,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> QgsVectorLayer:
        if vector_layer.crs() == target_crs:
            return vector_layer

        feedback.pushInfo(
            self.tr("Reprojecting vector layer to reference raster CRS..."),
        )
        reproject_result = processing.run(
            "native:reprojectlayer",
            {
                "INPUT": vector_layer,
                "TARGET_CRS": target_crs,
                "OPERATION": "",
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        if feedback.isCanceled():
            return QgsVectorLayer()

        if reproject_result is None or "OUTPUT" not in reproject_result:
            raise QgsProcessingException(
                self.tr("Failed to reproject vector layer.")
            )
        return reproject_result["OUTPUT"]

    def _buffer_vector_for_presence(
        self,
        vector_layer: QgsVectorLayer,
        buffer_size: Optional[float],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> QgsVectorLayer:
        if buffer_size is None or buffer_size <= 0:
            return vector_layer

        feedback.pushInfo(
            self.tr(
                "Buffering features by {0} map units (Presence mode)..."
            ).format(buffer_size),
        )

        end_cap_style = self._end_cap_style_to_processing(
            Qgis.EndCapStyle.Round
        )
        join_style = self._join_style_to_processing(Qgis.JoinStyle.Round)

        buffer_result = processing.run(
            "native:buffer",
            {
                "INPUT": vector_layer,
                "DISTANCE": buffer_size,
                "SEGMENTS": 5,
                "END_CAP_STYLE": end_cap_style,
                "JOIN_STYLE": join_style,
                "MITER_LIMIT": 2,
                "DISSOLVE": False,
                "SEPARATE_DISJOINT": False,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        if feedback.isCanceled():
            return QgsVectorLayer()
        if buffer_result is None or "OUTPUT" not in buffer_result:
            raise QgsProcessingException(
                self.tr("Failed to buffer vector layer for Presence mode.")
            )

        return buffer_result["OUTPUT"]

    def _run_proximity_mode(
        self,
        vector_layer: QgsVectorLayer,
        cols: int,
        rows: int,
        extent_string: str,
        proximity_mode_index: int,
        output_path: str,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> Dict[str, Any]:
        # Build simple presence raster with BURN=1
        feedback.pushInfo(
            self.tr("Rasterizing vector layer (for Proximity mode)..."),
        )

        tmp_presence = QgsProcessingUtils.generateTempFilename("presence.tif")

        presence_raster_params = {
            "INPUT": vector_layer,
            "FIELD": None,  # ignore numeric field
            "BURN": 1.0,
            "USE_Z": False,
            "UNITS": 0,  # 0 = Pixels
            "WIDTH": cols,
            "HEIGHT": rows,
            "EXTENT": extent_string,
            "NODATA": 0,
            "OPTIONS": "",
            "DATA_TYPE": gdal.GDT_Byte,
            "INIT": 0,  # initialize background to 0
            "INVERT": False,
            "EXTRA": "",
            "OUTPUT": tmp_presence,
        }

        presence_result = processing.run(
            "gdal:rasterize",
            presence_raster_params,
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        if feedback.isCanceled():
            return {}
        if presence_result is None or "OUTPUT" not in presence_result:
            raise QgsProcessingException(
                self.tr("Failed to rasterize vector layer for Proximity mode.")
            )

        presence_raster_path = presence_result["OUTPUT"]

        feedback.pushInfo(self.tr("Computing proximity raster..."))

        # UNITS: 0 = Georeferenced units, 1 = Pixels
        distunits = 0 if proximity_mode_index == 0 else 1

        proximity_params = {
            "INPUT": presence_raster_path,
            "BAND": 1,
            "VALUES": "1",
            "UNITS": distunits,
            "MAX_DISTANCE": 0,  # no limit
            "NODATA": 0,
            "OPTIONS": "",
            "DATA_TYPE": gdal.GDT_Float32,
            "EXTRA": "",
            "OUTPUT": output_path,
        }

        proximity_result = processing.run(
            "gdal:proximity",
            proximity_params,
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        if feedback.isCanceled():
            return {}
        if proximity_result is None or "OUTPUT" not in proximity_result:
            raise QgsProcessingException(
                self.tr("Failed to compute proximity raster.")
            )

        return {self.OUTPUT: proximity_result["OUTPUT"]}

    def _run_presence_mode(
        self,
        vector_layer: QgsVectorLayer,
        field_name: str,
        cols: int,
        rows: int,
        extent_string: str,
        buffer_size: Optional[float],
        output_path: str,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> Dict[str, Any]:
        vector_layer = self._buffer_vector_for_presence(
            vector_layer, buffer_size, context, feedback
        )
        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            self.tr("Rasterizing vector layer (Presence mode)...")
        )

        rasterize_params = {
            "INPUT": vector_layer,
            "USE_Z": False,
            "UNITS": 0,
            "WIDTH": cols,
            "HEIGHT": rows,
            "EXTENT": extent_string,
            "NODATA": 0,
            "OPTIONS": "",
            "DATA_TYPE": gdal.GDT_Float32,
            "INIT": 0,
            "INVERT": False,
            "EXTRA": "",
            "OUTPUT": output_path,
        }

        if field_name:
            rasterize_params["FIELD"] = field_name
        else:
            rasterize_params["FIELD"] = None
            rasterize_params["BURN"] = 1.0

        result = processing.run(
            "gdal:rasterize",
            rasterize_params,
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        if feedback.isCanceled():
            return {}
        if result is None or "OUTPUT" not in result:
            raise QgsProcessingException(
                self.tr("Failed to rasterize vector layer.")
            )
        return {self.OUTPUT: result["OUTPUT"]}

    @staticmethod
    def _end_cap_style_to_processing(end_cap: Qgis.EndCapStyle) -> int:
        """
        Convert a Qgis.EndCapStyle value to a Processing enum index
        suitable for the native:buffer algorithm.

        Qgis.EndCapStyle values start at 1, while Processing enum
        parameters are zero-based indices.

        :param end_cap: End cap style defined by the QGIS geometry API.
        :type end_cap: Qgis.EndCapStyle

        :returns: Zero-based Processing enum value for END_CAP_STYLE.
        :rtype: int
        """
        return int(end_cap) - 1

    @staticmethod
    def _join_style_to_processing(join_style: Qgis.JoinStyle) -> int:
        """
        Convert a Qgis.JoinStyle value to a Processing enum index
        suitable for the native:buffer algorithm.

        Qgis.JoinStyle values start at 1, while Processing enum
        parameters are zero-based indices.

        :param join_style: Join style defined by the QGIS geometry API.
        :type join_style: Qgis.JoinStyle

        :returns: Zero-based Processing enum value for JOIN_STYLE.
        :rtype: int
        """
        return int(join_style) - 1
