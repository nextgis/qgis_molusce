# -*- coding: utf-8 -*-
"""
Prepare vector dataset for MOLUSCE.

Creates a raster in the spatial domain of a reference raster layer based on
vector data, using either presence (rasterization) or proximity mode.
"""

from qgis import processing
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication


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
    PROX_MODE = "PROX_MODE"
    OUTPUT = "OUTPUT"

    MODE_PRESENCE = 0
    MODE_PROXIMITY = 1

    def tr(self, string: str) -> str:
        return QCoreApplication.translate("MoluscePrepareVectorAlgorithm", string)

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

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR,
                self.tr("Vector layer to be processed"),
                [QgsWkbTypes.PointGeometry,
                 QgsWkbTypes.LineGeometry,
                 QgsWkbTypes.PolygonGeometry],
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
                defaultValue=self.MODE_PRESENCE,
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD,
                self.tr(
                    "Numeric field for raster values (Presence mode only, leave empty to use 1 for each feature)"
                ),
                parentLayerParameterName=self.INPUT_VECTOR,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER,
                self.tr(
                    "Buffer zone size (map units, Presence mode only; leave empty for no buffer)"
                ),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=None,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.PROX_MODE,
                self.tr("Proximity units (Proximity mode only)"),
                options=[
                    self.tr("Georeferenced units"),
                    self.tr("Pixels"),
                ],
                defaultValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr("Output raster"),
            )
        )

    def processAlgorithm(self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        vector_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)
        ref_raster = self.parameterAsRasterLayer(parameters, self.REFERENCE_RASTER, context)

        if vector_layer is None:
            raise QgsProcessingException(self.tr("Vector layer is not valid."))
        if ref_raster is None:
            raise QgsProcessingException(self.tr("Reference raster layer is not valid."))

        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        if not output_path:
            raise QgsProcessingException(self.tr("Could not determine output raster path."))

        mode = self.parameterAsEnum(parameters, self.MODE, context)

        # Numeric field (Presence mode)
        field_name = self.parameterAsString(parameters, self.FIELD, context)
        if field_name is None:
            field_name = ""

        # Buffer size (Presence mode)
        buffer_size = None
        raw_buffer = parameters.get(self.BUFFER, None)
        if raw_buffer not in (None, ""):
            try:
                buffer_size = float(raw_buffer)
            except Exception:
                buffer_size = None

        # Proximity units (Proximity mode)
        prox_mode_index = self.parameterAsEnum(parameters, self.PROX_MODE, context)

        if feedback.isCanceled():
            return {}

        ref_provider = ref_raster.dataProvider()
        cols = ref_provider.xSize()
        rows = ref_provider.ySize()

        if cols <= 0 or rows <= 0:
            raise QgsProcessingException(
                self.tr("Reference raster has invalid size (width or height is zero).")
            )

        ref_extent = ref_raster.extent()
        extent_string = "{},{},{},{}".format(
            ref_extent.xMinimum(),
            ref_extent.xMaximum(),
            ref_extent.yMinimum(),
            ref_extent.yMaximum(),
        )

        target_crs = ref_raster.crs()
        if not target_crs.isValid():
            raise QgsProcessingException(self.tr("Reference raster has invalid CRS."))

        if vector_layer.crs() != target_crs:
            feedback.pushInfo(
                self.tr("Reprojecting vector layer to reference raster CRS...")
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
            vector_layer = reproject_result["OUTPUT"]

        vector_for_rasterize = vector_layer

        if mode == self.MODE_PRESENCE and buffer_size is not None and buffer_size > 0:
            feedback.pushInfo(
                self.tr("Buffering features by {0} map units (Presence mode)...").format(
                    buffer_size
                )
            )
            buffer_result = processing.run(
                "native:buffer",
                {
                    "INPUT": vector_layer,
                    "DISTANCE": buffer_size,
                    "SEGMENTS": 5,
                    "END_CAP_STYLE": 0,  # Round
                    "JOIN_STYLE": 0,     # Round
                    "MITER_LIMIT": 2,
                    "DISSOLVE": False,
                    "SEPARATE_DISJOINT": False,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            vector_for_rasterize = buffer_result["OUTPUT"]

        if feedback.isCanceled():
            return {}

        if mode == self.MODE_PROXIMITY:
            # Proximity mode: build simple presence raster with BURN=1
            feedback.pushInfo(self.tr("Rasterizing vector layer (for Proximity mode)..."))

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
                "DATA_TYPE": 1,  # Byte
                "INIT": 0,       # initialize background to 0
                "INVERT": False,
                "EXTRA": "",
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            }

            presence_result = processing.run(
                "gdal:rasterize",
                presence_raster_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            presence_raster_path = presence_result["OUTPUT"]

            if feedback.isCanceled():
                return {}

            feedback.pushInfo(self.tr("Computing proximity raster..."))

            # UNITS: 0 = Georeferenced units, 1 = Pixels
            distunits = 0 if prox_mode_index == 0 else 1

            proximity_params = {
                "INPUT": presence_raster_path,
                "BAND": 1,
                "VALUES": "1",
                "UNITS": distunits,
                "MAX_DISTANCE": 0,    # no limit
                "NODATA": 0,
                "OPTIONS": "",
                "DATA_TYPE": 5,       # Float32
                "EXTRA": "",
                "OUTPUT": output_path,
            }

            prox_result = processing.run(
                "gdal:proximity",
                proximity_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )

            return {self.OUTPUT: prox_result["OUTPUT"]}

        else:
            feedback.pushInfo(self.tr("Rasterizing vector layer (Presence mode)..."))

            rasterize_params = {
                "INPUT": vector_for_rasterize,
                "USE_Z": False,
                "UNITS": 0,
                "WIDTH": cols,
                "HEIGHT": rows,
                "EXTENT": extent_string,
                "NODATA": 0,
                "OPTIONS": "",
                "DATA_TYPE": 5,  # Float32
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

            return {self.OUTPUT: result["OUTPUT"]}
