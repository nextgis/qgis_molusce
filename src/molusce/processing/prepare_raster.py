# -*- coding: utf-8 -*-

import math

from osgeo import gdal
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtCore import QCoreApplication


class MoluscePrepareRasterAlgorithm(QgsProcessingAlgorithm):
    """
    Prepare raster dataset for MOLUSCE.

    - Input raster is resampled/reprojected to the spatial domain of reference raster
    - Output has the same extent, rows/cols and CRS as the reference raster
    """

    INPUT_RASTER = "INPUT_RASTER"
    REFERENCE_RASTER = "REFERENCE_RASTER"
    NODATA = "NODATA"
    RESAMPLING = "RESAMPLING"
    OUTPUT = "OUTPUT"

    # Resampling labels, matching GDAL resampling algorithms order
    RESAMPLING_METHODS = [
        "Nearest neighbour",
        "Bilinear (2x2 kernel)",
        "Cubic (4x4 kernel)",
        "Cubic B-Spline (4x4 kernel)",
        "Lanczos (6x6 kernel)",
        "Average",
        "Mode",
        "Maximum",
        "Minimum",
        "Median",
        "First quartile (Q1)",
        "Third quartile (Q3)",
    ]

    def tr(self, string: str) -> str:
        return QCoreApplication.translate("MoluscePrepareRasterAlgorithm", string)

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

    def initAlgorithm(self, configuration=None):
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
                type=QgsProcessingParameterNumber.Double,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESAMPLING,
                self.tr("Resampling algorithm"),
                options=self.RESAMPLING_METHODS,
                defaultValue=0,  # Nearest neighbour
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr("Prepared raster"),
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        gdal.UseExceptions()

        input_layer = self.parameterAsRasterLayer(
            parameters, self.INPUT_RASTER, context
        )
        if input_layer is None:
            raise QgsProcessingException(
                self.tr("Input raster layer is not valid.")
            )

        ref_layer = self.parameterAsRasterLayer(
            parameters, self.REFERENCE_RASTER, context
        )
        if ref_layer is None:
            raise QgsProcessingException(
                self.tr("Reference raster layer is not valid.")
            )

        output_path = self.parameterAsOutputLayer(
            parameters, self.OUTPUT, context
        )
        if not output_path:
            raise QgsProcessingException(
                self.tr("Could not determine output raster path.")
            )

        ref_provider = ref_layer.dataProvider()
        cols = ref_provider.xSize()
        rows = ref_provider.ySize()

        if cols <= 0 or rows <= 0:
            raise QgsProcessingException(
                self.tr(
                    "Reference raster has invalid size (width or height is zero)."
                )
            )

        ref_extent = ref_layer.extent()
        xmin = ref_extent.xMinimum()
        xmax = ref_extent.xMaximum()
        ymin = ref_extent.yMinimum()
        ymax = ref_extent.yMaximum()

        target_crs = ref_layer.crs()
        if not target_crs.isValid():
            raise QgsProcessingException(
                self.tr("Reference raster has invalid CRS.")
            )

        input_crs = input_layer.crs()
        if not input_crs.isValid():
            raise QgsProcessingException(
                self.tr("Input raster has invalid CRS.")
            )

        input_source = input_layer.source()

        if feedback.isCanceled():
            return {}

        nodata_value = None
        if self.NODATA in parameters and parameters[self.NODATA] not in (
            None,
            "",
        ):
            try:
                nodata_value = float(parameters[self.NODATA])
            except Exception:
                nodata_value = None

        resampling_index = self.parameterAsInt(
            parameters, self.RESAMPLING, context
        )

        # Map to GDAL resampling algorithms
        resampling_map = {
            0: gdal.GRA_NearestNeighbour,
            1: gdal.GRA_Bilinear,
            2: gdal.GRA_Cubic,
            3: gdal.GRA_CubicSpline,
            4: gdal.GRA_Lanczos,
            5: gdal.GRA_Average,
            6: gdal.GRA_Mode,
            7: gdal.GRA_Max,
            8: gdal.GRA_Min,
            9: gdal.GRA_Med,
            10: gdal.GRA_Q1,
            11: gdal.GRA_Q3,
        }

        resample_alg = resampling_map.get(resampling_index, gdal.GRA_NearestNeighbour)

        feedback.pushInfo(
            self.tr("Running gdal.Warp to align raster to reference grid...")
        )

        try:
            warp_kwargs = {
                "format": "GTiff",
                "dstSRS": target_crs.toWkt(),
                "srcSRS": input_crs.toWkt(),
                "outputBounds": (xmin, ymin, xmax, ymax),
                "width": cols,
                "height": rows,
                "resampleAlg": resample_alg,
                "multithread": True,
            }

            if nodata_value is not None and not math.isnan(nodata_value):
                warp_kwargs["dstNodata"] = nodata_value

            ds = gdal.Warp(
                destNameOrDestDS=output_path,
                srcDSOrSrcDSTab=input_source,
                **warp_kwargs,
            )

            if ds is None:
                raise QgsProcessingException(
                    self.tr("gdal.Warp returned None – failed to create output raster.")
                )

            # Flush to disk
            ds.FlushCache()
            ds = None

        except Exception as exc:
            raise QgsProcessingException(self.tr("Error while running gdal.Warp: {}").format(str(exc)))

        if feedback.isCanceled():
            return {}

        return {self.OUTPUT: output_path}