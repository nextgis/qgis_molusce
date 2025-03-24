from pathlib import Path

import numpy as np
from numpy import ma as ma
from osgeo import gdal, osr

from .utils import binaryzation, get_gradations

# If a raster band has more then MAX_CATEGORIES categories, we think that the band contains continues values
MAX_CATEGORIES = 256


class ProviderError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class FormatConverter:
    """Tarnslates formats between GDAL and numpy data formats"""

    def __init__(self):
        self.dtypes = (
            bool,
            int,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            float,
            np.float32,
            np.float64,
        )  # np.float16
        self.GDT = (
            gdal.GDT_Byte,
            gdal.GDT_UInt16,
            gdal.GDT_Int16,
            gdal.GDT_UInt32,
            gdal.GDT_Int32,
            gdal.GDT_Float32,
            gdal.GDT_Float64,
        )
        self.dtype2GDT = {  # raster type, max value
            np.dtype("bool"): (gdal.GDT_Byte, 255),
            np.dtype("int"): (gdal.GDT_Int32, 2147483647),  #!!!
            np.dtype("int8"): (gdal.GDT_Int16, 32767),
            np.dtype("int16"): (gdal.GDT_Int16, 32767),
            np.dtype("int32"): (gdal.GDT_Int32, 2147483647),
            np.dtype("int64"): (gdal.GDT_Int32, 2147483647),  #!!!
            np.dtype("uint8"): (gdal.GDT_Byte, 255),
            np.dtype("uint16"): (gdal.GDT_UInt16, 65535),
            np.dtype("uint32"): (gdal.GDT_UInt32, 4294967295),
            np.dtype("uint64"): (gdal.GDT_UInt32, 4294967295),  #!!!
            np.dtype("float"): (gdal.GDT_Float64, -9999),
            # np.dtype('float16'): (gdal.GDT_Float32,    -9999),
            np.dtype("float32"): (gdal.GDT_Float32, -9999),
            np.dtype("float64"): (gdal.GDT_Float64, -9999),
        }


class Raster:
    # TODO: use dtypes for raster inutialization
    def __init__(self, filename=None, maskVals=None):
        if filename == "":
            raise ProviderError("File name can't be empty string!")
        if isinstance(filename, Path):
            filename = str(filename)
        self.filename = filename
        self.maskVals = maskVals  # List of the "transparent" pixel values
        self.bands = None  # Array of the bands (stored as numpy mask array)
        self.bandcount = 0  # Count of the bands. (numpy.array.shape is too slow => we need to save number of bands)
        self.geodata = None  # Georeferensing information
        self.stat = None  # Initial (before normalizing) statistic (means and stds) of the bands
        self.isNormalazed = None  # Is the bands of the raster normalized? It contains the mode of normalization.
        self.bandgradation = dict()  # Dict for store lists of bands categories
        if self.filename:
            self._read()

    def binaryzation(self, trueVals, bandNum):
        """Reclass band bandNum to true/false mode. Set true for pixels from trueVals."""
        r = self.getBand(bandNum)
        r = binaryzation(r, trueVals)
        self.setBand(r, bandNum)

    def create(self, bands, geodata):
        self.bands = np.ma.array(bands)
        self.bandcount = len(bands)
        # for i in range(1, self.bandcount+1):
        #    self.bandgradation[i] = None
        self.geodata = geodata

    def denormalize(self):
        """Denormalisation (see self.normalize)"""
        if self.isNormalazed:
            mode = self.isNormalazed
            bandcount = self.getBandsCount()
            for i in range(1, bandcount + 1):
                stat = self.stat[i - 1]
                if mode == "mean":
                    newBand = (
                        1.0 * self.getBand(i) * stat["std"] + stat["mean"]
                    )
                elif mode == "maxmin":
                    newBand = (
                        1.0 * self.getBand(i) * (stat["max"] - stat["min"])
                        - stat["min"]
                    )
                else:
                    raise ProviderError("The normalization mode is unknown!")
                self.setBand(newBand, i)
            self.isNormalazed = False

    def geoDataMatch(self, raster, geodata=None):
        """Return true if RasterSize, Projection and GetGeoTransform of the rasters are matched"""
        if (geodata is not None) and (raster is not None):
            raise ProviderError("Use raster or geodata, not both!")

        if geodata is None:
            geodata = raster.geodata

        # TODO: check proj also.
        for key in ["xSize", "ySize"]:
            if self.geodata[key] != geodata[key]:
                return False
        return self.geoTransformMatch(raster=None, geodata=geodata)

    def geoTransformMatch(self, raster, geodata=None):
        """Return True if GetGeoTransform of the rasters are matched:
        1) the difference of the top left x less then pixel size,
        2) the difference of the top left y less then pixel size,
        3) for x and y:
            (difference of the pixel sizes) * (pixel count) < (pixel size),
        4) rotations are equal.
        """
        if (geodata is not None) and (raster is not None):
            raise ProviderError("Use raster or geodata, not both!")

        if geodata is None:
            geodata = raster.geodata
        s_cornerX, s_width, s_rot1, s_cornerY, s_rot2, s_height = self.geodata[
            "transform"
        ]
        r_cornerX, r_width, r_rot1, r_cornerY, r_rot2, r_height = geodata[
            "transform"
        ]
        if (s_rot1 != r_rot1) or (s_rot2 != r_rot2):
            return False

        dx = abs(s_cornerX - r_cornerX)
        if dx > min(abs(s_width), abs(r_width)):
            return False
        dy = abs(s_cornerY - r_cornerY)
        if dy > min(abs(s_height), abs(r_height)):
            return False
        dw = abs(s_width - r_width)
        if dw * self.geodata["xSize"] > 1.5 * min(abs(s_width), abs(r_width)):
            return False
        dh = abs(s_height - r_height)
        return not dh * self.geodata["ySize"] > 1.5 * min(
            abs(s_height), abs(r_height)
        )

    def getBand(self, bandNo):
        return self.bands[bandNo - 1]

    def getBandsCount(self) -> int:
        return self.bandcount

    def getBandGradation(self, bandNo):
        """Return list of categories of raster's band"""
        if bandNo < len(self.bandgradation):
            res = self.bandgradation[bandNo]
        else:
            band = self.getBand(bandNo)
            res = get_gradations(band.compressed())
            self.bandgradation[bandNo] = res
        return res

    def getBandStat(self, bandNo):
        """Return mean and std of the raster's band"""
        band = self.getBand(bandNo)
        result = {}
        result["mean"] = np.ma.mean(band)
        result["std"] = np.ma.std(band)
        result["min"] = np.ma.min(band)
        result["max"] = np.ma.max(band)
        return result

    def get_dtype(self):
        # All bands of the raster have the same dtype now
        band = self.getBand(1)
        return band.dtype

    def getGDALMaxVal(self):
        dtype = self.get_dtype()
        conv = FormatConverter()
        rastertype, maxVal = conv.dtype2GDT[dtype]
        return maxVal

    def getFileName(self):
        return self.filename

    def getGeodata(self):
        return self.geodata

    def getNeighbours(self, row, col, size):
        """Return subset of the bands -- neighbourhood of the central pixel (row,col)"""
        bcount = self.getBandsCount()
        row_size = 2 * size + 1  # Length of the neighbourhood square side
        pixel_count = (
            bcount * row_size**2
        )  # Count of pixels in the neighbourhood
        neighbourhood = self.bands[
            :bcount,
            row - size : (row + size + 1),
            col - size : (col + size + 1),
        ]
        # neighbourhood = neighbourhood.flatten()
        if len(neighbourhood.flatten()) != pixel_count:
            raise ProviderError(
                "Incorrect neighbourhood size or the central pixel lies on the raster boundary."
            )
        return neighbourhood

    def getNeighbourhoodSize(self, ns):
        """Return pixel count in the neighbourhood of ns size"""
        # pixel count in the 1-band neighbourhood of ns size
        neighbours = (2 * ns + 1) ** 2
        return self.getBandsCount() * neighbours

    def getPixelFromBand(self, row, col, band=1):
        """Return pixel value from band"""
        return self.bands[band - 1, row, col]

    def getPixelArea(self):
        cornerX, width, rot1, cornerY, rot2, height = self.geodata["transform"]
        return {"area": abs(width * height), "unit": self.getProjUnits()}

    def getPixelCoords(self, px, py):
        """Pixel to Coords transform
        @param  px        Input x pixel coordinate
        @param  py        Input y pixel coordinate
        @return outx,outy Output coordinates (two doubles)
        """
        gt = self.geodata["transform"]
        outx = gt[0] + px * gt[1] + py * gt[2]
        outy = gt[3] + px * gt[4] + py * gt[5]
        return (outx, outy)

    def getProjUnits(self):
        return self.geodata["units"]

    def getXSize(self):
        return self.geodata["xSize"]

    def getYSize(self):
        return self.geodata["ySize"]

    def isCountinues(self, bandNo):
        """Return true, if the band contains continues value."""
        return len(self.getBandGradation(bandNo)) > MAX_CATEGORIES

    def isMetricProj(self):
        """Return true if projection of the raster uses metric units"""
        return self.getProjUnits() in ("metre", "Meter")

    def normalize(self, mode="mean"):
        """Linear normalization of the bands: new = (old-mean(old)/std(old))

        @param mode     Type of normalization:
                mean    new = (old-mean(old)/std(old))
                maxmin  new = (old-min(old)/(max(old)-min(old))
        """
        if self.isNormalazed != mode:
            self.denormalize()  # Reset raster values to initail
            bandcount = self.getBandsCount()
            self.stat = []
            for i in range(1, bandcount + 1):
                stat = self.getBandStat(i)
                self.stat.append(stat)
                if mode == "mean":
                    newBand = (
                        1.0 * (self.getBand(i) - stat["mean"]) / stat["std"]
                    )
                elif mode == "maxmin":
                    newBand = (
                        1.0
                        * (self.getBand(i) - stat["min"])
                        / (stat["max"] - stat["min"])
                    )
                else:
                    raise ProviderError("The normalization mode is unknown!")
                self.setBand(newBand, i)
            self.isNormalazed = mode

    def _read(self):
        try:
            data = gdal.Open(self.filename)
        except RuntimeError as exc:
            raise ProviderError(
                f"Can't read the file '{self.filename}'"
            ) from exc
        if data is None:
            raise ProviderError(f"Can't read the file '{self.filename}'")

        self.geodata = {}
        self.geodata["xSize"] = data.RasterXSize
        self.geodata["ySize"] = data.RasterYSize
        self.geodata["proj"] = data.GetProjection()
        self.geodata["transform"] = data.GetGeoTransform()

        # Get units of the projection
        sr = osr.SpatialReference()
        sr.ImportFromWkt(self.geodata["proj"])
        self.geodata["units"] = sr.GetLinearUnitsName()

        self.bands = np.ma.zeros(
            (data.RasterCount, self.geodata["ySize"], self.geodata["xSize"]),
            dtype=float,
        )
        self.bandcount = data.RasterCount
        if self.maskVals is not None and len(self.maskVals) != self.bandcount:
            raise ProviderError(
                f"Band count of the file '{self.filename}' does't match nodata values count"
            )

        for i in range(1, data.RasterCount + 1):
            r = data.GetRasterBand(i)
            nodataValue = r.GetNoDataValue()
            nodataValues = [nodataValue] if nodataValue is not None else []
            r = r.ReadAsArray()
            # self.bandgradation[i] = None
            if self.maskVals is None:
                userNodataValues = []
            else:
                try:
                    userNodataValues = self.maskVals[i]
                except TypeError:
                    userNodataValues = []
            nodataValues += userNodataValues
            if nodataValues != []:
                mask = binaryzation(r, nodataValues)
                r = ma.array(data=r, mask=mask)
            else:
                r = ma.array(data=r, mask=False)
            self.bands[i - 1, :, :] = r
        self.isNormalazed = False

    def resetMask(self, maskVals=None):
        """Set mask of _ALL_ bands.  maskVals is a list of masked values."""
        for i in range(self.getBandsCount()):
            r = self.getBand(i)
            mask = binaryzation(r, maskVals) if maskVals is not None else False
            r = ma.array(data=r, mask=mask)
            self.setBand(r, i)

    def roundBands(self, decimals=0):
        """Round the bands to the given number of decimals."""
        self.bands = np.around(self.bands, decimals)

    def save(
        self, filename, driver_name="GTiff", rastertype=None, nodata=None
    ):
        driver = gdal.GetDriverByName(driver_name)
        metadata = driver.GetMetadata()
        if (
            gdal.DCAP_CREATE in metadata
            and metadata[gdal.DCAP_CREATE] == "YES"
        ):
            if not rastertype:
                dtype = self.get_dtype()
                conv = FormatConverter()
                rastertype, maxVal = conv.dtype2GDT[dtype]
            if nodata is None:
                nodata = maxVal
            xsize, ysize = self.getXSize(), self.getYSize()
            bandcount = self.getBandsCount()
            outRaster = driver.Create(
                filename, xsize, ysize, bandcount, rastertype
            )
            geodata = self.getGeodata()
            outRaster.SetProjection(geodata["proj"])
            outRaster.SetGeoTransform(geodata["transform"])
            for i in range(bandcount):
                band = self.getBand(i + 1)
                outRaster.GetRasterBand(i + 1).WriteArray(
                    band.filled(fill_value=nodata)
                )
                outRaster.GetRasterBand(i + 1).SetNoDataValue(nodata)
            outRaster = None
        else:
            raise ProviderError(
                f"Driver {driver_name} does not support Create() method!"
            )

    def setBand(self, raster, bandNum=1):
        self.bands[bandNum - 1] = raster

    def setGeoData(self, geodata):
        # Check raster's geometry
        if self.getBandsCount() > 0:
            band = self.getBand(1)
            if band.shape != (geodata["xSize"], geodata["ySize"]):
                raise ProviderError(
                    "Existing bands don't match new geodata geometry!"
                )

        self.geodata = geodata
