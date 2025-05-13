import os.path
from typing import Optional

import numpy as np
from numpy import ma as ma
from osgeo import ogr, osr
from qgis.PyQt.QtCore import *

from molusce.algorithms.dataprovider import ProviderError, Raster
from molusce.molusceutils import PickleQObjectMixin


class SamplerError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class Sampler(PickleQObjectMixin, QObject):
    """Create training set based on input-output rasters.

    A sample is a set of input data for a model and output data that has to be predicted via the model.

    A sample contains:

    coordinates of pixel,
    input data consists of 2 parts:
        state is data that is read from 1-band raster, this raster contains initaial states (categories). Categories are splitted into set of dummy variables.
        factors is list of rasters (multiband probably) that explain transition between states (categories).
    output data is read from 1-band raster, this raster contains final states.

    In the simplest case we have pixel-by-pixel model. In such case:
        sample = np.array(
            ([Dummy_variables_for_pixel_from_state_raster], [pixel_from_factor1, ..., pixel_from_factorN], pixel_from_output_raster),
            dtype=[('state', float, 1),('factors',  float, N), ('output', float, 1)]
        )
    But we can use moving windows to collect samples, then input data contains several (eg 3x3) pixels for every raster (band).
    For example if we use 1-pixel neighbourhood (3x3 moving windows):
        sample = np.array(
            ( [Dummy1_1st-pixel_from_state_raster,..., DummyK_1st-pixel_from_state_raster, ..., DummyK_9th-pixel_from_state_raster],
              [1st-pixel_from_factor1, ..., 9th-pixel_from_factor1, ..., 1st-pixel_from_factorN..., 9th-pixel_from_factorN],
              pixel_from_output_raster
            ),
            dtype=[('state', float, 9*DummyVariablesCount),('factors',  float, 9*N), ('output', float, 1)]
        )
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    samplingFinished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    data: Optional[np.ndarray]

    def __init__(self, state, factors, output=None, ns=0):
        """@param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        @param output           Raster that contains states (categories) to predict.
        @param ns               Neighbourhood size.
        """
        QObject.__init__(self)

        self.ns = ns  # Neighbourhood size

        self.factorsGeoData = state.getGeodata()
        for _r in factors:
            if not state.geoDataMatch(
                raster=None, geodata=self.factorsGeoData
            ):
                raise SamplerError(
                    self.tr(
                        "Geometries of the inputs and output rasters are different!"
                    )
                )

        self.stateCategories = state.getBandGradation(
            1
        )  # Categories of state raster
        self.categoriesCount = len(
            self.stateCategories
        )  # Count of the categories

        self.outputVecLen = 1  # Len of output vector
        # Len of the vector of input states:
        self.stateVecLen = (
            self.categoriesCount - 1
        ) * state.getNeighbourhoodSize(self.ns)
        # Set up dummy variables
        self.catCodes = {}
        self.__calcCatVector()

        self.factorCount = len(factors)
        self.factorVectLen = 0  # Length of vector of the factor's pixels
        self.factors = []
        for raster in factors:
            self.factorVectLen = (
                self.factorVectLen + raster.getNeighbourhoodSize(self.ns)
            )
            for bandNum in range(raster.getBandsCount()):
                self.factors.append(raster.getBand(bandNum + 1))
        self.factors = np.ma.array(self.factors, dtype=float)

        self.proj = self.factorsGeoData[
            "proj"
        ]  # Projection of the data coordinates
        self.data = None  # Sample data

    def __calcCatVector(self):
        """Split state category value into set of dummy variables and save them in a dictionary.
        self.stateCategories[-1] is base category.
        For example:
            if self.stateCategories = [1,2,3] then dummy vars are [V1, V2]: cat1 = [1, 0], cat2 = [0, 1], cat3 = [0 ,0]
        """
        for cat in self.stateCategories[:-1]:
            vect = np.zeros(self.categoriesCount - 1)
            num = self.stateCategories.index(cat)
            vect[num] = 1.0
            self.catCodes[cat] = vect
        self.catCodes[self.stateCategories[-1]] = vect = np.zeros(
            self.categoriesCount - 1
        )

    def cat2vect(self, category):
        """Return dummy variables for the category.
        @param category     The category number.
        """
        return self.catCodes[category]

    def getData(self):
        return self.data

    def get_inputs(self, state, row, col):
        """@param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        """
        try:
            state_data = self.get_state(state, row, col)
            if state_data is None:  # Eliminate incomplete samples
                return None
            factors_data = self.get_factors(row, col)
            if factors_data is None:  # Eliminate incomplete samples
                return None
        except ProviderError:
            return None
        return np.hstack((state_data, factors_data))

    def get_factors(self, row: str, col: str) -> Optional[np.ma.MaskedArray]:
        """Get input sample at (row, col) pixel and return it as array. Return None if the sample is incomplete."""
        neighbours = self.factors[
            :,
            row - self.ns : (row + self.ns + 1),
            col - self.ns : (col + self.ns + 1),
        ].flatten()

        # Mask neighbours.mask can be boolean array or single boolean => set it as iterable object
        mask = neighbours.mask
        if mask.shape == ():
            mask = [mask]
        if any(mask):  # Eliminate incomplete samples
            return None
        return neighbours

    def get_state(self, state, row, col):
        """Get current state at (row, col) pixel and return it as array. Return None if the sample is incomplete.
        The result is [Dummy_var1_for_pix1, ... Dummy_varK_for_pix1, ..., Dummy_var1_for_pixS, ... Dummy_varK_for_pixS],
            where K is count of dummy variables, S is count of pixels in the neighbours of the pixel.
        """
        neighbours = state.getNeighbours(row, col, self.ns).flatten()

        # Mask neighbours.mask can be boolean array or single boolean => set it as iterable object
        mask = neighbours.mask
        if mask.shape == ():
            mask = [mask]

        if any(mask):  # Eliminate incomplete samples
            return None
        result = np.zeros(self.stateVecLen)
        for i, cat in enumerate(neighbours):
            result[
                i * (self.categoriesCount - 1) : self.categoriesCount
                - 1
                + i * (self.categoriesCount - 1)
            ] = self.cat2vect(cat)
        return result

    def _getSample(self, state, output, row, col):
        """Get one sample from (row,col) pixel. See params in setTrainingData."""
        state_params = (
            ("state", float, self.stateVecLen)
            if self.stateVecLen > 1
            else ("state", float)
        )
        factors_params = (
            ("factors", float, self.factorVectLen)
            if self.factorVectLen > 1
            else ("factors", float)
        )
        output_params = (
            ("output", float, self.outputVecLen)
            if self.outputVecLen > 1
            else ("output", float)
        )
        data = np.zeros(
            1,
            dtype=[
                ("coords", float, 2),
                state_params,
                factors_params,
                output_params,
            ],
        )
        try:
            out_data = output.getPixelFromBand(
                row, col, band=1
            )  # Get the pixel
            if out_data is None:  # Eliminate masked samples
                return None

            data["output"] = out_data

            state_data = self.get_state(state, row, col)
            if state_data is None:  # Eliminate incomplete samples
                return None

            data["state"] = state_data

            factors_data = self.get_factors(row, col)
            if factors_data is None:  # Eliminate incomplete samples
                return None

            data["factors"] = factors_data
        except ProviderError:
            return None
        x, y = state.getPixelCoords(col, row)
        data["coords"] = x, y
        return data  # (coords, state_data, factors_data, out_data)

    def saveSamples(self, fileName):
        data = self.getData()
        if data is None:
            raise SamplerError(self.tr("Samples cannot be created!"))

        workdir = os.path.dirname(fileName)
        fileName = os.path.splitext(os.path.basename(fileName))[0]

        driver = ogr.GetDriverByName("ESRI Shapefile")
        sr = osr.SpatialReference()
        sr.ImportFromWkt(self.proj)

        ds = driver.CreateDataSource(workdir)
        lyr = ds.CreateLayer(fileName, sr, ogr.wkbPoint)
        if lyr is None:
            raise SamplerError(self.tr("Creating output file failed!"))

        fieldnames = ["state" + str(i) for i in range(self.stateVecLen)]
        fieldnames = fieldnames + [
            "factor" + str(i) for i in range(self.factorVectLen)
        ]
        fieldnames = fieldnames + [
            "out" + str(i) for i in range(self.outputVecLen)
        ]

        for name in fieldnames:
            field_defn = ogr.FieldDefn(name, ogr.OFTReal)
            if lyr.CreateField(field_defn) != 0:
                raise SamplerError(self.tr("Creating Name field failed!"))

        for row in data:
            x, y = row["coords"]
            if x and y:
                feat = ogr.Feature(lyr.GetLayerDefn())
                if self.stateVecLen > 1:
                    for i in range(self.stateVecLen):
                        name = fieldnames[i]
                        r = row["state"][i]
                        feat.SetField(name, r)
                else:
                    name = fieldnames[0]
                    r = row["state"]
                    feat.SetField(name, r)
                if self.factorVectLen > 1:
                    for i in range(self.factorVectLen):
                        name = fieldnames[i + self.stateVecLen]
                        r = row["factors"][i]
                        feat.SetField(name, r)
                else:
                    name = fieldnames[self.stateVecLen]
                    r = row["factors"]
                    feat.SetField(name, r)
                if self.outputVecLen > 1:
                    for i in range(self.outputVecLen):
                        name = fieldnames[
                            i + self.stateVecLen + self.factorVectLen
                        ]
                        r = row["output"][i]
                        feat.SetField(name, r)
                else:
                    name = fieldnames[self.stateVecLen + self.factorVectLen]
                    r = row["output"]
                    feat.SetField(name, r)
                pt = ogr.Geometry(ogr.wkbPoint)
                pt.SetPoint_2D(0, x, y)
                feat.SetGeometry(pt)
                if lyr.CreateFeature(feat) != 0:
                    raise SamplerError(
                        self.tr("Failed to create feature in shapefile!")
                    )
                feat.Destroy()
        ds = None

    def setTrainingData(
        self,
        state: Raster,
        output: Raster,
        shuffle: bool = True,
        mode: str = "All",
        samples: Optional[int] = None,
    ) -> None:
        """@param state            Raster of the current state (categories) values.
        @param output           Raster of the output (target) data
        @param shuffle          Perform random shuffle.
        @param mode             Type of sampling method:
                                    All             Get all pixels
                                    Random          Get samples. Count of samples in the data=samples.
                                    Stratified      Undersampling of major categories and/or oversampling of minor categories.
        @samples                Sample count of the training data (doesn't used in 'All' mode).
        """
        try:
            geodata = self.factorsGeoData
            for r in [state, output]:
                if not r.geoDataMatch(raster=None, geodata=geodata):
                    raise SamplerError(
                        self.tr(
                            "Geometries of the inputs or output rasters are distinct from factor's geometry!"
                        )
                    )

            # Real count of the samples
            # (if self.ns>0 some samples may be incomplete because a neighbour has NoData value)
            samples_count = 0

            cols, rows = state.getXSize(), state.getYSize()

            if mode == "All":
                # Approximate sample count:
                band = state.getBand(1)
                nulls = band.mask.sum()  # Count of NA
                samples = rows * cols - nulls

            # Array for samples
            state_params = (
                ("state", float, self.stateVecLen)
                if self.stateVecLen > 1
                else ("state", float)
            )
            factors_params = (
                ("factors", float, self.factorVectLen)
                if self.factorVectLen > 1
                else ("factors", float)
            )
            output_params = (
                ("output", float, self.outputVecLen)
                if self.outputVecLen > 1
                else ("output", float)
            )
            self.data = np.zeros(
                samples,
                dtype=[
                    ("coords", float, 2),
                    state_params,
                    factors_params,
                    output_params,
                ],
            )

            if mode == "All":
                self.rangeChanged.emit(
                    self.tr("Sampling..."), rows - 2 * self.ns
                )
                # i,j  are pixel indexes
                for i in range(
                    self.ns, rows - self.ns
                ):  # Eliminate the raster boundary (of (ns)-size width) because
                    for j in range(
                        self.ns, cols - self.ns
                    ):  # the samples are incomplete in that region
                        sample = self._getSample(state, output, i, j)
                        if sample is not None:
                            self.data[samples_count] = sample
                            samples_count = samples_count + 1
                    self.updateProgress.emit()
                self.data = self.data[
                    :samples_count
                ]  # Crop unused part of the array

            elif mode == "Random":
                self.rangeChanged.emit(self.tr("Sampling..."), samples)
                while samples_count < samples:
                    row = np.random.randint(rows)
                    col = np.random.randint(cols)
                    sample = self._getSample(state, output, row, col)
                    if sample is not None:
                        self.data[samples_count] = sample
                        samples_count = samples_count + 1
                        self.updateProgress.emit()
            elif mode == "Stratified":
                # Analyze output categories:
                categories = output.getBandGradation(1)
                band = output.getBand(1)

                # Select pixels
                average = samples / len(categories)

                samples_count = 0
                self.rangeChanged.emit(self.tr("Sampling..."), samples)
                # Get counts[i] samples of "cat" categories
                for i, cat in enumerate(categories):
                    # Find indices of "cat"-category pixels
                    rows, cols = np.where(band == cat)
                    indices = [(rows[i], cols[i]) for i in range(len(cols))]

                    # Get samples
                    count = 0
                    while count < average:
                        index = np.random.randint(len(indices))
                        row, col = indices[index]
                        sample = self._getSample(state, output, row, col)
                        if sample is not None and samples_count < samples:
                            self.data[samples_count] = sample
                            samples_count = samples_count + 1
                            count = count + 1
                            self.updateProgress.emit()
                        else:
                            count = count + 1
            else:
                raise SamplerError(self.tr("The mode of sampling is unknown!"))

            if shuffle:
                np.random.shuffle(self.data)
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during sampling")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during sampling")
            )
            raise
        finally:
            self.samplingFinished.emit()
