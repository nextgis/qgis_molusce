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

# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)

from typing import Dict, List, Optional, Union

import numpy as np
from qgis.core import QgsVectorLayer
from qgis.PyQt.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.correlation.model import (
    CoeffError,
    DependenceCoef,
)
from molusce.algorithms.models.crosstabs.model import CrossTabError
from molusce.algorithms.models.sampler.sampler import Sampler, SamplerError
from molusce.molusceutils import PickleQObjectMixin

from . import multinomial_logistic_regression as mlr


class LRError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg: str) -> None:
        """
        Initialize the exception with a message.

        :param msg: The error message.
        """
        self.msg = msg


class LR(PickleQObjectMixin, QObject):
    """Implements Logistic Regression model definition and calibration
    (maximum liklihood parameter estimation).
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    samplingFinished = pyqtSignal()
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)

    logreg: mlr.MLR
    sampler: Optional[Sampler]
    data: Optional[np.ndarray]
    maxiter: int

    def __init__(self, ns: int = 0, logreg: Optional[mlr.MLR] = None) -> None:
        """
        Initialize the Logistic Regression model.

        :param ns: Neighborhood size of training rasters, defaults to 0.
        :param logreg: An optional instance of the
                       multinomial logistic regression model.
        """
        super().__init__()

        if logreg:
            self.logreg = logreg
        else:
            self.logreg = mlr.MLR()

        self.state = None
        self.factors = None
        self.output = None
        self.mode = "All"
        self.samples = None
        self.catlist = None

        self.ns = ns  # Neighbourhood size of training rasters.
        self.data = None  # Training data
        self.maxiter = 100  # Maximum of fitting iterations

        self.sampler = None  # Sampler

        # Results of the LR prediction
        self.prediction = None  # Raster of the LR prediction results
        self.confidence = None  # Raster of the LR results confidence (1 = the maximum confidence, 0 = the least confidence)
        self.Kappa = 0  # Kappa value
        self.pseudoR = 0  # Pseudo R-squared (Count) (http://www.ats.ucla.edu/stat/mult_pkg/faq/general/Psuedo_RSquareds.htm)
        self.transitionPotentials = None  # Dictionary of transition potencial maps: {category1: map1, category2: map2, ...}

    def getCoef(self) -> np.ndarray:
        """
        Get the coefficients of the logistic regression model.

        :returns: A NumPy array of coefficients.
        """
        return self.logreg.get_weights().T

    def getConfidence(self) -> Optional[Raster]:
        """
        Get the confidence raster of the logistic regression results.

        :returns: A raster representing confidence values.
        """
        return self.confidence

    def getIntercept(self) -> Optional[np.ndarray]:
        """
        Get the intercepts of the logistic regression model.

        :returns: A NumPy array of intercepts.
        """
        return self.logreg.get_intercept()

    def getKappa(self) -> Union[float, Dict[str, float]]:
        """
        Get the Kappa value of the logistic regression model.

        :returns: The Kappa value.
        """
        return self.Kappa

    def getStdErrIntercept(self) -> np.ndarray:
        """
        Get the standard errors of the intercepts.

        :returns: A NumPy array of standard errors for the intercepts.
        """
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_stderr_intercept(X)

    def getStdErrWeights(self) -> np.ndarray:
        """
        Get the standard errors of the weights.

        :returns: A NumPy array of standard errors for the weights.
        """
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_stderr_weights(X).T

    def get_PvalIntercept(self) -> np.ndarray:
        """
        Get the p-values of the intercepts.

        :returns: A NumPy array of p-values for the intercepts.
        """
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_pval_intercept(X)

    def get_PvalWeights(self) -> np.ndarray:
        """
        Get the p-values of the weights.

        :returns: A NumPy array of p-values for the weights.
        """
        X = np.column_stack((self.data["state"], self.data["factors"]))
        return self.logreg.get_pval_weights(X).T

    def getPrediction(
        self,
        state: Raster,
        factors: List[Raster],
        calcTransitions: bool = False,
    ) -> Optional[Raster]:
        """
        Get the prediction raster based on the input state and factors.

        :param state: The raster of the current state (categories).
        :param factors: A list of factor rasters (predicting variables).
        :param calcTransitions: Whether to calculate transition potentials, defaults to False.

        :returns: A raster of prediction results.
        """
        self._predict(state, factors, calcTransitions)
        return self.prediction

    def getPseudoR(self) -> float:
        """
        Get the pseudo R-squared value of the logistic regression model.

        :returns: The pseudo R-squared value.
        """
        return self.pseudoR

    def getTransitionPotentials(self) -> Optional[Dict[int, Raster]]:
        """
        Get the transition potentials of the logistic regression model.

        :returns: A dictionary of transition potentials.
        """
        return self.transitionPotentials

    def _outputConfidence(self, input_data: np.ndarray) -> int:
        """Return confidence (difference between 2 biggest probabilities) of the LR output.
        1 = the maximum confidence, 0 = the least confidence
        """
        out_scl = self.logreg.predict_proba(input_data)[0]
        # Calculate the confidence:
        out_scl.sort()
        return int(100 * (out_scl[-1] - out_scl[-2]))

    def outputTransitions(self, input_data: np.ndarray) -> Dict[int, int]:
        """Return transition potential of the outputs"""
        out_scl = self.logreg.predict_proba(input_data)[0]
        out_scl = [int(100 * x) for x in out_scl]
        result = {}
        for r, v in enumerate(out_scl):
            cat = self.catlist[r]
            result[cat] = v
        return result

    def _predict(
        self,
        state: Raster,
        factors: List[Raster],
        calcTransitions: bool = False,
    ) -> None:
        """Calculate output and confidence rasters using LR model and input rasters
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        """
        try:
            self.rangeChanged.emit(self.tr("Initialize model %p%"), 1)
            geodata = state.getGeodata()
            rows, cols = geodata["ySize"], geodata["xSize"]
            for r in factors:
                if not state.geoDataMatch(r):
                    raise LRError(
                        self.tr(
                            "Geometries of the input rasters are different!"
                        )
                    )

            self.transitionPotentials = (
                None  # Reset tr.potentials if they exist
            )

            # Normalize factors before prediction:
            for f in factors:
                f.normalize(mode="mean")

            predicted_band = np.zeros([rows, cols], dtype=np.uint8)
            confidence_band = np.zeros([rows, cols], dtype=np.uint8)
            if calcTransitions:
                self.transitionPotentials = {}
                for cat in self.catlist:
                    self.transitionPotentials[cat] = np.zeros(
                        [rows, cols], dtype=np.uint8
                    )

            self.sampler = Sampler(state, factors, ns=self.ns)
            mask = state.getBand(1).mask.copy()
            if mask.shape == ():
                mask = np.zeros([rows, cols], dtype=bool)
            self.updateProgress.emit()
            self.rangeChanged.emit(self.tr("Prediction %p%"), rows)
            for i in range(rows):
                for j in range(cols):
                    if not mask[i, j]:
                        input_data = self.sampler.get_inputs(state, i, j)
                        if input_data is not None:
                            input_data = np.array([input_data])
                            out = self.logreg.predict(input_data)
                            predicted_band[i, j] = out
                            confidence = self._outputConfidence(input_data)
                            confidence_band[i, j] = confidence

                            if calcTransitions:
                                potentials = self.outputTransitions(input_data)
                                for cat in self.catlist:
                                    potential_map = self.transitionPotentials[
                                        cat
                                    ]
                                    potential_map[i, j] = potentials[cat]
                        else:  # Input sample is incomplete => mask this pixel
                            mask[i, j] = True
                self.updateProgress.emit()
            predicted_bands = [
                np.ma.array(data=predicted_band, mask=mask, dtype=np.uint8)
            ]
            confidence_bands = [
                np.ma.array(data=confidence_band, mask=mask, dtype=np.uint8)
            ]

            self.prediction = Raster()
            self.prediction.create(predicted_bands, geodata)
            self.confidence = Raster()
            self.confidence.create(confidence_bands, geodata)

            if calcTransitions:
                for cat in self.catlist:
                    band = [
                        np.ma.array(
                            data=self.transitionPotentials[cat],
                            mask=mask,
                            dtype=np.uint8,
                        )
                    ]
                    self.transitionPotentials[cat] = Raster()
                    self.transitionPotentials[cat].create(band, geodata)
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during LR prediction")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during LR prediction")
            )
            raise
        finally:
            self.processFinished.emit()

    def __propagateSamplerSignals(self) -> None:
        """
        Connect the sampler signals to the corresponding slot methods.
        """
        self.sampler.rangeChanged.connect(self.__samplerProgressRangeChanged)
        self.sampler.updateProgress.connect(self.__samplerProgressChanged)
        self.sampler.samplingFinished.connect(self.__samplerFinished)

    @pyqtSlot()
    def __samplerFinished(self) -> None:
        """
        Handle the sampler finished signal and disconnect its connections.
        """
        self.sampler.rangeChanged.disconnect(
            self.__samplerProgressRangeChanged
        )
        self.sampler.updateProgress.disconnect(self.__samplerProgressChanged)
        self.sampler.samplingFinished.disconnect(self.__samplerFinished)
        self.samplingFinished.emit()

    @pyqtSlot(str, int)
    def __samplerProgressRangeChanged(
        self, message: str, max_value: int
    ) -> None:
        """
        Handle the sampler progress range changed signal.

        :param message: The message with progress information.
        :param max_value: The maximum value of the range.
        """
        self.rangeChanged.emit(message, max_value)

    @pyqtSlot()
    def __samplerProgressChanged(self) -> None:
        """
        Handle the sampler progress changed signal.
        """
        self.updateProgress.emit()

    def save(self):
        pass

    def createSamplePointsLayer(self) -> QgsVectorLayer:
        """
        Returns sample points as temporary QgsVectorLayer.
        """
        return self.sampler.createSamplePointsLayer()

    def setMaxIter(self, maxiter: int) -> None:
        """
        Set the maximum number of iterations for the logistic regression model.

        :param maxiter: The maximum number of iterations.
        """
        self.maxiter = maxiter

    def setTrainingData(self) -> None:
        """
        Set the training data for the logistic regression model.
        """
        state, factors, output, mode, samples = (
            self.state,
            self.factors,
            self.output,
            self.mode,
            self.samples,
        )
        if not self.logreg:
            raise LRError(
                self.tr("You must create a Logistic Regression model before!")
            )

        # Normalize factors before sampling:
        for factor in factors:
            factor.normalize(mode="mean")

        self.sampler = Sampler(state, factors, output, ns=self.ns)
        self.__propagateSamplerSignals()
        self.sampler.setTrainingData(
            state, output, shuffle=False, mode=mode, samples=samples
        )

        self.data = self.sampler.data
        assert self.data is not None
        self.catlist = np.unique(self.data["output"])

    def train(self) -> None:
        """
        Train the logistic regression model with the current training data.
        """
        assert self.data is not None
        X = np.column_stack((self.data["state"], self.data["factors"]))
        Y = self.data["output"]
        self.labelCodes = np.unique(Y)
        self.logreg.fit(X, Y, maxiter=self.maxiter)
        out = self.logreg.predict(X)
        depCoef = DependenceCoef(np.ma.array(out), np.ma.array(Y), expand=True)
        self.Kappa = depCoef.kappa(mode=None)
        self.pseudoR = depCoef.correctness(percent=False)

    def setState(self, state: Raster) -> None:
        """
        Set the state raster for the logistic regression model.

        :param state: The state raster (categories).
        """
        self.state = state

    def setFactors(self, factors: List[Raster]) -> None:
        """
        Set the factor rasters for the logistic regression model.

        :param factors: A list of factor rasters (predicting variables).
        """
        self.factors = factors

    def setOutput(self, output: Raster) -> None:
        """
        Set the output raster for the logistic regression model.

        :param output: The output raster (classifications).
        """
        self.output = output

    def setMode(self, mode: str) -> None:
        """
        Set the mode for the logistic regression model.

        :param mode: The mode for prediction (e.g., 'All', 'Histo', 'Loc').
        """
        self.mode = mode

    def setSamples(self, samples: int) -> None:
        """
        Set the number of samples for training.

        :param samples: The number of samples.
        """
        self.samples = samples

    @pyqtSlot()
    def startTrain(self) -> None:
        """
        Start the training process for the Logistic Regression model.

        This method performs the following steps:
        - Sets up the training data.
        - Trains the logistic regression model.
        """
        try:
            self.setTrainingData()
            self.train()
        except SamplerError as error:
            self.error_occurred.emit(self.tr("Sampling error"), str(error))
            return
        except CrossTabError as error:
            self.error_occurred.emit(
                self.tr("Model training failed"), str(error)
            )
            return
        except CoeffError as error:
            self.error_occurred.emit(
                self.tr("Model training failed"), str(error)
            )
            return
        except LRError as error:
            self.error_occurred.emit(self.tr("Missed LR model"), str(error))
            return
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during LR training")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during LR training")
            )
            raise
        finally:
            self.finished.emit()
