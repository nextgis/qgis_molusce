# TODO: make abstract class for all models/managers
# to prevent code coping of common methods (for example _predict method)

import copy

import numpy as np
from qgis.PyQt.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.correlation.model import (
    CoeffError,
    DependenceCoef,
)
from molusce.algorithms.models.mlp.model import MLP, sigmoid
from molusce.algorithms.models.sampler.sampler import Sampler
from molusce.molusceutils import PickleQObjectMixin


class MlpManagerError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


class MlpManager(PickleQObjectMixin, QObject):
    """This class gets the data extracted from the UI and
    pass it to multi-layer perceptron, then gets and stores the result.
    """

    updateGraph = pyqtSignal()
    updateGraphValues = pyqtSignal(float, float)  # Train error, val. error
    updateMinValErr = pyqtSignal(float)  # Min validation error
    updateDeltaRMS = pyqtSignal(
        float
    )  # Delta of RMS: min(valError) - currentValError
    updateKappa = pyqtSignal(float)  # Kappa value
    processFinished = pyqtSignal()
    processInterrupted = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()

    def __init__(self, ns=0, MLP=None):
        super().__init__()

        self.MLP = MLP
        self.interrupted = False

        self.layers = None
        if self.MLP:
            self.layers = self.getMlpTopology()

        self.ns = ns  # Neighbourhood size of training rasters.
        self.data = None  # Training data
        self.catlist = (
            None  # List of unique output values of the output raster
        )
        self.train_error = None  # Error on training set
        self.val_error = None  # Error on validation set
        self.minValError = (
            None  # The minimum error that is achieved on the validation set
        )
        self.valKappa = 0  # Kappa on on the validation set
        self.sampler = None  # Sampler

        # Results of the MLP prediction
        self.prediction = None  # Raster of the MLP prediction results
        self.confidence = None  # Raster of the MLP results confidence (1 = the maximum confidence, 0 = the least confidence)
        self.transitionPotentials = None  # Dictionary of transition potencial maps: {category1: map1, category2: map2, ...}

        # Outputs of the activation function for small and big numbers
        self.sigmax, self.sigmin = (
            sigmoid(100),
            sigmoid(-100),
        )  # Max and Min of the sigmoid function
        self.sigrange = self.sigmax - self.sigmin  # Range of the sigmoid

    def computeMlpError(self, sample):
        """Get MLP error on the sample"""
        input_data = np.hstack((sample["state"], sample["factors"]))
        out = self.getOutput(input_data)
        err = ((sample["output"] - out) ** 2).sum() / len(out)
        return err

    def computePerformance(self, train_indexes, val_ind):
        """Check errors of training and validation sets
        @param train_indexes     Tuple that contains indexes of the first and last elements of the training set.
        @param val_ind           Tuple that contains indexes of the first and last elements of the validation set.
        """
        train_error = 0
        train_sampl = (
            train_indexes[1] - train_indexes[0]
        )  # Count of training samples
        for i in range(int(train_indexes[0]), int(train_indexes[1])):
            train_error = train_error + self.computeMlpError(
                sample=self.data[i]
            )
        self.setTrainError(train_error / train_sampl)

        if val_ind:
            val_error = 0
            val_sampl = int(val_ind[1]) - int(val_ind[0])
            answers = np.ma.zeros(val_sampl)
            out = np.ma.zeros(val_sampl)
            for i in range(int(val_ind[0]), int(val_ind[1])):
                sample = self.data[i]
                val_error = val_error + self.computeMlpError(
                    sample=self.data[i]
                )

                input_data = np.hstack((sample["state"], sample["factors"]))
                output = self.getOutput(input_data)
                out[i - int(val_ind[0])] = self.outCategory(output)
                answers[i - int(val_ind[0])] = self.outCategory(
                    sample["output"]
                )
            self.setValError(val_error / val_sampl)
            depCoef = DependenceCoef(out, answers, expand=True)
            self.valKappa = depCoef.kappa(mode=None)

    def copyWeights(self):
        """Deep copy of the MLP weights"""
        return copy.deepcopy(self.MLP.weights)

    def createMlp(self, state, factors, output, hidden_layers):
        """@param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        @param hidden_layers    List of neuron counts in hidden layers.
        @param ns               Neighbourhood size.
        """
        if output.getBandsCount() != 1:
            raise MlpManagerError(self.tr("Output layer must have one band!"))

        input_neurons = 0
        for raster in factors:
            input_neurons = input_neurons + raster.getNeighbourhoodSize(
                self.ns
            )

        # state raster contains categories. We need use n-1 dummy variables (where n = number of categories)
        input_neurons = input_neurons + (
            len(state.getBandGradation(1)) - 1
        ) * state.getNeighbourhoodSize(self.ns)

        # Output category's (neuron) list and count
        self.catlist = output.getBandGradation(1)
        categories = len(self.catlist)

        # set neuron counts in the MLP layers
        self.layers = hidden_layers
        self.layers.insert(0, input_neurons)
        self.layers.append(categories)

        self.MLP = MLP(*self.layers)

    def getConfidence(self):
        return self.confidence

    def getInputVectLen(self):
        """Length of input data vector of the MLP"""
        shape = self.getMlpTopology()
        return shape[0]

    def getOutput(self, input_vector):
        out = self.MLP.propagate_forward(input_vector)
        return out

    def getOutputVectLen(self):
        """Length of input data vector of the MLP"""
        shape = self.getMlpTopology()
        return shape[-1]

    def getOutputVector(self, val):
        """Convert a number val into vector,
        for example, let self.catlist = [1, 3, 4] then
        if val = 1, result = [ 1, -1, -1]
        if val = 3, result = [-1,  1, -1]
        if val = 4, result = [-1, -1,  1]
        where -1 is minimum of the sigmoid, 1 is max of the sigmoid
        """
        size = self.getOutputVectLen()
        res = np.ones(size) * (self.sigmin)
        ind = np.where(self.catlist == val)
        res[ind] = self.sigmax
        return res

    def getMinValError(self):
        return self.minValError

    def getMlpTopology(self):
        return self.MLP.shape

    def getKappa(self):
        return self.valKappa

    def getPrediction(self, state, factors, calcTransitions=False):
        self._predict(state, factors, calcTransitions)
        return self.prediction

    def getTrainError(self):
        return self.train_error

    def getTransitionPotentials(self):
        return self.transitionPotentials

    def getValError(self):
        return self.val_error

    def outCategory(self, out_vector):
        # Get index of the biggest output value as the result
        biggest = max(out_vector)
        res = list(out_vector).index(biggest)
        res = self.catlist[res]
        return res

    def outputConfidence(self, output, scale=True):
        """Return confidence (difference between 2 biggest values) of the MLP output.
        @param output: The confidence
        @param scale: If True, then scale the confidence to int [0, 1, ..., 100] percent
        """
        out_scl = self.scaleOutput(output, percent=scale)
        out_scl.sort()
        return out_scl[-1] - out_scl[-2]

    def outputTransitions(self, output, scale=True):
        """Return transition potencial of the outputs scaled to [0,1] or 1-100
        @param output: The output of MLP
        @param scale: If True, then scale the transitions to int ([0, 1, ..., 100]) percent
        """
        out_scl = self.scaleOutput(output, percent=scale)
        result = {}
        for r, v in enumerate(out_scl):
            cat = self.catlist[r]
            result[cat] = v
        return result

    def scaleOutput(self, output, percent=True):
        """Scale the output to range [0,1] or 1-100
        @param output: Output of a MLP
        @param percent: If True, then scale the output to int [0, 1, ..., 100] percent
        """
        res = 1.0 * (output - self.sigmin) / self.sigrange
        if percent:
            res = [int(100 * x) for x in res]
        return res

    def _predict(self, state, factors, calcTransitions=False):
        """Calculate output and confidence rasters using MLP model and input rasters
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        """
        try:
            self.rangeChanged.emit(self.tr("Initialize model %p%"), 1)
            geodata = state.getGeodata()
            rows, cols = geodata["ySize"], geodata["xSize"]
            for r in factors:
                if not state.geoDataMatch(r):
                    raise MlpManagerError(
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
                            out = self.getOutput(input_data)
                            res = self.outCategory(out)
                            predicted_band[i, j] = res

                            confidence = self.outputConfidence(out)
                            confidence_band[i, j] = confidence

                            if calcTransitions:
                                potentials = self.outputTransitions(out)
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
                self.tr("The system is out of memory during ANN prediction")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during ANN prediction")
            )
            raise

    def readMlp(self):
        pass

    def resetErrors(self):
        self.val_error = np.finfo(float).max
        self.train_error = np.finfo(float).max

    def resetMlp(self):
        self.MLP.reset()
        self.resetErrors()

    def saveMlp(self):
        pass

    def saveSamples(self, fileName):
        self.sampler.saveSamples(fileName)

    def setMlpWeights(self, w):
        """Set weights of the MLP"""
        self.MLP.weights = w

    def setTrainingData(
        self, state, factors, output, shuffle=True, mode="All", samples=None
    ):
        """@param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        @param output           Raster that contains categories to predict.
        @param shuffle          Perform random shuffle.
        @param mode             Type of sampling method:
                                    All             Get all pixels
                                    Random          Get samples. Count of samples in the data=samples.
                                    Stratified      Undersampling of major categories and/or oversampling of minor categories.
        @samples                Sample count of the training data (doesn't used in 'All' mode).
        """
        if not self.MLP:
            raise MlpManagerError(self.tr("You must create a MLP before!"))

        # Normalize factors before sampling:
        for f in factors:
            f.normalize(mode="mean")

        self.sampler = Sampler(state, factors, output, self.ns)
        self.sampler.setTrainingData(
            state=state,
            output=output,
            shuffle=shuffle,
            mode=mode,
            samples=samples,
        )

        outputVecLen = self.getOutputVectLen()
        stateVecLen = self.sampler.stateVecLen
        factorVectLen = self.sampler.factorVectLen
        size = len(self.sampler.data)

        state_params = (
            ("state", float, stateVecLen)
            if stateVecLen > 1
            else ("state", float)
        )
        factors_params = (
            ("factors", float, factorVectLen)
            if factorVectLen > 1
            else ("factors", float)
        )
        output_params = (
            ("output", float, outputVecLen)
            if outputVecLen > 1
            else ("output", float)
        )
        self.data = np.zeros(
            size,
            dtype=[
                ("coords", float, 2),
                state_params,
                factors_params,
                output_params,
            ],
        )
        self.data["coords"] = self.sampler.data["coords"]
        self.data["state"] = self.sampler.data["state"]
        self.data["factors"] = self.sampler.data["factors"]
        self.data["output"] = [
            self.getOutputVector(sample["output"])
            for sample in self.sampler.data
        ]

    def setTrainError(self, error):
        self.train_error = error

    def setValError(self, error):
        self.val_error = error

    def setEpochs(self, epochs):
        self.epochs = epochs

    def setValPercent(self, value=20):
        self.valPercent = value

    def setLRate(self, value=0.1):
        self.lrate = value

    def setMomentum(self, value=0.01):
        self.momentum = value

    def setContinueTrain(self, value=False):
        self.continueTrain = value

    @pyqtSlot()
    def startTrain(self):
        """
        Start the training process for the Artificial Neural Network (ANN) model.
        """
        try:
            self.train(
                self.epochs,
                self.valPercent,
                self.lrate,
                self.momentum,
                self.continueTrain,
            )
        except CoeffError as error:
            self.error_occurred.emit(
                self.tr("Model training failed"), str(error)
            )

    @pyqtSlot()
    def stopTrain(self):
        self.interrupted = True

    def train(
        self,
        epochs,
        valPercent=20,
        lrate=0.1,
        momentum=0.01,
        continue_train=False,
    ):
        """Perform the training procedure on the MLP and save the best neural net
        @param epoch            Max iteration count.
        @param valPercent       Percent of the validation set.
        @param lrate            Learning rate.
        @param momentum         Learning momentum.
        @param continue_train   If False then it is new training cycle, reset weights training and validation error. If True, then continue training.
        """
        try:
            samples_count = len(self.data)
            val_sampl_count = samples_count * valPercent / 100
            apply_validation = bool(
                val_sampl_count > 0
            )  # Use or not use validation set
            train_sampl_count = samples_count - val_sampl_count

            # Set first train_sampl_count as training set, the other as validation set
            train_indexes = (0, train_sampl_count)
            val_indexes = (
                (train_sampl_count, samples_count)
                if apply_validation
                else None
            )

            if not continue_train:
                self.resetMlp()
            self.minValError = (
                self.getValError()
            )  # The minimum error that is achieved on the validation set
            last_train_err = self.getTrainError()
            best_weights = self.copyWeights()  # The MLP weights when minimum error that is achieved on the validation set

            update_graph_count = 10
            update_graph_step = (
                epochs // update_graph_count
                if epochs >= update_graph_count
                else 1
            )
            self.rangeChanged.emit(self.tr("Train model %p%"), epochs)
            for _epoch in range(epochs):
                self.trainEpoch(train_indexes, lrate, momentum)
                self.computePerformance(train_indexes, val_indexes)
                self.updateGraphValues.emit(
                    self.getTrainError(), self.getValError()
                )
                if _epoch % update_graph_step == 0:
                    self.updateGraph.emit()
                self.updateDeltaRMS.emit(
                    self.getMinValError() - self.getValError()
                )
                self.updateKappa.emit(self.getKappa())

                QCoreApplication.processEvents()
                if self.interrupted:
                    self.processInterrupted.emit()
                    break

                last_train_err = self.getTrainError()
                self.setTrainError(last_train_err)
                if apply_validation and (
                    self.getValError() < self.getMinValError()
                ):
                    self.minValError = self.getValError()
                    best_weights = self.copyWeights()
                    self.updateMinValErr.emit(self.getMinValError())
                self.updateProgress.emit()

            self.setMlpWeights(best_weights)
        except MemoryError:
            self.errorReport.emit(
                self.tr("The system is out of memory during ANN training")
            )
            raise
        except:
            self.errorReport.emit(
                self.tr("An unknown error occurs during ANN training")
            )
            raise
        finally:
            self.processFinished.emit()

    def trainEpoch(self, train_indexes, lrate=0.1, momentum=0.01):
        """Perform a training epoch on the MLP
        @param train_ind        Tuple of the min&max indexes of training samples in the samples data.
        @param val_ind          Tuple of the min&max indexes of validation samples in the samples data.
        @param lrate            Learning rate.
        @param momentum         Learning momentum.
        """
        train_sampl = train_indexes[1] - train_indexes[0]

        for _i in range(int(train_sampl)):
            n = np.random.randint(*train_indexes)
            sample = self.data[n]
            input_data = np.hstack((sample["state"], sample["factors"]))
            self.getOutput(input_data)  # Forward propagation
            self.MLP.propagate_backward(sample["output"], lrate, momentum)
