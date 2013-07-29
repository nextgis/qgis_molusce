import numpy as np

from PyQt4.QtCore import *

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.mlp.manager import MlpManager
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst

class Simulator(QObject):
    """
    Based on a model, controls simulation via cellular automaton
    over a number of cycles
    """

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    simFinished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorReport = pyqtSignal(str)

    def __init__(self, state, factors, model, crosstable):
        '''
        @param state            Raster of the current state (categories) values.
        @param factors          List of the factor rasters (predicting variables).
        @param model            Model that is used for predict. The model mast implement next methods:
                                    getConfidence(),
                                    getPrediction(state, factors,calcTransitions=False),
                                    getTransitionPotentials()
        @param crosstable       Crosstable, contains transition matrix between states T(i,j).
                                The matrix contains number of pixels that are moved
                                from init category i to final category j.
        '''
        QObject.__init__(self)

        self.state = state
        self.factors = factors
        self.iterationCount = 1     # Count of simulation iterations
        self.predicted = None      # Raster of predicted categories

        self.model  = model
        self.crosstable = crosstable
        self.calcTransitions = False

        try:    # Not all models have the signals
            self.model.rangeChanged.connect(self.__modelProgressRangeChanged)
            self.model.updateProgress.connect(self.__modelProgressChanged)

            self.model.errorReport.connect(self.__modelErrorReport)
        except AttributeError:
            pass

    def getConfidence(self):
        '''
        Return raster of model's prediction confidence.
        '''
        return self.model.getConfidence()

    def getTransitionPotentials(self):
        return self.model.getTransitionPotentials()

    def getPrediction(self):
        '''
        Predict new states via model.
        '''
        return self.predicted

    def getState(self):
        return self.state

    def errorMap(self, answer):
        '''
        Create map of correct and incorrect prediction.
        This function compares the known answer and the result of predicting procedure,
        correct pixel is marked as 0.
        '''
        state = self.getState()
        b = state.getBand(1)
        a = answer.getBand(1)
        diff = (a-b).astype(np.int16)
        result = Raster()
        result.create([diff], state.getGeodata())
        return result

    def __modelProgressRangeChanged(self, message, maxValue):
        self.rangeChanged.emit(message, maxValue)
        QCoreApplication.processEvents()

    def __modelProgressChanged(self):
        self.updateProgress.emit()
        QCoreApplication.processEvents()

    def __modelErrorReport(self, message):
        self.errorReport.emit(message)
        QCoreApplication.processEvents()

    def setCalcTransitions(self, calcTransitions):
        self.calcTransitions = calcTransitions

    def setIterationCount(self, Count):
        self.iterationCount = Count

    def __sim(self):
        '''
        1 iteracion of simulation.
        '''
        transition = self.crosstable.getCrosstable()

        self.updatePrediction(self.state)
        changes = self.getPrediction().getBand(1)   # Predicted change map
        state = self.getState()
        new_state = state.getBand(1).copy().astype(np.uint8)    # New states (the result of simulation) will be stored there.

        self.rangeChanged.emit(self.tr("Area Change Analysis %p%"), 2)
        self.updateProgress.emit()
        QCoreApplication.processEvents()
        analyst = AreaAnalyst(state, second = None)
        self.updateProgress.emit()
        QCoreApplication.processEvents()

        categories = state.getBandGradation(1)

        # Make transition between categories according to
        # number of moved pixel in crosstable
        self.rangeChanged.emit(self.tr("Simulation process %p%"), len(categories)**2 - len(categories))
        QCoreApplication.processEvents()
        for initClass in categories:
            for finalClass in categories:
                if initClass == finalClass: continue

                # TODO: Calculate number of pixels to be moved via TransitoionMatrix and state raster
                n = transition.getTransition(initClass, finalClass)   # Number of pixels that have to be
                                                                      # changed the categories
                                                                      # (use TransitoionMatrix only).
                if n==0:
                    continue
                # Find n appropriate places for transition initClass -> finalClass
                cat_code = analyst.encode(initClass, finalClass)
                places = (changes==cat_code)      # Array of places where transitions initClass -> finalClass are occured
                placesCount = np.sum(places)
                if placesCount < n:
                    self.logMessage.emit(self.tr("There are more transitions in the transition matrix, then the model have found"))
                    QCoreApplication.processEvents()
                    n = placesCount
                if n >0:
                    confidence = np.ma.filled(confidence, 0)
                    ind = confidence.argsort(axis=None)[-n:]
                    indices = [np.unravel_index(i, confidence.shape) for i in ind]

                    # Now "indices" contains indices of the appropriate places,
                    # make transition initClass -> finalClass
                    for index in indices:
                        new_state[index] = finalClass
                self.updateProgress.emit()
                QCoreApplication.processEvents()

        result = Raster()
        result.create([new_state], state.getGeodata())
        self.state = result

    def simN(self):
        '''
        Make N iterations of simulation.
        '''
        try:
            for i in range(self.iterationCount):
                self.__sim()
        except MemoryError:
            self.errorReport.emit(self.tr("The system out of memory during simulation"))
            raise
        except:
            self.errorReport.emit(self.tr("An unknown error occurs during simulation"))
            raise
        finally:
            try:    # Not all models have the signals
                self.model.rangeChanged.disconnect(self.__modelProgressRangeChanged)
                self.model.updateProgress.disconnect(self.__modelProgressChanged)
            except AttributeError:
                pass
            self.simFinished.emit()
            QCoreApplication.processEvents()

    def updatePrediction(self, state):
        '''
        Update prediction using new categories (raster "state")
        '''
        self.predicted = self.model.getPrediction(state, self.factors, calcTransitions=self.calcTransitions)

