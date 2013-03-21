import unittest

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
    processFinished = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, state, factors, model, crosstable):
        '''
        @param state            Raster of the current state (classes) values.
        @param factors          List of the factor rasters (predicting variables).
        @param model            Model that is used for predict. The model implements metods:
                                getConfidence(), getPrediction(state, self.factors)
        @param crosstable       Crosstable, contains transition matrix between states T(i,j).
                                The matrix contains number of pixels that are moved
                                from init class i to final class j.
        '''
        QObject.__init__(self)

        self.state = state
        self.factors = factors
        self.predicted = None      # Raster of predicted classes

        self.model  = model
        self.crosstable = crosstable

        self.updatePrediction(self.state)


    def getConfidence(self):
        '''
        Return raster of model's prediction confidence.
        '''
        return self.model.getConfidence()

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
        diff = (a-b).astype(int)
        result = Raster()
        result.create([diff], state.getGeodata())
        return result

    def sim(self):
        '''
        Make 1 iteracion of simulation.
        '''
        #TODO: eleminate AreaAnalyst.getChangeMap() from the process

        transition = self.crosstable.getCrosstable()

        prediction = self.getPrediction()
        state = self.getState()
        new_state = state.getBand(1).copy()         # New states (the result of simulation) will be stored there.
        analyst = AreaAnalyst(state, prediction)
        classes = analyst.classes
        changes = analyst.getChangeMap().getBand(1)

        # Make transition between classes according to
        # number of moved pixel in crosstable
        self.rangeChanged.emit(self.tr("Simulation process %p%"), len(classes)**2 - len(classes))
        for initClass in classes:
            for finalClass in classes:
                if initClass == finalClass: continue

                # TODO: Calculate number of pixels to be moved via TransitoionMatrix and state raster
                n = transition.getTransition(initClass, finalClass)   # Number of pixels to be moved (constant count now).
                # Find n appropriate places for transition initClass -> finalClass
                class_code = analyst.encode(initClass, finalClass)
                places = (changes==class_code)      # Array of places where transitions initClass -> finalClass are occured
                placesCount = np.sum(places)
                if placesCount < n:
                    self.logMessage.emit(self.tr("There are more transitions in the transition matrix, then the model have found"))

                confidence = self.getConfidence().getBand(1)
                confidence = confidence * places # The higher is number in cell, the higer is probability of transition in the cell
                indices = []
                for i in range(n):
                    index = np.unravel_index(confidence.argmax(), confidence.shape)     # Select the cell with biggest probability
                    indices.append(index)
                    confidence[index] = -1       # Mark the cell to prevent second selection

                # Now "indices" contains indices of the appropriate places,
                # make transition initClass -> finalClass
                for index in indices:
                    new_state[index] = finalClass
                self.updateProgress.emit()

        result = Raster()
        result.create([new_state], state.getGeodata())
        self.state = result
        self.updatePrediction(result)
        self.processFinished.emit()


    def simN(self, N):
        '''
        Make N iterations of simulation.
        '''
        for i in range(N):
            self.sim()

    def updatePrediction(self, state):
        '''
        Update prediction using new classes (raster "state")
        '''
        self.predicted = self.model.getPrediction(state, self.factors)

