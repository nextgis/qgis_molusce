#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt4.QtCore import *

import math
import numpy as np
from numpy import ma as ma

from molusce.algorithms.utils import masks_identity, sizes_equal
from molusce.algorithms.models.crosstabs.model import CrossTable

from time import sleep

class CoeffError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

class DependenceCoef(QObject):

    rangeChanged = pyqtSignal(str, int)
    updateProgress = pyqtSignal()
    processFinished = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, X, Y):

        QObject.__init__(self)

        self.X = X
        self.Y = Y

        self.crosstable = None

    def getCrosstable(self):
        if self.crosstable == None:
            self.calculateCrosstable()
        return self.crosstable

    def calculateCrosstable(self):
        self.rangeChanged.emit('Initialization...', 2)
        self.updateProgress.emit()
        self.crosstable = CrossTable(self.X, self.Y)
        self.updateProgress.emit()
        self.__propagateCrossTableSignals()
        self.crosstable.computeCrosstable()
        self.processFinished.emit()


    def correlation(self):
        '''
        Define correlation coefficient of the rasters.
        @param X    First raster's array
        @param Y    Second raster's array
        '''
        X, Y = masks_identity(self.X, self.Y)
        R = np.corrcoef(np.ma.compressed(X),np.ma.compressed(Y))
        # function np.corrcoef return array of coefficients
        # R[0][0] = R[1][1] = 1.0 - correlation X--X and Y--Y
        # R[0][1] = R[1][0] - correlation X--Y and Y--X

        return R[0][1]

    def correctness(self):
        """
        % of correctness
        """
        table = self.getCrosstable()
        crosstable = table.getCrosstable()
        rows, cols = table.shape
        if rows != cols:
            raise CoeffError('The method is applicable for NxN crosstable only!')
        n = table.n
        s = 0
        for i in range(rows):
            s = s+ crosstable[i][i]

        return 100.0*s/n

    def cramer(self):
        '''
        Define Cramer's relationship coefficient of the rasters for discrete values
        Coefficient change between [0, 1]
        0 - no dependence
        1 - full connection
        @param X    First raster's array
        @param Y    Second raster's array
        '''
        table = self.getCrosstable()
        crosstable = table.getCrosstable()
        rows, cols = table.shape
        t_expect =  table.getExpectedTable()

        # Mask T* to prevent division by zero
        t_expect = np.ma.array(t_expect, mask=(t_expect == 0))
        # chi-square coeff = sum((T-T*)^2/T*)
        x2 = np.sum(np.square(crosstable - t_expect)/t_expect)
        # CRAMER CONTINGENCY COEF. = sqrt(chi-square / (total * min(s-1,r-1)))
        # s, r - raster grauations
        Cramer = math.sqrt(x2/(table.n*min(cols-1, rows-1)))

        return Cramer

    def jiu(self):
        '''
        Define Joint Information Uncertainty coef., based on entropy., for discrete values
        Coefficient change between [0, 1]
        0 - no connection
        1 - full connection
        @param X    First raster's array
        @param Y    Second raster's array
        '''
        #T, sum_r, sum_s, total, r, s = compute_table(X, Y)
        table = self.getCrosstable()
        T = table.getProbtable()             #Pij = Tij / total
        sum_rows = table.getProbRows()       #Pi. = Ti. / total  i=[0,(r-1)]
        sum_cols = table.getProbCols()       #P.j = T.j / total  j=[0,(s-1)]

        #to calculate the entropy we take the logarithm,
        #logarithm of zero does not exist, so we must mask zero values
        sum_rows = np.compress(sum_rows != 0, sum_rows)
        sum_cols = np.compress(sum_cols != 0, sum_cols)
        #Compute the entropy coeff. of two raster
        H_x = -np.sum(sum_rows * np.log(sum_rows))
        H_y = -np.sum(sum_cols * np.log(sum_cols))
        #Compute the joint entropy coeff.
        T = np.ma.array(T, mask=(T == 0))
        T = np.ma.compressed(T)
        H_xy = -np.sum(T * np.log(T))
        # Compute the Joint Information Uncertainty
        U = 2.0 * ((H_x + H_y - H_xy)/(H_x + H_y))

        return U

    def kappa(self, mode=None):
        '''
        Kappa statistic
        @param X    Raster array.
        @param Y    Raster array.
        @param mode Kappa sttistic to compute:
            mode = None:    classic kappa
            mode = loc:     kappa location
            mode = histo    kappa histogram
        '''
        table = self.getCrosstable()
        rows, cols = table.shape
        if rows != cols:
            raise CoeffError('Kappa is applicable for NxN crosstable only!')
        t_expect =  table.getProbtable()
        pa = 0
        for i in range(rows):
            pa = pa + t_expect[i,i]
        prows = table.getProbRows()
        pcols = table.getProbCols()
        pexpect = sum(prows * pcols)
        pmax = sum(np.min([prows, pcols], axis=0))

        if mode == None:
            result = (pa - pexpect)/(1-pexpect)
        elif mode == "loc":
            result = (pa - pexpect)/(pmax - pexpect)
        elif mode == "histo":
            result = (pmax - pexpect)/(1 - pexpect)
        elif mode == "all":
            result = {"loc": (pa - pexpect)/(pmax - pexpect), "histo": (pmax - pexpect)/(1 - pexpect), "overal": (pa - pexpect)/(1-pexpect)}
        else:
            raise CoeffError('Unknown mode of kappa statistics!')

        return result

    def __propagateCrossTableSignals(self):
        self.crosstable.rangeChanged.connect(self.__crosstableProgressRangeChanged)
        self.crosstable.updateProgress.connect(self.__crosstableProgressChanged)
        self.crosstable.crossTableFinished.connect(self.__crosstableFinished)

    def __crosstableFinished(self):
        self.crosstable.rangeChanged.disconnect(self.__crosstableProgressRangeChanged)
        self.crosstable.updateProgress.disconnect(self.__crosstableProgressChanged)
        self.crosstable.crossTableFinished.disconnect(self.__crosstableFinished)
    def __crosstableProgressChanged(self):
        self.updateProgress.emit()

    def __crosstableProgressRangeChanged(self, message, maxValue):
        self.rangeChanged.emit(message, maxValue)


