#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from molusce.algorithms.utils import masks_identity, sizes_equal, get_gradations

class CrossTabError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


class CrossTable(object):
    '''Class for compute gradations, contingency (cross)table T'''
    def __init__(self, band1, band2):

        if not sizes_equal(band1, band2):
            raise CrossTabError('Sizes of rasters are not equal!')

        band1, band2 = masks_identity(band1, band2)

        self.X = np.ma.compressed(band1)
        self.Y = np.ma.compressed(band2)

        # Compute gradations of the bands
        self.graduation_x = get_gradations(self.X)
        self.graduation_y = get_gradations(self.Y)

        rows, cols = len(self.graduation_x), len(self.graduation_y)
        self.shape = (rows, cols)

        self._T = None       # Crosstable


    def __computeCrosstable(self):
        # Compute crosstable
        rows, cols = self.shape
        self._T = np.zeros([rows, cols], dtype=int)
        self.n = len(self.X)                 # Count of unmasked elements  (= sum of all elements of the table)
        for i in range(self.n):
            class_num_x = self.graduation_x.index(self.X[i])
            class_num_y = self.graduation_y.index(self.Y[i])
            self._T[class_num_x][class_num_y] +=1

    def getCrosstable(self):
        if self._T == None:
            self.__computeCrosstable()
        return self._T

    def getExpectedProbtable(self):
        '''
        Return expected probabilities table. (if dependencies between X, Y are not present).
        '''
        t = self.getExpectedTable()
        return t/self.n

    def getExpectedTable(self):
        '''
        Return expected crosstable. (if dependencies between X, Y are not present).
        '''
        #compute expected table T*
        #creation array : T*ij = (sum_r[i] * sum_s[j])/ total
        crostable = self.getCrosstable()
        rows, cols = crostable.shape
        sum_rows = self.getSumRows()
        sum_cols = self.getSumCols()
        sum_rows = np.tile(np.reshape(sum_rows, (rows,1)),(1,cols))
        sum_cols = np.tile(sum_cols, (rows,1))
        return 1.0*sum_rows*sum_cols/self.n

    def getProbCols(self):
        return 1.0*self.getSumCols() / self.n

    def getProbRows(self):
        return 1.0*self.getSumRows() / self.n

    def getProbtable(self):
        '''
        Return probability table of transitions
        '''
        return 1.0*self.getCrosstable() / self.n

    def getSumRows(self):
        '''This function returns sums in the rows (Ti.)'''
        crosstable = self.getCrosstable()
        return crosstable.sum(axis=1)

    def getSumCols(self):
        '''This function returns sums in the cols (T.j)'''
        crosstable = self.getCrosstable()
        return crosstable.sum(axis=0)

    def getTransition(self, fromClass, toClass):
        '''
        Return number of transitions from "fromClass" to "toClass"
        '''
        i = self.graduation_x.index(fromClass)
        j = self.graduation_y.index(toClass)
        crosstable = self.getCrosstable()
        return crosstable[i,j]


