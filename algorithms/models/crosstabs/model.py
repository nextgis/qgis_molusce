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

        X = np.ma.compressed(band1)
        Y = np.ma.compressed(band2)

        # Compute gradations of the rasters
        self.graduation_x = get_gradations(X)
        self.graduation_y = get_gradations(Y)

        rows, cols = len(self.graduation_x), len(self.graduation_y)
        self.shape = (rows, cols)

        # Compute crosstable
        self.T = np.zeros([rows, cols], dtype=int)
        self.n = len(X)                 # Count of unmasked elements  (= sum of all elements of the table)
        for i in range(self.n):
            class_num_x = self.graduation_x.index(X[i])
            class_num_y = self.graduation_y.index(Y[i])
            self.T[class_num_x][class_num_y] +=1

    def getCrosstable(self):
        return self.T

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
        rows, cols = self.T.shape
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
        return self.T.sum(axis=1)

    def getSumCols(self):
        '''This function returns sums in the cols (T.j)'''
        return self.T.sum(axis=0)

    def getTransition(self, fromClass, toClass):
        '''
        Return number of transitions from "fromClass" to "toClass"
        '''
        i = self.graduation_x.index(fromClass)
        j = self.graduation_y.index(toClass)
        return self.T[i,j]


