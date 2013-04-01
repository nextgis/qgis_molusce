#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy import ma as ma

from molusce.algorithms.utils import masks_identity, sizes_equal
from molusce.algorithms.models.crosstabs.model import CrossTable


class CoeffError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg




def correlation(X, Y):
    '''
    Define correlation coefficient of the rasters.
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    X, Y = masks_identity(X, Y)
    R = np.corrcoef(np.ma.compressed(X),np.ma.compressed(Y))
    # function np.corrcoef return array of coefficients
    # R[0][0] = R[1][1] = 1.0 - correlation X--X and Y--Y
    # R[0][1] = R[1][0] - correlation X--Y and Y--X
    return R[0][1]

def cramer(X, Y):
    '''
    Define Cramer's relationship coefficient of the rasters for discrete values
    Coefficient change between [0, 1]
    0 - no dependence
    1 - full connection
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    table = CrossTable(X, Y)
    rows, cols = table.shape
    t_expect =  table.getExpectedCrosstable()

    # Mask T* to prevent division by zero
    t_expect = np.ma.array(t_expect, mask=(t_expect == 0))
    # chi-square coeff = sum((T-T*)^2/T*)
    x2 = np.sum(np.square(table.T - t_expect)/t_expect)
    # CRAMER CONTINGENCY COEF. = sqrt(chi-square / (total * min(s-1,r-1)))
    # s, r - raster grauations
    Cramer = math.sqrt(x2/(table.n*min(cols-1, rows-1)))

    return Cramer


def jiu(X, Y):
    '''
    Define Joint Information Uncertainty coef., based on entropy., for discrete values
    Coefficient change between [0, 1]
    0 - no connection
    1 - full connection
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    #T, sum_r, sum_s, total, r, s = compute_table(X, Y)
    table = CrossTable(X, Y)
    T = 1.0*table.T / table.n                         #Pij = Tij / total
    sum_rows = 1.0*table.getSumRows() / table.n       #Pi. = Ti. / total  i=[0,(r-1)]
    sum_cols = 1.0*table.getSumCols() / table.n       #P.j = T.j / total  j=[0,(s-1)]

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








