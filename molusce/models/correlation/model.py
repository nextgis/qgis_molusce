#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy import ma as ma

from molusce.utils import masks_identity

class CoeffError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

def size_equals(X, Y):
    '''
    Define equality dimensions of the two rasters
    @param X    First raster's array
    @param Y    Second raster's array
    '''

    return (np.shape(X) == np.shape(Y))

def correlation(X, Y):
    '''
    Define correlation coefficient of the two rasters for continuous 
    values.
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
    Define relationship coefficient of two rasters for discrete values
    Coefficient change between [0, 1]
    0 - no connection
    1 - full connection
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    #T, sum_r, sum_s, total, r, s = compute_table(X, Y)
    table = compute_table(X, Y)
    #compute expected contingency table T*
    #creation array : T*ij = (sum_r[i] * sum_s[j])/ total
    sum_r = np.tile(np.reshape(table.compute_sum_r(), (table.compute_r(),1)),(1,table.compute_s()))
    sum_s = np.tile(table.compute_sum_s(),(table.compute_r(),1))
    T_expect = sum_r*sum_s/table.compute_total()
    
    # masked T*, because forbid to divide by zero
    T_expect = np.ma.array(T_expect, mask=(T_expect == 0))
    # chi-square coeff = sum((T-T*)^2/T*)
    x2 = np.sum(np.square(table.T - T_expect)/T_expect)
    # CRAMER CONTINGENCY COEF. = sqrt(chi-square / total * min(s-1,r-1))
    # s, r - raster grauations
    Cramer = math.sqrt(x2/(table.compute_total()*min(table.compute_s()-1,table.compute_r()-1)))   

    return Cramer    

def jiu(X, Y): 
    '''
    Define relationship coef., based on entropy., for discrete values
    Coefficient change between [0, 1]
    0 - no connection
    1 - full connection
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    #T, sum_r, sum_s, total, r, s = compute_table(X, Y)
    table = compute_table(X, Y)
    T = np.divide(table.T, table.compute_total())     #Pij = Tij / total
    sum_r = np.divide(table.compute_sum_r(), table.compute_total()) #Pi. = Ti. / total  i=[0,(r-1)]
    sum_s = np.divide(table.compute_sum_s(), table.compute_total()) #P.j = T.j / total  j=[0,(s-1)]
    
    #to calculate the entropy we take the logarithm,
    #logarithm of zero does not exist, so we must masked zero values
    sum_r = np.compress(sum_r != 0, sum_r) 
    sum_s = np.compress(sum_s != 0, sum_s)
    #Compute the entropy coeff. of two raster 
    H_x = -np.sum(sum_r * np.log(sum_r))
    H_y = -np.sum(sum_s * np.log(sum_s))
    #Compute the joint entropy coeff.
    T = np.ma.array(T, mask=(T == 0))
    T = np.ma.compressed(T)
    H_xy = -np.sum(T * np.log(T)) 
    # Compute the Joint Information Uncertainty 
    U = 2 * ((H_x + H_y - H_xy)/(H_x + H_y))
    
    return U  
     
def compute_table(X, Y):
    '''
    @param X    First raster's array
    @param Y    Second raster's array   
    '''
    if not size_equals(X, Y):
        raise CoeffError('Sizes of rasters not equals!')
    X, Y = masks_identity(X, Y)
    table = CoeffTable(X, Y)
    
    table.T = np.zeros([table.compute_r(), table.compute_s()])
    for i in range(table.n):         
        table.T[table.graduation_x.index(table.X[i])][table.graduation_y.index(table.Y[i])] +=1 
    
    return table

class CoeffTable:
    '''class for compute gradations, contingency table T'''
    def __init__(self, raster1, raster2):
        
        self.X = np.ma.compressed(raster1)
        self.Y = np.ma.compressed(raster2)
        self.n = len(self.X)
        # Compute gradations
        self.graduation_x = list(np.unique(np.array(self.X)))
        self.graduation_y = list(np.unique(np.array(self.Y)))
        self.T = np.array([0])
        
    def compute_r(self):
        '''This function return gradations lenghts for first rasters'''
        return   len(self.graduation_x)
         
    def compute_s(self):
        '''This function return gradations lenghts for second rasters'''
        return   len(self.graduation_y) 
     
    def compute_total(self):
        '''This function return T..'''
        return np.sum(self.T)
        
    def compute_sum_r(self):
        '''This function return Ti.'''
        return self.T.sum(axis=1)
        
    def compute_sum_s(self):
        '''This function return T.j'''
        return self.T.sum(axis=0)
