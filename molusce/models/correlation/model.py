#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy import ma as ma

class CoeffError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg

def size_equals(X, Y):
    '''
    Define dimensions equality of the rasters
    @param X    First array
    @param Y    Second array
    '''
    x1, y1 = np.shape(X)
    x2, y2 = np.shape(Y)
    return (x1,y1) == (x2,y2)

def resize(X):
    '''
    Change raster's shape to 1D-dimension (m*n,1),
    because function numpy.corrcoef are compute the correlation
    coefficient for 1D-array's rasters
    @param X    First raster's array
    @param Y    Second raster's array
    '''
    sh = np.shape(X)                 #shape=(m,n)
    X = np.resize(X, (sh[0]*sh[1],)) #X.shape=(m*n,1)
    return X

def correlation(X, Y):
    '''
    Define correlation coefficient of the two rasters for continuous 
    values. Coefficient change between [-1, 1]
    @param X    First raster's array
    @param Y    Second raster's array
    ''' 
    X = resize(X)
    Y = resize(Y)
    R = np.corrcoef(X,Y)   
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
    T, sum_r, sum_s, total, r, s = compute_table(X, Y)
    #compute expected contingency table T*
    #creation array : T*ij = (sum_r[i] * sum_s[j])/ total
    sum_r = np.tile(np.reshape(sum_r, (r,1)),(1,s))
    sum_s = np.tile(sum_s,(r,1))
    T_expect = np.zeros([r, s])
    T_expect = np.divide(np.multiply(sum_r, sum_s),total)
    # masked T*, because forbid to divide by zero
    T_expect = np.ma.array(T_expect, mask=(T_expect == 0))
    # chi-square coeff = sum((T-T*)^2/T*)
    x2 = np.sum(np.divide(np.square(np.subtract(T, T_expect)), T_expect))
    # CRAMER CONTINGENCY COEF. = sqrt(chi-square / total * min(s-1,r-1))
    # s, r - raster grauations
    Cramer = math.sqrt(x2/(total*min(s-1,r-1)))   

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
    T, sum_r, sum_s, total, r, s = compute_table(X, Y)
    
    T = np.divide(T, total)         #Pij = Tij / total
    sum_r = np.divide(sum_r, total) #Pi. = Ti. / total  i=[0,(r-1)]
    sum_s = np.divide(sum_s, total) #P.j = T.j / total  j=[0,(s-1)]
    
    #to calculate the entropy we take the logarithm,
    #logarithm of zero does not exist, so we must masked zero values
    sum_r = np.compress(sum_r != 0, sum_r) 
    sum_s = np.compress(sum_s != 0, sum_s)
    #Compute the entropy coeff. of two raster 
    H_x = -np.sum(np.multiply(sum_r,np.log(sum_r)))
    H_y = -np.sum(np.multiply(sum_s,np.log(sum_s)))
    #Compute the joint entropy coeff.
    T = np.ma.array(T, mask=(T == 0))
    T = np.ma.compressed(T)
    H_xy = -np.sum(np.multiply(T, np.log(T))) 
    # Compute the Joint Information Uncertainty 
    U = 2 * ((H_x + H_y - H_xy)/(H_x + H_y))
    
    return U  
     
def compute_table(X, Y):
    '''
    This function computes: 
    1. contingency table T
    2. list sum_r : SUM_j (Tij),  j=[0, ..., (s-1)]
    3. list sum_s : SUM_i (Tij),  i=[0, ..., (r-1)]
    4. number r of gradations for raster X
    5. number s of gradations for raster Y
    @param X    First array
    @param Y    Second array   
    '''
    if size_equals(X, Y):
        m, n = np.shape(X)
        # empty gradations list  creation
        graduation_x = []
        graduation_y = []
        #compute gradations
        graduation_x = list(np.unique(np.array(X)))
        graduation_y = list(np.unique(np.array(Y)))
        #compute gradations lenght
        r = len(graduation_x)
        s = len(graduation_y)
        #compute contingency table T
        T = np.zeros([r,s])        
        for k in range(r):
            for col in range(n):
                for rows in range(m): 
                    if X[rows][col] == graduation_x[k]:   
                        T[k][graduation_y.index(Y[rows][col])] +=1 
        total = np.sum(T)       #T..
        sum_r = T.sum(axis=1)   #Ti.
        sum_s = T.sum(axis=0)   #T.j
        return T, sum_r, sum_s, total, r, s
    else:
        raise CoeffError('Sizes of rasters not equals!')
    
    
    
