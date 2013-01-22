#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from molusce.utils import masks_identity, sizes_equal

class CrossTabError(Exception):
    '''Base class for exceptions in this module.'''
    def __init__(self, msg):
        self.msg = msg


class CrossTable:
    '''Class for compute gradations, contingency (cross)table T'''
    def __init__(self, raster1, raster2):
        
        if not sizes_equal(raster1, raster2):
            raise CoeffError('Sizes of rasters not equals!')
        
        raster1, raster2 = masks_identity(raster1, raster2)
        
        X = np.ma.compressed(raster1)
        Y = np.ma.compressed(raster2)
        
        # Compute gradations of the rasters
        self.graduation_x = list(np.unique(np.array(X)))
        self.graduation_y = list(np.unique(np.array(Y)))
        
        rows, cols = len(self.graduation_x), len(self.graduation_y) 
        self.shape = (rows, cols)
        
        # Compute crosstable
        self.T = np.zeros([rows, cols])
        self.n = len(X)                 # Unmasked elements count (= sum of all elements of the table)
        for i in range(self.n):
            class_num_x = self.graduation_x.index(X[i])
            class_num_y = self.graduation_y.index(Y[i])
            self.T[class_num_x][class_num_y] +=1 

    def compute_sum_rows(self):
        '''This function returns sums in the rows (Ti.)'''
        return self.T.sum(axis=1)
        
    def compute_sum_cols(self):
        '''This function returns sums in the cols (T.j)'''
        return self.T.sum(axis=0)
