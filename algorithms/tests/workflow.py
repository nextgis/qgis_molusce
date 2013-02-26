# encoding: utf-8

import sys
import os
from time import clock
sys.path.insert(0, '../../../')

import numpy as np
from numpy import ma as ma

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.crosstabs.manager  import CrossTableManager
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst
from molusce.algorithms.models.lr.lr import LR
from molusce.algorithms.models.mlp.manager import MlpManager
from molusce.algorithms.models.simulator.sim import Simulator

def main(initRaster, finalRaster, factors):
    print 'Start Reading Init Data...', clock()
    initRaster = Raster(initRaster)
    finalRaster = Raster(finalRaster)
    factors = [Raster(rasterName) for rasterName in factors]
    print 'Finish Reading Init Data', clock(), '\n'
    
    print "Start Making CrossTable...", clock()
    crosstab = CrossTableManager(initRaster, finalRaster)
    print "Finish Making CrossTable", clock(), '\n'
    
    
    print 'Start Making Change Map...', clock()
    analyst = AreaAnalyst(initRaster,finalRaster)
    changeMapRaster = analyst.makeChangeMap()
    filename = 'tempChangeMap.tiff'
    try:
        changeMapRaster.save(filename)
    finally:
        #os.remove(filename)
        pass
    print 'Finish Making Change Map', clock(), '\n'
    
    
    #~ # Create and Train LR Model
    #~ model = LR(ns=1)
    #~ print 'Start Setting LR Trainig Data...', clock()
    #~ model.setTrainingData(initRaster, factors, finalRaster, mode='Balanced', samples=10000)
    #~ print 'Finish Setting Trainig Data', clock(), '\n'
    #~ print 'Start LR Training...', clock()
    #~ model.train()
    #~ print 'Finish Trainig', clock(), '\n'
    #~ 
    #~ 
    #~ print 'Start LR Prediction...', clock()
    #~ predict = model.getPrediction(initRaster, factors)
    #~ filename = 'lr_predict.tiff'
    #~ try:
        #~ predict.save(filename)
    #~ finally:
        #~ #os.remove(filename)
        #~ pass
    #~ print 'Finish LR Prediction...', clock(), '\n'
    
    
    # Create and Train ANN Model
    model = MlpManager(ns=1)
    model.createMlp(initRaster, factors, finalRaster, [10])
    print 'Start Setting MLP Trainig Data...', clock()
    model.setTrainingData(initRaster, factors, finalRaster, mode='Normal', samples=10000)
    print 'Finish Setting Trainig Data', clock(), '\n'
    print 'Start MLP Training...', clock()
    model.train(50, valPercent=20)
    print 'Finish Trainig', clock(), '\n'

    #~ print 'Start ANN Prediction...', clock()
    #~ predict = model.getPrediction(initRaster, factors)
    #~ filename = 'ann_predict.tiff'
    #~ try:
        #~ predict.save(filename)
    #~ finally:
        #~ #os.remove(filename)
        #~ pass
    #~ print 'Finish ANN Prediction...', clock(), '\n'
    
    print 'Start Simulation...', clock()
    simulator = Simulator(initRaster, factors, model, crosstab)
    simulator.sim()
    state = simulator.getState()
    filename = 'simulation_result.tiff'
    try:
        state.save(filename)
    finally:
        #os.remove(filename)
        pass
    print 'Finish Simulation', clock(), '\n'
    
    print 'Done', clock()
    
    
if __name__=="__main__":
    main('examples/init.tif', 'examples/final.tif', ['examples/dist_river.tif', 'examples/dist_roads.tif'])
    
    
