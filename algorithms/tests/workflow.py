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
from molusce.algorithms.models.woe.manager import WoeManager
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
    
    #~ # Create and Train ANN Model
    #~ model = MlpManager(ns=0)
    #~ model.createMlp(initRaster, factors, finalRaster, [10])
    #~ print 'Start Setting MLP Trainig Data...', clock()
    #~ model.setTrainingData(initRaster, factors, finalRaster, mode='Balanced', samples=1000)    
    #~ print 'Finish Setting Trainig Data', clock(), '\n'
    #~ print 'Start MLP Training...', clock()
    #~ model.train(1000, valPercent=20)
    #~ print 'Finish Trainig', clock(), '\n'
    #~ 
    #~ print 'Start ANN Prediction...', clock()
    #~ predict = model.getPrediction(initRaster, factors)
    #~ filename = 'ann_predict.tiff'
    #~ try:
        #~ predict.save(filename)
    #~ finally:
        #~ #os.remove(filename)
        #~ pass
    #~ print 'Finish ANN Prediction...', clock(), '\n'

    #~ # Create and Train LR Model
    #~ model = LR(ns=1)
    #~ print 'Start Setting LR Trainig Data...', clock()
    #~ model.setTrainingData(initRaster, factors, finalRaster, mode='Balanced', samples=1000)
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

    # Create and Train WoE Model
    print 'Start creating AreaAnalyst...', clock()
    analyst = AreaAnalyst(initRaster, finalRaster)
    print 'Finish creating AreaAnalyst ...', clock(), '\n'
    print 'Start creating WoE model...', clock()
    bins = {0: [[1000, 2000, 3000]], 1: [[200, 500, 1000, 1500]]}
    model = WoeManager(factors, analyst, bins= bins)
    print 'Finish creating WoE model...', clock(), '\n'

    # simulation
    print 'Start Simulation...', clock()
    simulator = Simulator(initRaster, factors, model, crosstab)
    # Make 1 cycle of simulation:
    simulator.sim()
    monteCarloSim   = simulator.getState()              # Result of MonteCarlo simulation
    errors          = simulator.errorMap(finalRaster)   # Risk class validation
    riskFunct       = simulator.getConfidence()         # Risk function
    
    # Make K cycles of simulation:
    # simulator.simN(K)
    try:
        monteCarloSim.save('simulation_result.tiff')
        errors.save('risk_validation.tiff')
        riskFunct.save('risk_func.tiff')
    finally:
        pass
        # os.remove('simulation_result.tiff')
        # os.remove('risk_validation.tiff')
        # os.remove('risk_func.tiff')
    print 'Finish Simulation', clock(), '\n'
    
    print 'Done', clock()
    
    
if __name__=="__main__":
    main('examples/init.tif', 'examples/final.tif', ['examples/dist_river.tif', 'examples/dist_roads.tif'])
    
    
