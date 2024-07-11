import os
from time import clock

import numpy as np
from numpy import ma as ma

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.crosstabs.manager  import CrossTableManager
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst
from molusce.algorithms.models.lr.lr import LR
from molusce.algorithms.models.mlp.manager import MlpManager
from molusce.algorithms.models.woe.manager import WoeManager
from molusce.algorithms.models.simulator.sim import Simulator
from molusce.algorithms.models.mce.mce import MCE


def main(initRaster, finalRaster, factors):
    print('Start Reading Init Data...', clock())
    initRaster = Raster(initRaster)
    finalRaster = Raster(finalRaster)
    factors = [Raster(rasterName) for rasterName in factors]
    print('Finish Reading Init Data', clock(), '\n')

    print("Start Making CrossTable...", clock())
    crosstab = CrossTableManager(initRaster, finalRaster)
    print("Finish Making CrossTable", clock(), '\n')

    # Create and Train LR Model
    model = LR(ns=1)
    print('Start Setting LR Trainig Data...', clock())
    model.setTrainingData(initRaster, factors, finalRaster, mode='Stratified', samples=1000)
    print('Finish Setting Trainig Data', clock(), '\n')
    print('Start LR Training...', clock())
    model.train()
    print('Finish Trainig', clock(), '\n')

    print('Start LR Prediction...', clock())
    predict = model.getPrediction(initRaster, factors)
    filename = 'lr_predict.tiff'
    try:
        predict.save(filename)
    finally:
        os.remove(filename)
    print('Finish LR Prediction...', clock(), '\n')

    # simulation
    print('Start Simulation...', clock())
    simulator = Simulator(initRaster, factors, model, crosstab)
    # Make 1 cycle of simulation:
    simulator.simN(1)
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
        os.remove('simulation_result.tiff')
        os.remove('risk_validation.tiff')
        os.remove('risk_func.tiff')
    print('Finish Simulation', clock(), '\n')

    print('Done', clock())


if __name__=="__main__":
    main('examples/init.tif', 'examples/final.tif', ['examples/dist_river.tif', 'examples/dist_roads.tif'])
