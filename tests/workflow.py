from time import clock

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst
from molusce.algorithms.models.crosstabs.manager import CrossTableManager
from molusce.algorithms.models.mlp.manager import MlpManager
from molusce.algorithms.models.simulator.sim import Simulator
from numpy import ma as ma


def main(initRaster, finalRaster, factors):
    print("Start Reading Init Data...", clock())
    initRaster = Raster(initRaster)
    finalRaster = Raster(finalRaster)
    factors = [Raster(rasterName) for rasterName in factors]
    print("Finish Reading Init Data", clock(), "\n")

    print("Start Making CrossTable...", clock())
    crosstab = CrossTableManager(initRaster, finalRaster)
    # print crosstab.getTransitionStat()
    print("Finish Making CrossTable", clock(), "\n")

    # Create and Train Analyst
    print("Start creating AreaAnalyst...", clock())
    analyst = AreaAnalyst(initRaster, finalRaster)
    print("Finish creating AreaAnalyst ...", clock(), "\n")

    print("Start Making Change Map...", clock())
    analyst = AreaAnalyst(initRaster, finalRaster)
    changeMap = analyst.getChangeMap()
    print("Finish Making Change Map", clock(), "\n")

    # ~ # Create and Train ANN Model
    model = MlpManager(ns=1)
    model.createMlp(initRaster, factors, changeMap, [10])
    print("Start Setting MLP Trainig Data...", clock())
    model.setTrainingData(
        initRaster, factors, changeMap, mode="Stratified", samples=1000
    )
    print("Finish Setting Trainig Data", clock(), "\n")
    print("Start MLP Training...", clock())
    model.train(20, valPercent=20)
    print("Finish Trainig", clock(), "\n")

    # print 'Start ANN Prediction...', clock()
    # predict = model.getPrediction(initRaster, factors, calcTransitions=True)
    # confidence = model.getConfidence()
    # potentials = model.getTransitionPotentials()

    # ~ # Create and Train LR Model
    # ~ model = LR(ns=0)
    # ~ print 'Start Setting LR Trainig Data...', clock()
    # ~ model.setState(initRaster)
    # ~ model.setFactors(factors)
    # ~ model.setOutput(changeMap)
    # ~ model.setMode('Stratified')
    # ~ model.setSamples(100)
    # ~ model.setTrainingData()
    # ~ print 'Finish Setting Trainig Data', clock(), '\n'
    # ~ print 'Start LR Training...', clock()
    # ~ model.train()
    # ~ print 'Finish Trainig', clock(), '\n'
    # ~
    # ~ print 'Start LR Prediction...', clock()
    # ~ predict = model.getPrediction(initRaster, factors, calcTransitions=True)
    # ~ print 'Finish LR Prediction...', clock(), '\n'

    # Create and Train WoE Model
    # print 'Start creating AreaAnalyst...', clock()
    # analyst = AreaAnalyst(initRaster, finalRaster)
    # print 'Finish creating AreaAnalyst ...', clock(), '\n'
    # print 'Start creating WoE model...', clock()
    # bins = {0: [[1000, 3000]], 1: [[200, 500, 1500]]}
    # model = WoeManager(factors, analyst, bins= bins)
    # model.train()
    # print 'Finish creating WoE model...', clock(), '\n'

    # ~ # Create and Train MCE Model
    # ~ print 'Start creating MCE model...', clock()
    # ~ matrix = [
    # ~ [1,     6],
    # ~ [1.0/6,   1]
    # ~ ]
    # ~ model = MCE(factors, matrix, 2, 3, analyst)
    # ~ print 'Finish creating MCE model...', clock(), '\n'

    # predict = model.getPrediction(initRaster, factors, calcTransitions=True)
    # confidence = model.getConfidence()
    # potentials = model.getTransitionPotentials()
    # filename = 'predict.tif'
    # confname = 'confidence.tif'
    # trans_prefix='trans_'
    # try:
    #     predict.save(filename)
    #     confidence.save(confname)
    #     if potentials is not None:
    #         for k,v in potentials.iteritems():
    #             map = v.save(trans_prefix+str(k) + '.tif')
    # finally:
    #     os.remove(filename)
    #     #pass
    # print 'Finish Saving...', clock(), '\n'

    # simulation
    print("Start Simulation...", clock())
    simulator = Simulator(initRaster, factors, model, crosstab)
    # Make 1 cycle of simulation:
    simulator.setIterationCount(1)
    simulator.simN()
    monteCarloSim = simulator.getState()  # Result of MonteCarlo simulation
    errors = simulator.errorMap(finalRaster)  # Risk class validation
    riskFunct = simulator.getConfidence()  # Risk function

    try:
        monteCarloSim.save("simulation_result.tiff")
        errors.save("risk_validation.tiff")
        riskFunct.save("risk_func.tiff")
    finally:
        pass
        # os.remove('simulation_result.tiff')
        # os.remove('risk_validation.tiff')
        # os.remove('risk_func.tiff')
    print("Finish Simulation", clock(), "\n")

    print("Done", clock())


if __name__ == "__main__":
    # main('examples/init.tif', 'examples/final.tif', ['examples/dist_river.tif', 'examples/dist_roads.tif'])
    main(
        "Original/Pak_lucc00.tif",
        "Original/Pak_lucc07.tif",
        [
            "Original/dist_main_roads1.tif",
            "Original/dist_rivers1.tif",
            "Original/LPB_dem_res1.tif",
        ],
    )
