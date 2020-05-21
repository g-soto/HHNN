from HERMES import KP, ProblemSet, SubSet

from HH.util import predictionSet

features = ['NORM_MEAN_WEIGHT', 'NORM_MEAN_PROFIT', 'NORM_CORRELATION']
heuristics = ["DEFAULT", "MAX_PROFIT", "MAX_PROFIT/WEIGHT", "MIN_WEIGHT"]

# ProblemSet.Subset
setName = "instances"
seed = 12345
trainingSet = ProblemSet(setName, SubSet.TRAIN, 0.60, seed)
testSet = ProblemSet(setName, SubSet.TEST, 0.60, seed)

print(predictionSet(trainingSet, features, heuristics))
