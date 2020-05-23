from HERMES import KP, ProblemSet, SubSet

from HH.util import predictionSet, bayesian
from HH import OHNN

import numpy as np

features = ['NORM_MEAN_WEIGHT', 'NORM_MEAN_PROFIT', 'NORM_CORRELATION']
heuristics = ["DEFAULT", "MAX_PROFIT", "MAX_PROFIT/WEIGHT", "MIN_WEIGHT"]

# ProblemSet.Subset
setName = "instances"
seed = 12345
trainingSet = ProblemSet(setName, SubSet.TRAIN, 0.60, seed)
testSet = ProblemSet(setName, SubSet.TEST, 0.60, seed)

train_data, train_labels, train_profits = predictionSet(trainingSet, features, heuristics)
test_data, test_labels, test_profits = predictionSet(testSet, features, heuristics)

print(bayesian(np.concatenate((train_profits, test_profits), axis=0)))
o = OHNN()
# o.load_model()
o.train(train_data, train_profits)

o.evaluate(test_data, test_profits)

print(bayesian(np.concatenate((train_profits, test_profits), axis=0)))
# print(train_data, train_labels, train_profits)
