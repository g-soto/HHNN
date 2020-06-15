from HERMES import KP, ProblemSet, SubSet, NNHH

from HH.util import predictionSet, bayesian
from HH import OHNN

import numpy as np

import pickle as pk

features = ["NORM_MEAN_WEIGHT", "NORM_MEAN_PROFIT", "NORM_MEAN_PROFIT_WEIGHT", "NORM_MEDIAN_WEIGHT", "NORM_MEDIAN_PROFIT", "NORM_MEDIAN_PROFIT_WEIGHT", "NORM_STD_WEIGHT", "NORM_STD_PROFIT", "NORM_STD_PROFIT_WEIGHT", "NORM_CORRELATION"]

heuristics = ["DEFAULT", "MAX_PROFIT", "MAX_PROFIT/WEIGHT", "MIN_WEIGHT", "MARKOVITZ"]

# ProblemSet.Subset
setName = "instances\\training"
setNameA = "instances\\testA"
setNameB = "instances\\testB"

seed = 12345
trainingSet = ProblemSet(setName)
testSetA = ProblemSet(setNameA)
testSetB = ProblemSet(setNameB)


train_data, train_labels, train_profits = predictionSet(trainingSet, features, heuristics)
test_dataA, test_labelsA, test_profitsA = predictionSet(testSetA, features, heuristics)
test_dataB, test_labelsB, test_profitsB = predictionSet(testSetB, features, heuristics)

o = OHNN()

#comment this line...
o.load_model()

#...and uncoment this two to perfrom the trainning.
# print(bayesian(train_profits))
# o.train(train_data, train_profits)

print(bayesian(train_profits))
print(bayesian(test_profitsA))
print(bayesian(test_profitsB))
o.evaluate(train_data, train_profits)
o.evaluate(test_dataA, test_profitsA)
o.evaluate(test_dataB, test_profitsB)

print('####Using it as HH####')
problem = KP()
HH = NNHH(o, heuristics, features)

a = np.array(problem.solve(testSetA, [HH]).split())
a = a.reshape(-1,2)

resultsA = np.concatenate((a[1:,:],test_profitsA), axis=1)

b = np.array(problem.solve(testSetB, [HH]).split())
b = b.reshape(-1,2)

resultsB = np.concatenate((b[1:,:],test_profitsB), axis=1)


with open('resultA.dat','wb') as fa, open('resultB.dat','wb') as fb:
    pk.dump(resultsA, fa)
    pk.dump(resultsB, fb)

print(resultsA)
print(resultsB)
