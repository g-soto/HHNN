from HERMES import KP, ProblemSet, SubSet

from HH.util import predictionSet
from HH import OHNN

features = ['NORM_MEAN_WEIGHT', 'NORM_MEAN_PROFIT', 'NORM_CORRELATION']
heuristics = ["DEFAULT", "MAX_PROFIT", "MAX_PROFIT/WEIGHT", "MIN_WEIGHT"]

# ProblemSet.Subset
setName = "instances"
seed = 12345
trainingSet = ProblemSet(setName, SubSet.TRAIN, 0.60, seed)
testSet = ProblemSet(setName, SubSet.TEST, 0.60, seed)

train_data, train_labels = predictionSet(trainingSet, features, heuristics)
test_data, test_labels = predictionSet(trainingSet, features, heuristics)

o = OHNN()
o.train(train_data, train_labels)
print(o.evaluate(test_data, test_labels))
