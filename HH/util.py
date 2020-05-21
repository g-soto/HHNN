from HERMES import ProblemSet, KP, String
import numpy as np

def predictionSet(instances, features, heuristics):
    dataSet = np.empty((instances.getSize(), len(features)+1))
    profits = np.empty(len(heuristics))
    for i_idx,i in enumerate(instances.getFiles()):
        p = KP(i)
        for f_idx,f in enumerate(features):
            dataSet[i_idx][f_idx] = p.getFeature(f)
        for h_idx, h in enumerate(heuristics):
            p = KP(i)
            p.solve(h)
            profits[h_idx] = p.getObjValue()
        dataSet[i_idx][-1] = profits.argmin()
    return dataSet
    